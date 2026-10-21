"""Ablation round on validation (ADR 009). Seed 0 first; borderline results get 3 seeds.

    uv run python -m recsys.experiments.ablations reference <run id>   # scores of the reference GNN
    uv run python -m recsys.experiments.ablations train A1 A2 ... [--seed 0]
    uv run python -m recsys.experiments.ablations report

Each ablation is trained with early stopping (best epoch per run), scored on 14 Nov with its
own variant of the 9-13 Nov scoring graph, and compared with the reference GNN by paired
bootstrap (1,000 resamples) on its deciding segment(s) and the test-mix weighted row.
Writes reports/gnn_ablations.md and reports/gnn_ablations.csv.
"""

import argparse
import json

import pandas as pd
import torch

from recsys.evaluation import evaluate, paired_bootstrap
from recsys.models.gnn import load_model_config
from recsys.paths import PROCESSED_DIR, REPORTS_DIR
from recsys.training.train_gnn import MODELS_DIR, Recommender, Scorer, train, validation_eval_split
from recsys.training.variants import ABLATIONS, Variant

SCORES_DIR = PROCESSED_DIR / "ablations"


def score_model(model, variant: Variant, split, cfg) -> pd.DataFrame:
    graph, ids = variant.graph("validation_score")
    return Scorer(split, graph, ids, cfg["user"]["max_history"], cfg["gnn"]["fanout"],
                  variant.history_clicks).scores(model)


def save(key: str, scores: pd.DataFrame, meta: dict) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_DIR / f"{key}.parquet", index=False)
    (SCORES_DIR / f"{key}.json").write_text(json.dumps(meta, indent=2))


def run_reference(run_id: str) -> None:
    from recsys.graph.periods import load_period

    saved = torch.load(MODELS_DIR / f"{run_id}.pt", weights_only=False)
    model = Recommender(load_period("validation_train")[0], saved["cfg"])
    model.load_state_dict(saved["state_dict"])
    scores = score_model(model.eval(), Variant(), validation_eval_split(), saved["cfg"])
    save(f"reference_seed{saved['seed']}", scores, {"run_id": run_id, "best_epoch": saved["best_epoch"]})


def run_ablation(ablation_id: str, seed: int) -> None:
    variant, _ = ABLATIONS[ablation_id]
    cfg = load_model_config()
    model, best_epoch, run_id = train("validation", seed, cfg=cfg, variant=variant,
                                      run_name=f"ablation-{ablation_id}-seed{seed}")
    scores = score_model(model.eval(), variant, validation_eval_split(), cfg)
    save(f"{ablation_id}_seed{seed}", scores, {"run_id": run_id, "best_epoch": best_epoch, "variant": variant.name})
    print(f"{ablation_id} seed {seed}: best epoch {best_epoch}, run {run_id}", flush=True)


def classify(lo: float, hi: float) -> str:
    """ADR 009: clear / no effect / borderline, from the 95% interval [lo, hi]."""
    w = hi - lo
    if lo > 0 or hi < 0:
        return "clear effect" if min(abs(lo), abs(hi)) > w / 4 else "borderline"
    return "no effect" if lo + w / 4 <= 0 <= hi - w / 4 else "borderline"


def reference_curve(run_id: str) -> pd.Series:
    """Reference model's validation test-mix weighted AUC per epoch (from MLflow)."""
    import os

    import mlflow

    from recsys.paths import PROJECT_ROOT

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    history = mlflow.MlflowClient().get_metric_history(run_id, "val_auc_test-mix_weighted")
    return pd.Series({m.step: m.value for m in history}).sort_index()


def report() -> None:
    split = validation_eval_split()
    ref = pd.read_parquet(SCORES_DIR / "reference_seed0.parquet")
    ref_meta = json.loads((SCORES_DIR / "reference_seed0.json").read_text())
    rows, segment_rows = [], []
    for aid, (variant, deciding) in ABLATIONS.items():
        path = SCORES_DIR / f"{aid}_seed0.parquet"
        if not path.exists():
            continue
        meta = json.loads((SCORES_DIR / f"{aid}_seed0.json").read_text())
        scores = pd.read_parquet(path)
        boot = paired_bootstrap(scores, ref, split, model_a=aid, model_b="reference", metrics=("auc", "ndcg@10"))
        segment_rows.append(boot)
        for seg in dict.fromkeys(deciding + ["test-mix weighted"]):
            r = boot[(boot["segment"] == seg) & (boot["metric"] == "auc")].iloc[0]
            rows.append({"id": aid, "ablation": variant.name, "segment": seg, "deciding": seg in deciding,
                         "best_epoch": meta["best_epoch"], "diff": r["diff"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"], "verdict": classify(r["ci_low"], r["ci_high"])})
    table = pd.DataFrame(rows)
    curve = reference_curve(ref_meta["run_id"])
    # Undertraining confound: how far below its own best was the reference at the ablation's best epoch?
    # n/a when the ablation trained past the reference's last epoch (no undertraining concern).
    table["ref_gap_at_epoch"] = table["best_epoch"].map(lambda e: curve.get(e, float("nan")) - curve.max())
    pd.concat(segment_rows).to_csv(REPORTS_DIR / "gnn_ablations.csv", index=False)

    lines = [f"| {r.id} | {r.ablation[3:]} | {r.segment}{' (deciding)' if r.deciding else ''} | {r.best_epoch} | "
             f"{'n/a' if pd.isna(r.ref_gap_at_epoch) else f'{r.ref_gap_at_epoch:+.4f}'} | {r.diff:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.verdict} |"
             for r in table.itertuples()]
    confounded = table.drop_duplicates("id")
    confounded = confounded[confounded["ref_gap_at_epoch"] <= -0.005]
    borderline = table[table["deciding"] & (table["verdict"] == "borderline")]
    (REPORTS_DIR / "gnn_ablations.md").write_text("\n\n".join([
        "# GNN ablations (validation, 14 Nov, seed 0)",
        f"Reference: the current GNN (MLflow run `{ref_meta['run_id']}`, best epoch {ref_meta['best_epoch']}). "
        "Each ablation is trained with early stopping and compared with the reference by paired bootstrap "
        "(AUC, 1,000 resamples). **difference = ablation − reference**: negative means the removed or "
        "changed component was helping.",
        "Verdict (ADR 009), with w the interval width: **clear effect** = interval excludes 0 by more than "
        "w/4; **no effect** = 0 lies in the middle half of the interval; **borderline** = anything else, "
        "which gets 3 seeds before a decision.",
        "| id | ablation | segment | best epoch | reference at that epoch vs its best | difference | 95% CI | verdict |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(lines),
        "**Borderline on a deciding segment (to rerun with 3 seeds):** "
        + (", ".join(f"{r.id} ({r.segment})" for r in borderline.itertuples()) or "none"),
        "## Caveat: early stopping confounds several results\n\n"
        "Validation AUC is noisy from epoch to epoch, and with patience 2 a single dip ends a run. "
        f"{len(confounded)} of {table['id'].nunique()} ablations stopped with their best epoch at "
        f"{', '.join(sorted({str(e) for e in confounded['best_epoch']}))} (the reference's best epoch is "
        f"{ref_meta['best_epoch']}). The column *reference at that epoch vs its best* shows how far below its "
        "own final score the reference was at the same epoch (test-mix weighted AUC). Where that gap is as "
        "large as the ablation's difference, the comparison mixes *component removed* with *trained for "
        "fewer epochs*. Affected: "
        + ", ".join(f"{r.id} (best epoch {r.best_epoch}, reference gap {r.ref_gap_at_epoch:+.4f})"
                    for r in confounded.itertuples())
        + ".\n\nThese verdicts, especially on the test-mix weighted row, should not be used to choose the "
        "final configuration until the runs are made comparable in training length.",
        "All segments and nDCG@10: `reports/gnn_ablations.csv`.",
    ]) + "\n")
    print(open(REPORTS_DIR / "gnn_ablations.md").read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reference", "train", "report"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    if a.command == "reference":
        run_reference(a.args[0])
    elif a.command == "train":
        for aid in a.args:
            run_ablation(aid, a.seed)
    else:
        report()


if __name__ == "__main__":
    main()
