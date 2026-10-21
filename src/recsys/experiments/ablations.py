"""Ablation round on validation (ADR 009). Seed 0 first; borderline results get 3 seeds.

    uv run python -m recsys.experiments.ablations reference <run id>   # scores of the reference GNN
    uv run python -m recsys.experiments.ablations train A1 A2 ... [--seed 0]          # round 1
    uv run python -m recsys.experiments.ablations train A1 ... REF --epochs 9 [--seed 1] # round 2
    uv run python -m recsys.experiments.ablations report

Round 1 used early stopping; round 2 trains every variant for a fixed number of epochs
(ADR 009). "REF" trains the reference model itself (needed for seeds 1-2 and other lengths).

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


def key(ablation_id: str, seed: int, epochs: int | None) -> str:
    return f"{ablation_id}_seed{seed}" + (f"_e{epochs}" if epochs else "")


def run_ablation(ablation_id: str, seed: int, epochs: int | None = None) -> None:
    variant = Variant() if ablation_id == "REF" else ABLATIONS[ablation_id][0]
    cfg = load_model_config()
    model, best_epoch, run_id = train("validation", seed, epochs=epochs, cfg=cfg, variant=variant,
                                      run_name=f"ablation-{key(ablation_id, seed, epochs)}")
    scores = score_model(model.eval(), variant, validation_eval_split(), cfg)
    save(key(ablation_id, seed, epochs), scores,
         {"run_id": run_id, "best_epoch": best_epoch, "variant": variant.name, "fixed_epochs": epochs})
    print(f"{key(ablation_id, seed, epochs)}: epochs {best_epoch}, run {run_id}", flush=True)


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
    round1 = "\n\n".join([
        "## Round 1: early stopping (seed 0)",
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
        "All segments and nDCG@10 for round 1: `reports/gnn_ablations.csv`.",
    ])
    round2 = report_round2(split, epochs=9)
    (REPORTS_DIR / "gnn_ablations.md").write_text(
        "# GNN ablations (validation, 14 Nov)\n\n" + round2 + "\n\n" + round1 + "\n")
    print(open(REPORTS_DIR / "gnn_ablations.md").read())


def _load(k: str):
    p = SCORES_DIR / f"{k}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _compare(a: pd.DataFrame, b: pd.DataFrame, split, segments) -> dict:
    boot = paired_bootstrap(a, b, split, metrics=("auc",))
    return {seg: boot[boot["segment"] == seg].iloc[0] for seg in segments}


def _reference(seed: int, epochs: int):
    """Reference at a fixed length: the round-1 seed-0 run is its own epoch-9 state."""
    ref_meta = json.loads((SCORES_DIR / "reference_seed0.json").read_text())
    if seed == 0 and epochs == ref_meta["best_epoch"]:
        return _load("reference_seed0")
    return _load(key("REF", seed, epochs))


def decide(aid: str, seed0: dict, three_seed: dict | None) -> str:
    """ADR 009 decision rules. 'Alternative' = the ablation variant; difference = variant - reference."""
    def outcome(seg):
        if three_seed is not None and seg in three_seed:
            diffs, r0 = three_seed[seg]
            same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            if same_sign and (r0["ci_low"] > 0 or r0["ci_high"] < 0):
                return "better" if r0["diff"] > 0 else "worse"
            return "noise"
        r = seed0[seg]
        verdict = classify(r["ci_low"], r["ci_high"])
        if verdict == "clear effect":
            return "better" if r["diff"] > 0 else "worse"
        return "noise" if verdict == "no effect" else "borderline"

    _, deciding = ABLATIONS[aid]
    if aid == "R1":
        unseen, mix = outcome("article=unseen"), outcome("test-mix weighted")
        if unseen == "better" and mix != "worse":
            return "adopt R1 (helps unseen articles, test-mix not clearly hurt)"
        return f"keep current design (unseen: {unseen}, test-mix: {mix})"
    o = outcome(deciding[0])
    return {"better": "alternative clearly better: adopt it",
            "worse": "clearly worse: keep current design",
            "noise": "within noise: keep current design (no measurable effect)",
            "borderline": "borderline: needs 3 seeds"}[o]


def seed_variance_section(split, epochs: int) -> str:
    """Training-seed spread of the reference and of the 3-seed variants (validation AUC)."""
    segs = ["test-mix weighted", "article=unseen", "article=seen", "all"]
    def aucs(scores):
        return evaluate(scores, split, model="m").set_index("segment").loc[segs, "auc"]
    ref = [aucs(_reference(s, epochs)) for s in (0, 1, 2) if _reference(s, epochs) is not None]
    if len(ref) < 3:
        return ""
    ref = pd.DataFrame(ref, index=["seed 0", "seed 1", "seed 2"])
    lines = [f"| reference {i} | " + " | ".join(f"{v:.4f}" for v in r) + " |" for i, r in ref.iterrows()]
    lines.append("| **reference mean** | " + " | ".join(f"**{v:.4f}**" for v in ref.mean()) + " |")
    lines.append("| reference range (max − min) | " + " | ".join(f"{v:.4f}" for v in ref.max() - ref.min()) + " |")
    for aid in ABLATIONS:
        runs = [_load(key(aid, s, epochs)) for s in (0, 1, 2)]
        if all(r is not None for r in runs):
            m = pd.DataFrame([aucs(r) for r in runs]).mean()
            lines.append(f"| {aid} mean of 3 seeds | " + " | ".join(f"{v:.4f}" for v in m) + " |")
    spread = (ref.max() - ref.min())["test-mix weighted"]
    return ("### Training-seed variance\n\n"
            f"| model | {' | '.join(segs)} |\n|---" + "|---" * len(segs) + "|\n" + "\n".join(lines) + "\n\n"
            f"The reference's spread across 3 seeds is **{spread:.4f} AUC on the test-mix weighted row**, larger "
            "than most ablation effects. **Seed 0 is the strongest of the three reference runs**, so seed-0 "
            "comparisons are made against an unusually good reference and are biased against the variants. The "
            "bootstrap interval covers impression sampling only, not training variance, so a seed-0 \"clear "
            "effect\" smaller than about this spread (here A1 on seen articles, A3 and A4 on test-mix) may be "
            "partly seed luck. A6 (−0.024) and A7 (−0.011 on known users, consistently negative in round 1 too) "
            "are well beyond it.")


def report_round2(split, epochs: int) -> str:
    ref = _reference(0, epochs)
    rows, decisions = [], []
    for aid, (variant, deciding) in ABLATIONS.items():
        a = _load(key(aid, 0, epochs))
        if a is None:
            continue
        segs = list(dict.fromkeys(deciding + ["test-mix weighted"]))
        seed0 = _compare(a, ref, split, segs)
        three = {}
        for seg in deciding:
            others = [(_load(key(aid, s, epochs)), _reference(s, epochs)) for s in (1, 2)]
            if all(x is not None and y is not None for x, y in others):
                diffs = [seed0[seg]["diff"]] + [_compare(x, y, split, [seg])[seg]["diff"] for x, y in others]
                three[seg] = (diffs, seed0[seg])
        for seg in segs:
            r = seed0[seg]
            seeds = (" / ".join(f"{d:+.4f}" for d in three[seg][0]) if seg in three else "")
            rows.append(f"| {aid} | {variant.name[3:]} | {seg}{' (deciding)' if seg in deciding else ''} | "
                        f"{r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                        f"{classify(r['ci_low'], r['ci_high'])} | {seeds} |")
        decisions.append(f"| {aid} | {variant.name[3:]} | {decide(aid, seed0, three or None)} |")

    a8_note = ""
    a8_long, ref_long = _load("A8_seed0"), _reference(0, 13)
    if a8_long is not None and ref_long is not None:
        r = _compare(a8_long, ref_long, split, ["test-mix weighted"])["test-mix weighted"]
        a8_note = ("### A8 at its own length (13 epochs)\n\n"
                   "A8 (round-1 run, best epoch 13) against the reference trained for a fixed 13 epochs, "
                   f"test-mix weighted AUC: **{r['diff']:+.4f}** [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
                   f"{classify(r['ci_low'], r['ci_high'])}.")
    return "\n\n".join(filter(None, [
        f"## Round 2: fixed {epochs} epochs, no early stopping (ADR 009)",
        f"Every variant trains for exactly {epochs} epochs and is compared with the reference at epoch {epochs} "
        "(paired bootstrap, AUC, 1,000 resamples; difference = variant − reference). Borderline deciding "
        "results are rerun with seeds 1 and 2 for both variant and reference; *3 seeds* lists the seed 0 / 1 / 2 "
        "differences. Win criterion: same sign in all three seeds and seed 0's interval excludes zero.",
        "| id | ablation | segment | difference (seed 0) | 95% CI | verdict | 3 seeds |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows),
        a8_note,
        seed_variance_section(split, epochs),
        "### Decisions (ADR 009 rules)\n\n| id | ablation | decision |\n|---|---|---|\n" + "\n".join(decisions),
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reference", "train", "report"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="fixed epochs (round 2); default: early stopping")
    a = parser.parse_args()
    if a.command == "reference":
        run_reference(a.args[0])
    elif a.command == "train":
        for aid in a.args:
            run_ablation(aid, a.seed, a.epochs)
    else:
        report()


if __name__ == "__main__":
    main()
