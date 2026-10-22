"""Validation results (14 Nov): TF-IDF, embedding baseline and the trained GNN, all segments.

    uv run python -m recsys.experiments.validation_gnn <mlflow run id of the GNN>

The GNN is scored with the 9-13 Nov scoring graph (the deployed setting) and, as a one-off
check, with the 9-11 Nov training graph. Segments (known users, seen articles) are always
defined against 9-13 Nov, so all rows are comparable. Also reports paired bootstraps on
validation (seed 0 only; informative: the win criterion of ADR 009 applies to test).
Writes reports/gnn_validation.csv, reports/gnn_validation_bootstrap.csv and
reports/gnn_validation.md (tables, scoring-graph decision, and the findings in FINDINGS).
"""

import sys

import pandas as pd
import torch

from recsys.evaluation import evaluate, paired_bootstrap
from recsys.experiments.compare_baselines import frozen_tfidf_params
from recsys.experiments.run_baselines import load_catalogue
from recsys.graph.periods import load_period
from recsys.models.baselines import EmbeddingContent, TfidfContent
from recsys.paths import REPORTS_DIR
from recsys.training.train_gnn import MODELS_DIR, Recommender, Scorer, validation_eval_split


FINDINGS = """## Findings so far

- **Better text features give the largest gain.** The frozen MiniLM embedding baseline beats
  TF-IDF by **+0.049 AUC** overall (95% CI 0.045 to 0.052) and +0.057 on unseen articles. That
  is more than twice the gain the graph adds on top.
- **The GNN's gain over the embedding baseline comes from training on clicks, not from the graph.**
  GNN versus embedding is +0.022 AUC overall and +0.024 on the test-mix weighted row (seed 0),
  driven by **+0.037 on seen articles**. But the trained no-graph model A0 gets the same gain
  (A0 vs embedding: +0.015 test-mix, +0.037 seen, all 3 seeds), and **GNN vs A0 is −0.000 on
  test-mix** (−0.007 to +0.010 across seeds, signs mixed). On validation the graph adds nothing
  measurable (`reports/gnn_ablations.md`, round 3). Users with an empty history also gain: +0.066 from the learned
  default vector, but on only 160 impressions.
- **On unseen articles the GNN is worse than the embedding baseline.** Across 3 training seeds
  the difference is **−0.012 on average** (−0.022 to −0.003), and two of the three seeds'
  intervals exclude zero. The first single-seed runs (−0.006 and −0.003) understated it.
  **Cause (corrected after ablation A0):** it is *not* message passing. The trained no-graph model
  A0, which has no message passing, shows the same deficit against the embedding baseline
  (−0.013 on unseen articles, all 3 seeds below zero). Training on clicks shifts the ranking
  towards the kinds of articles clicked during training, at the expense of new ones. An earlier
  version of this section blamed message passing; the A0 result rules that out.
- **The overall gain is smaller than seed 0 suggests.** Test-mix weighted: +0.024 at seed 0,
  **+0.015 averaged over 3 seeds** (+0.008 to +0.024). Seed 1's interval only just excludes zero.
  On seen articles the gain is robust (+0.030 to +0.037 in every seed).
- **Rebuilding the scoring graph daily makes no measurable difference** (see Scoring graph above;
  ADR 008).
- **Training:** early stopping picks epoch 9 (stopped at 11, patience 2). Seed-0 reruns are not
  bit-identical (pyg-lib neighbour sampling), but differ by about 0.001 AUC per epoch."""

ORDER = ["all", "test-mix weighted", "user=known", "user=new", "history=empty", "article=seen", "article=unseen"]


def scoring_graph_decision(boot: pd.DataFrame) -> str:
    """Pre-registered rule: switch to the training graph only if its advantage on the
    test-mix weighted AUC has a 95% interval excluding zero; else keep the rebuilt graph."""
    r = boot[(boot["model_a"] == "gnn") & (boot["model_b"] == "gnn (9-11 Nov graph)")
             & (boot["segment"] == "test-mix weighted") & (boot["metric"] == "auc")].iloc[0]
    # diff = rebuilt - training graph; the training graph's advantage is -diff.
    adv, lo, hi = -r["diff"], -r["ci_high"], -r["ci_low"]
    switch = lo > 0
    return (f"Training-graph advantage on test-mix weighted AUC: {adv:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}). "
            + ("The interval excludes zero, so **switch to scoring with the training graph**."
               if switch else "The interval includes zero, so **keep the rebuilt scoring graph** (as pre-registered in ADR 008)."))


def markdown(results: pd.DataFrame, boot: pd.DataFrame, models: list[str], header: str, seeds: str = "") -> str:
    def table(metric):
        t = results.pivot(index="segment", columns="model", values=metric).loc[ORDER, models]
        n = results[results["model"] == models[0]].set_index("segment")["n_impressions"].loc[ORDER]
        rows = [f"| {seg} | {n[seg]:,} | " + " | ".join(f"{v:.4f}" for v in t.loc[seg]) + " |" for seg in ORDER]
        return "\n".join([f"| segment | impressions | {' | '.join(models)} |", "|---" * (len(models) + 2) + "|", *rows])

    b = boot[boot["metric"] == "auc"]
    brows = [f"| {r.model_a} vs {r.model_b} | {r.segment} | {r.diff:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
             f"{'yes' if r.significant else 'no'} |" for r in b.itertuples()]
    return "\n\n".join([
        "# GNN validation results (14 Nov)",
        header,
        "## AUC\n\n" + table("auc"),
        "## nDCG@10\n\n" + table("ndcg@10"),
        "## Paired bootstrap (AUC, seed 0, 1,000 resamples)\n\n"
        "Informative only: the ADR 009 win criterion is applied on test.\n\n"
        "| comparison | segment | difference | 95% CI | excludes 0 |\n|---|---|---|---|---|\n" + "\n".join(brows),
        "## Scoring graph (ADR 008)\n\n" + scoring_graph_decision(boot),
        seeds,
        FINDINGS,
    ]).strip() + "\n"


SEED_RUNS = ["reference_seed0", "REF_seed1_e9", "REF_seed2_e9"]  # GNN at 9 epochs, seeds 0-2 (ablation runs)


def seed_comparison_section(split, embedding_scores: pd.DataFrame) -> str:
    """GNN vs embedding baseline for each of 3 training seeds (paired bootstrap, AUC)."""
    from recsys.experiments.ablations import SCORES_DIR

    segs = ["test-mix weighted", "all", "article=seen", "article=unseen"]
    per_seed = []
    for seed, k in enumerate(SEED_RUNS):
        boot = paired_bootstrap(pd.read_parquet(SCORES_DIR / f"{k}.parquet"), embedding_scores, split,
                                "gnn", "embedding", metrics=("auc",)).set_index("segment")
        per_seed.append(boot.loc[segs])
    rows = []
    for seg in segs:
        diffs = [b.loc[seg, "diff"] for b in per_seed]
        cis = " · ".join(f"seed {i}: {b.loc[seg, 'diff']:+.4f} [{b.loc[seg, 'ci_low']:+.4f}, {b.loc[seg, 'ci_high']:+.4f}]"
                         for i, b in enumerate(per_seed))
        rows.append(f"| {seg} | {diffs[0]:+.4f} | **{sum(diffs) / 3:+.4f}** | {min(diffs):+.4f} to {max(diffs):+.4f} | {cis} |")
    return ("## GNN vs embedding baseline across 3 training seeds\n\n"
            "Each seed's GNN (9 epochs, the ablation reference runs) is compared with the embedding baseline by "
            "paired bootstrap (AUC). Seed 0 is the strongest of the three runs, so its difference is the most "
            "favourable; the mean is the better estimate of what the GNN adds (ADR 009).\n\n"
            "| segment | seed 0 | mean of 3 seeds | min to max | per seed: difference [95% CI] |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows))


def load_model(run_id: str):
    saved = torch.load(MODELS_DIR / f"{run_id}.pt", weights_only=False)
    data, _ = load_period("validation_train")
    model = Recommender(data, saved["cfg"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved


def main(run_id: str) -> None:
    split = validation_eval_split()
    news = load_catalogue()
    model, saved = load_model(run_id)
    cfg = saved["cfg"]

    scores = {
        "tfidf": TfidfContent(**frozen_tfidf_params()).fit(split.train, news).score(split.target),
        "embedding": EmbeddingContent(cfg["user"]["max_history"]).fit(split.train, news).score(split.target),
    }
    for label, period in (("gnn", "validation_score"), ("gnn (9-11 Nov graph)", "validation_score_on_train_graph")):
        graph, ids = load_period(period)
        scores[label] = Scorer(split, graph, ids, cfg["user"]["max_history"], cfg["gnn"]["fanout"]).scores(model)

    results = pd.concat([evaluate(s, split, model=m) for m, s in scores.items()], ignore_index=True)
    results.insert(2, "notes", f"gnn run {run_id}, best epoch {saved['best_epoch']}, seed {saved['seed']}")
    results.to_csv(REPORTS_DIR / "gnn_validation.csv", index=False)

    boot = pd.concat([
        paired_bootstrap(scores["gnn"], scores["embedding"], split, "gnn", "embedding"),
        paired_bootstrap(scores["embedding"], scores["tfidf"], split, "embedding", "tfidf"),
        paired_bootstrap(scores["gnn"], scores["gnn (9-11 Nov graph)"], split, "gnn", "gnn (9-11 Nov graph)"),
    ], ignore_index=True)
    boot.to_csv(REPORTS_DIR / "gnn_validation_bootstrap.csv", index=False)
    header = (f"GNN: MLflow run `{run_id}`, seed {saved['seed']}, best epoch {saved['best_epoch']} "
              f"(max {cfg['train']['max_epochs']}, patience {cfg['train']['patience']}). "
              "Generated by `uv run python -m recsys.experiments.validation_gnn <run id>`.")
    (REPORTS_DIR / "gnn_validation.md").write_text(markdown(results, boot, list(scores), header,
                                                           seed_comparison_section(split, scores["embedding"])))
    print(scoring_graph_decision(boot))

    order = ORDER
    for metric in ("auc", "ndcg@10"):
        print(f"\n{metric}\n" + results.pivot(index="segment", columns="model", values=metric)
              .loc[order, list(scores)].round(4).to_string())
    print("\n" + boot[boot["metric"] == "auc"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
