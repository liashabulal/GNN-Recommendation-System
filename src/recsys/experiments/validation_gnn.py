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
- **The graph's gain comes entirely from seen articles.** GNN versus embedding is +0.022 AUC
  overall and +0.024 on the test-mix weighted row, driven by **+0.037 on seen articles**,
  where co-click edges exist. Users with an empty history also gain: +0.066 from the learned
  default vector, but on only 160 impressions.
- **On unseen articles the GNN is slightly worse than the embedding baseline.** It was −0.006
  (95% CI excluding zero) on the first 10-epoch run and −0.003 (CI −0.008 to +0.003, including
  zero) on the 20-epoch rerun. For articles with no co-click edges, message passing so far
  adds nothing over their own text, and may dilute it. **Investigate with ablations A2
  (entity edges) and A3 (subcategory nodes) before any architecture change.**
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


def markdown(results: pd.DataFrame, boot: pd.DataFrame, models: list[str], header: str) -> str:
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
        FINDINGS,
    ]).strip() + "\n"


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
    (REPORTS_DIR / "gnn_validation.md").write_text(markdown(results, boot, list(scores), header))
    print(scoring_graph_decision(boot))

    order = ORDER
    for metric in ("auc", "ndcg@10"):
        print(f"\n{metric}\n" + results.pivot(index="segment", columns="model", values=metric)
              .loc[order, list(scores)].round(4).to_string())
    print("\n" + boot[boot["metric"] == "auc"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
