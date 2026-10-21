"""Validation results (14 Nov): TF-IDF, embedding baseline and the trained GNN, all segments.

    uv run python -m recsys.experiments.validation_gnn <mlflow run id of the GNN>

The GNN is scored with the 9-13 Nov scoring graph (the deployed setting) and, as a one-off
check, with the 9-11 Nov training graph. Segments (known users, seen articles) are always
defined against 9-13 Nov, so all rows are comparable. Also reports paired bootstraps on
validation (seed 0 only; informative: the win criterion of ADR 009 applies to test).
Writes reports/gnn_validation.csv and reports/gnn_validation_bootstrap.csv.
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

    order = ["all", "test-mix weighted", "user=known", "user=new", "history=empty", "article=seen", "article=unseen"]
    for metric in ("auc", "ndcg@10"):
        print(f"\n{metric}\n" + results.pivot(index="segment", columns="model", values=metric)
              .loc[order, list(scores)].round(4).to_string())
    print("\n" + boot[boot["metric"] == "auc"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
