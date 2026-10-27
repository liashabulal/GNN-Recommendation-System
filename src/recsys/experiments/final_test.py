"""Part G: final test run (15 Nov, MIND dev) per ADR 008/009.

    uv run python -m recsys.experiments.final_test train [--seeds 0 1 2 3 4]
    uv run python -m recsys.experiments.final_test report

Models (frozen configuration, fixed 9 epochs = best validation epoch, refit on the test setup:
training graph 9-12 Nov, examples 13-14 Nov, scoring graph 9-14 Nov):
    gnn        the GNN (ADR 007)
    a0         the same model trained with no edges (ADR 009)
    embedding  frozen MiniLM embeddings, user = mean of history embeddings (untrained)

Comparisons (paired bootstrap per seed, seed k vs seed k; vs embedding: seed k vs the one
untrained baseline): primary GNN vs A0, secondary GNN vs embedding and A0 vs embedding.
Win criterion (ADR 009): every seed's 95% interval for the AUC difference on all test
impressions excludes zero, all on the same side.

Beyond-accuracy metrics on each impression's top 5 (ties broken at random, fixed seed):
    freshness exposure  share of top-5 slots given to unseen articles, vs their share of clicks
    coverage            share of distinct candidate articles in at least one top 5
    diversity           mean number of distinct subcategories in a top 5
"""

import argparse
import dataclasses
import json

import numpy as np
import pandas as pd

from recsys.data.mind import explode_impressions
from recsys.evaluation import evaluate, paired_bootstrap
from recsys.experiments.run_baselines import load_catalogue
from recsys.graph.build import load_graph
from recsys.graph.periods import period_split
from recsys.models.baselines import EmbeddingContent
from recsys.models.gnn import load_model_config
from recsys.paths import PROCESSED_DIR, REPORTS_DIR
from recsys.training.train_gnn import Scorer, train
from recsys.training.variants import COMPARISONS, Variant

SCORES_DIR = PROCESSED_DIR / "test_scores"
EPOCHS = 9  # best validation epoch (ADR 008)
SEEDS = [0, 1, 2, 3, 4]
TRAINED = {"gnn": Variant(), "a0": COMPARISONS["A0"]}
TOP_K = 5
DECIDING_SEGMENT = "all"  # on test, "all" is the test user mix (test-mix weighted is a validation proxy)


def test_split():
    """15 Nov scored against the 9-14 Nov period (dev impression IDs offset by 1,000,000)."""
    return dataclasses.replace(period_split("test_score"), name="test")


def run_train(seeds: list[int]) -> None:
    cfg = load_model_config()
    split = test_split()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for name, variant in TRAINED.items():
            model, _, run_id = train("test", seed, epochs=EPOCHS, cfg=cfg, variant=variant,
                                     run_name=f"final-test-{name}-seed{seed}")
            graph, ids = variant.graph("test_score")
            scores = Scorer(split, graph, ids, cfg["user"]["max_history"], cfg["gnn"]["fanout"],
                            variant.history_clicks).scores(model.eval())
            scores.to_parquet(SCORES_DIR / f"{name}_seed{seed}.parquet", index=False)
            (SCORES_DIR / f"{name}_seed{seed}.json").write_text(json.dumps({"run_id": run_id, "epochs": EPOCHS}))
            print(f"{name} seed {seed}: run {run_id}", flush=True)


def embedding_scores(split) -> pd.DataFrame:
    path = SCORES_DIR / "embedding.parquet"
    if not path.exists():
        cfg = load_model_config()
        EmbeddingContent(cfg["user"]["max_history"]).fit(split.train, load_catalogue()).score(split.target) \
            .to_parquet(path, index=False)
    return pd.read_parquet(path)


def top_k_metrics(scores: pd.DataFrame, split, subcategory: pd.Series, seed: int = 0) -> dict:
    cands = explode_impressions(split.target)[["impression_id", "news_id", "clicked"]]
    df = cands.merge(scores, on=["impression_id", "news_id"], validate="one_to_one")
    df["unseen"] = ~df["news_id"].isin(split.seen_news)
    df["tiebreak"] = np.random.default_rng(seed).random(len(df))
    df = df.sort_values(["impression_id", "score", "tiebreak"], ascending=[True, False, True])
    top = df[df.groupby("impression_id").cumcount() < TOP_K]
    top = top.assign(subcategory=top["news_id"].map(subcategory))
    clicks = df[df["clicked"] == 1]
    return {
        "unseen share of top-5 slots": top["unseen"].mean(),
        "unseen share of clicks": clicks["unseen"].mean(),
        "unseen share of candidate rows": df["unseen"].mean(),
        "coverage (candidates in any top 5)": top["news_id"].nunique() / df["news_id"].nunique(),
        "diversity (distinct subcategories in top 5)": top.groupby("impression_id")["subcategory"].nunique().mean(),
    }


def per_seed_boot(pairs: list, split, comparison: str) -> pd.DataFrame:
    out = []
    for seed, (a, b) in pairs:
        boot = paired_bootstrap(a, b, split, metrics=("auc", "ndcg@10"))
        out.append(boot.assign(comparison=comparison, seed=seed))
    return pd.concat(out, ignore_index=True)


def verdict(boot: pd.DataFrame) -> dict:
    """ADR 009 win criterion on the deciding segment."""
    r = boot[(boot["segment"] == DECIDING_SEGMENT) & (boot["metric"] == "auc")]
    all_above, all_below = (r["ci_low"] > 0).all(), (r["ci_high"] < 0).all()
    return {"mean": r["diff"].mean(), "min": r["diff"].min(), "max": r["diff"].max(),
            "seeds excluding 0": int(((r["ci_low"] > 0) | (r["ci_high"] < 0)).sum()),
            "outcome": "win for the first model" if all_above else "win for the second model" if all_below
            else "no win (not every seed's interval excludes zero on the same side)"}


def run_report() -> None:
    split = test_split()
    emb = embedding_scores(split)
    runs = {name: {s: pd.read_parquet(SCORES_DIR / f"{name}_seed{s}.parquet") for s in SEEDS} for name in TRAINED}

    # Accuracy per model and seed.
    results = [evaluate(emb, split, "embedding").assign(seed="-")]
    results += [evaluate(sc, split, name).assign(seed=s) for name, d in runs.items() for s, sc in d.items()]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(REPORTS_DIR / "gnn_test_results.csv", index=False)

    # Paired bootstraps per seed.
    boot = pd.concat([
        per_seed_boot([(s, (runs["gnn"][s], runs["a0"][s])) for s in SEEDS], split, "GNN vs A0 (primary)"),
        per_seed_boot([(s, (runs["gnn"][s], emb)) for s in SEEDS], split, "GNN vs embedding"),
        per_seed_boot([(s, (runs["a0"][s], emb)) for s in SEEDS], split, "A0 vs embedding"),
    ], ignore_index=True)
    boot.to_csv(REPORTS_DIR / "gnn_test_bootstrap.csv", index=False)
    verdicts = pd.DataFrame({c: verdict(g) for c, g in boot.groupby("comparison", sort=False)}).T
    verdicts.to_csv(REPORTS_DIR / "gnn_test_verdicts.csv")

    # Beyond-accuracy (per seed for trained models).
    subcat = load_graph("test")["nodes_article"].set_index("news_id")["subcategory_id"]
    beyond = [{"model": "embedding", "seed": "-", **top_k_metrics(emb, split, subcat)}]
    beyond += [{"model": name, "seed": s, **top_k_metrics(sc, split, subcat)} for name, d in runs.items()
               for s, sc in d.items()]
    pd.DataFrame(beyond).to_csv(REPORTS_DIR / "gnn_test_beyond_accuracy.csv", index=False)
    print(verdicts.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "report"])
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    a = parser.parse_args()
    run_train(a.seeds) if a.command == "train" else run_report()


if __name__ == "__main__":
    main()
