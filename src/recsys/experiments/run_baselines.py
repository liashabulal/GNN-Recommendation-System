"""Tune and evaluate the three baselines. Run with:

    uv run python -m recsys.experiments.run_baselines

1. Tune TF-IDF on validation (fit on 9-13 Nov); pick by test-mix weighted AUC.
2. Freeze that configuration.
3. Score every baseline on validation (fit on 9-13 Nov) and on test (refit on 9-14 Nov).

Writes reports/tfidf_tuning.csv and reports/baselines.csv.
"""

import itertools
import time

import pandas as pd

from recsys.data.mind import load_news
from recsys.data.splits import load_split
from recsys.evaluation import evaluate
from recsys.models.baselines import Popularity, RandomScores, TfidfContent
from recsys.paths import REPORTS_DIR

TUNING_SEGMENT, TUNING_METRIC = "test-mix weighted", "auc"
TFIDF_GRID = {"ngram_range": [(1, 1), (1, 2)], "sublinear_tf": [False, True]}


def load_catalogue() -> pd.DataFrame:
    """All articles from both MIND splits (content only, no interactions)."""
    news = pd.concat([load_news("train"), load_news("dev")])
    return news.drop_duplicates("news_id").reset_index(drop=True)


def run(model, split, news) -> pd.DataFrame:
    start = time.time()
    results = evaluate(model.fit(split.train, news).score(split.target), split, model=model.name)
    print(f"  {model.name:<10} {split.name:<10} {time.time() - start:5.1f}s")
    return results


def tune_tfidf(validation, news) -> tuple[dict, pd.DataFrame]:
    rows = []
    for values in itertools.product(*TFIDF_GRID.values()):
        params = dict(zip(TFIDF_GRID, values))
        res = run(TfidfContent(**params), validation, news).set_index("segment")
        rows.append({**{k: str(v) for k, v in params.items()},
                     **{f"{seg} {m}": res.loc[seg, m] for seg in [TUNING_SEGMENT, "user=new", "all"]
                        for m in ["auc", "mrr", "ndcg@10"]}})
        rows[-1]["_params"] = params
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning[f"{TUNING_SEGMENT} {TUNING_METRIC}"].idxmax(), "_params"]
    return best, tuning.drop(columns="_params")


def main() -> None:
    news = load_catalogue()
    validation, test = load_split("validation"), load_split("test")
    print(f"Validation known-user share of test (reweighting): {validation.reweight_known_share:.4f}")

    print("Tuning TF-IDF on validation")
    best, tuning = tune_tfidf(validation, news)
    print(f"  selected: {best}")

    print("Scoring baselines (hyperparameters frozen)")
    results = pd.concat(
        [run(model, split, news)
         for split in (validation, test)
         for model in (RandomScores(seed=0), Popularity(), TfidfContent(**best))],
        ignore_index=True,
    )
    results.insert(2, "config", results["model"].map({"tfidf": str(best)}).fillna(""))

    REPORTS_DIR.mkdir(exist_ok=True)
    tuning.to_csv(REPORTS_DIR / "tfidf_tuning.csv", index=False)
    results.to_csv(REPORTS_DIR / "baselines.csv", index=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
