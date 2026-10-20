"""Paired bootstrap: TF-IDF vs popularity on validation and test. Run with:

    uv run python -m recsys.experiments.compare_baselines

Uses the TF-IDF configuration frozen in reports/baselines.csv (run run_baselines first).
Writes reports/bootstrap_tfidf_vs_popularity.csv.
"""

import ast

import pandas as pd

from recsys.data.splits import load_split
from recsys.evaluation import paired_bootstrap
from recsys.experiments.run_baselines import load_catalogue
from recsys.models.baselines import Popularity, TfidfContent
from recsys.paths import REPORTS_DIR

N_RESAMPLES, SEED = 1000, 0


def frozen_tfidf_params() -> dict:
    baselines = pd.read_csv(REPORTS_DIR / "baselines.csv")
    configs = baselines.loc[baselines["model"] == "tfidf", "config"].unique()
    assert len(configs) == 1, f"expected one frozen TF-IDF config, found {configs}"
    return ast.literal_eval(configs[0])


def main() -> None:
    news = load_catalogue()
    params = frozen_tfidf_params()
    print(f"TF-IDF config (frozen from validation): {params}")

    results = []
    for split in (load_split("validation"), load_split("test")):
        tfidf = TfidfContent(**params).fit(split.train, news).score(split.target)
        popularity = Popularity().fit(split.train, news).score(split.target)
        results.append(paired_bootstrap(tfidf, popularity, split, model_a="tfidf", model_b="popularity",
                                        n_resamples=N_RESAMPLES, seed=SEED))
        print(f"  {split.name} done")

    results = pd.concat(results, ignore_index=True)
    results.to_csv(REPORTS_DIR / "bootstrap_tfidf_vs_popularity.csv", index=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
