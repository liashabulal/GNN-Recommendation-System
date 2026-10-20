"""Paired bootstrap tests on a toy split with known and new users."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from recsys.data.splits import EvalSplit
from recsys.evaluation.bootstrap import paired_bootstrap, resampled_means

COLUMNS = ["impression_id", "user_id", "time", "history", "candidates", "labels"]
N_IMPRESSIONS = 20  # even impression IDs come from known users, odd from new users


@pytest.fixture
def split():
    day = pd.Timestamp("2019-11-14")
    train = pd.DataFrame(
        [(i, f"U{i}", pd.Timestamp("2019-11-09"), [], ["S"], [1]) for i in range(0, N_IMPRESSIONS, 2)],
        columns=COLUMNS,
    )
    # Every impression: three candidates, the first one clicked.
    target = pd.DataFrame(
        [(i, f"U{i}", day, [], [f"N{i}a", f"N{i}b", f"N{i}c"], [1, 0, 0]) for i in range(N_IMPRESSIONS)],
        columns=COLUMNS,
    )
    return EvalSplit(name="toy", train=train, target=target)


def _scores(split, rank_click_first) -> pd.DataFrame:
    """Score the clicked article highest where rank_click_first(impression_id) is True, else lowest."""
    rows = []
    for imp in split.target["impression_id"]:
        good = rank_click_first(imp)
        rows += [(imp, f"N{imp}a", 3.0 if good else 1.0), (imp, f"N{imp}b", 2.0), (imp, f"N{imp}c", 1.5 if good else 3.0)]
    return pd.DataFrame(rows, columns=["impression_id", "news_id", "score"])


def _row(res, segment, metric):
    return res.set_index(["segment", "metric"]).loc[(segment, metric)]


def test_identical_models_have_zero_difference(split):
    scores = _scores(split, lambda i: i % 3 == 0)
    res = paired_bootstrap(scores, scores, split, n_resamples=200)
    # The toy split has no seen articles, so that segment is empty (NaN, not significant).
    empty = res[res["segment"] == "article=seen"]
    assert (empty["n_impressions"] == 0).all() and empty["diff"].isna().all()
    res = res[res["segment"] != "article=seen"]
    assert (res["diff"] == 0).all() and (res["ci_low"] == 0).all() and (res["ci_high"] == 0).all()
    assert not res["significant"].any()


def test_dominant_model(split):
    perfect = _scores(split, lambda i: True)  # AUC 1 on every impression
    reversed_ = _scores(split, lambda i: False)  # AUC 0 on every impression
    res = paired_bootstrap(perfect, reversed_, split, n_resamples=200)
    row = _row(res, "all", "auc")
    assert row["diff"] == 1 and row["ci_low"] == 1 and row["ci_high"] == 1 and row["significant"]
    assert row["n_impressions"] == N_IMPRESSIONS


def test_seed_makes_results_reproducible(split):
    a = _scores(split, lambda i: True)
    b = _scores(split, lambda i: i % 2 == 0)
    first = paired_bootstrap(a, b, split, n_resamples=200, seed=7)
    pd.testing.assert_frame_equal(first, paired_bootstrap(a, b, split, n_resamples=200, seed=7))


def test_interval_width_matches_standard_error():
    # Differences of 0 and 1 in equal parts: mean 0.5, standard error 0.5 / sqrt(100) = 0.05,
    # so a 95% interval should be close to 0.5 +/- 0.098.
    diffs = np.array([0.0, 1.0] * 50)
    low, high = np.quantile(resampled_means(diffs, n_resamples=1000, seed=0), [0.025, 0.975])
    assert low == pytest.approx(0.402, abs=0.03)
    assert high == pytest.approx(0.598, abs=0.03)


def test_test_mix_weighted_row(split):
    perfect = _scores(split, lambda i: True)
    # B matches A on new users (odd IDs) and is reversed on known users (even IDs).
    b = _scores(split, lambda i: i % 2 == 1)
    res = paired_bootstrap(perfect, b, dataclasses.replace(split, reweight_known_share=0.25), n_resamples=200)
    assert _row(res, "user=known", "auc")["diff"] == 1
    assert _row(res, "user=new", "auc")["diff"] == 0
    weighted = _row(res, "test-mix weighted", "auc")
    assert weighted["diff"] == pytest.approx(0.25)
    assert weighted["ci_low"] == pytest.approx(0.25) and weighted["ci_high"] == pytest.approx(0.25)
