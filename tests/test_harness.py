"""Harness tests on a tiny hand-built split."""

import pandas as pd
import pytest

from recsys.data.splits import EvalSplit
from recsys.evaluation.harness import evaluate


def _behaviors(rows):
    return pd.DataFrame(rows, columns=["impression_id", "user_id", "time", "history", "candidates", "labels"])


@pytest.fixture
def split():
    # Training period: user U1 saw articles A and B (and has C in their history).
    train = _behaviors([
        (1, "U1", pd.Timestamp("2019-11-09"), ["C"], ["A", "B"], [1, 0]),
    ])
    # Target: impression 10 from known user U1, impression 11 from new user U2.
    # Articles A, B, C are seen; X and Y are unseen.
    target = _behaviors([
        (10, "U1", pd.Timestamp("2019-11-14"), ["C"], ["A", "X", "B"], [1, 0, 0]),
        (11, "U2", pd.Timestamp("2019-11-14"), [], ["B", "Y", "X"], [0, 1, 0]),
    ])
    return EvalSplit(name="toy", train=train, target=target)


def _scores(pairs):
    return pd.DataFrame(pairs, columns=["impression_id", "news_id", "score"])


def test_segments(split):
    scores = _scores([
        (10, "A", 3.0), (10, "X", 2.0), (10, "B", 1.0),  # imp 10: click at rank 1
        (11, "B", 3.0), (11, "Y", 2.0), (11, "X", 1.0),  # imp 11: click at rank 2
    ])
    res = evaluate(scores, split, model="toy").set_index("segment")

    assert res.loc["all", "n_impressions"] == 2
    assert res.loc["all", "mrr"] == pytest.approx((1 + 1 / 2) / 2)
    assert res.loc["user=known", "mrr"] == 1.0
    assert res.loc["user=new", "mrr"] == 0.5

    # Seen-only sub-lists: imp 10 -> [A(click), B]; imp 11 -> [B] has no click, so it is dropped.
    assert res.loc["article=seen", "n_impressions"] == 1
    assert res.loc["article=seen", "mrr"] == 1.0
    # Unseen-only sub-lists: imp 10 -> [X] has no click (dropped); imp 11 -> [Y(click), X].
    assert res.loc["article=unseen", "n_impressions"] == 1
    assert res.loc["article=unseen", "auc"] == 1.0

    assert set(res["model"]) == {"toy"} and set(res["split"]) == {"toy"}


def test_missing_score_raises(split):
    scores = _scores([(10, "A", 1.0), (10, "X", 1.0), (10, "B", 1.0), (11, "B", 1.0), (11, "Y", 1.0)])
    with pytest.raises(ValueError, match="no score"):
        evaluate(scores, split, model="toy")


def test_extra_score_raises(split):
    scores = _scores([
        (10, "A", 1.0), (10, "X", 1.0), (10, "B", 1.0),
        (11, "B", 1.0), (11, "Y", 1.0), (11, "X", 1.0),
        (99, "Z", 1.0),
    ])
    with pytest.raises(ValueError, match="not target candidates"):
        evaluate(scores, split, model="toy")


def test_duplicate_score_raises(split):
    scores = _scores([
        (10, "A", 1.0), (10, "A", 2.0), (10, "X", 1.0), (10, "B", 1.0),
        (11, "B", 1.0), (11, "Y", 1.0), (11, "X", 1.0),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(scores, split, model="toy")
