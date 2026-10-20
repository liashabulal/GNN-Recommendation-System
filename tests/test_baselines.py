"""Baseline tests, focused on what each model is allowed to learn from."""

import pandas as pd
import pytest

from recsys.models.baselines import Popularity, TfidfContent


def _behaviors(rows):
    return pd.DataFrame(rows, columns=["impression_id", "user_id", "time", "history", "candidates", "labels"])


NEWS = pd.DataFrame({
    "news_id": ["A", "B", "C", "D"],
    "title": ["football match tonight", "football score update", "stock market falls", "market football mix"],
    "abstract": [pd.NA, "late goal wins", "shares drop sharply", pd.NA],
})

TRAIN = _behaviors([
    (1, "U1", pd.Timestamp("2019-11-09"), [], ["A", "B"], [1, 0]),
    (2, "U2", pd.Timestamp("2019-11-10"), [], ["A", "B"], [1, 1]),
])
# Target labels click C every time; a model must not be able to learn that.
TARGET = _behaviors([
    (10, "U1", pd.Timestamp("2019-11-14"), ["A"], ["A", "B", "C"], [0, 0, 1]),
    (11, "U3", pd.Timestamp("2019-11-14"), [], ["C", "D"], [1, 0]),
])


def _by_pair(scores):
    return scores.set_index(["impression_id", "news_id"])["score"]


def test_popularity_counts_training_clicks_only():
    scores = _by_pair(Popularity().fit(TRAIN, NEWS).score(TARGET))
    assert scores[(10, "A")] == 2  # clicked twice in training
    assert scores[(10, "B")] == 1
    assert scores[(10, "C")] == 0  # clicked only in target -> must stay 0
    assert scores[(11, "D")] == 0  # never seen


def test_tfidf_vocabulary_from_training_articles_only():
    model = TfidfContent(min_df=1).fit(TRAIN, NEWS)
    vocab = set(model.vectorizer_.vocabulary_)
    assert "football" in vocab
    assert "stock" not in vocab and "shares" not in vocab  # only in unseen article C


def test_tfidf_scores_are_cosine_to_history_profile():
    scores = _by_pair(TfidfContent(min_df=1).fit(TRAIN, NEWS).score(TARGET))
    # U1's history is [A]: A itself is a perfect match, B shares "football".
    assert scores[(10, "A")] == pytest.approx(1.0)
    assert 0 < scores[(10, "B")] < 1
    # U3 has an empty history: every candidate ties at 0.
    assert scores[(11, "C")] == scores[(11, "D")] == 0
