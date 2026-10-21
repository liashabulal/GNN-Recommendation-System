"""Tests for starting-feature helpers (no model download, no real data)."""

import numpy as np
import pandas as pd

from recsys.features.build import article_texts, entity_features, group_mean


def test_entity_features_zero_vector_and_missing_flag():
    vectors = {"Q1": np.array([1.0, 2.0], dtype=np.float32)}
    feats, missing = entity_features(["Q1", "Q2"], vectors, dim=2)
    assert feats.tolist() == [[1.0, 2.0], [0.0, 0.0]]
    assert missing.tolist() == [0, 1]


def test_group_mean_averages_articles_per_subcategory():
    emb = pd.DataFrame([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], index=["A", "B", "C"])
    keys = pd.Series(["s1", "s1", "s2"], index=["A", "B", "C"])
    means = group_mean(emb, keys)
    assert means.loc["s1"].tolist() == [2.0, 0.0]
    assert means.loc["s2"].tolist() == [0.0, 2.0]


def test_article_texts_join_title_and_abstract_and_tolerate_missing_abstract():
    news = pd.DataFrame({"news_id": ["A", "B"], "title": ["Title A", "Title B"], "abstract": ["Abstract A", pd.NA]})
    texts = article_texts(pd.DataFrame({"news_id": ["B", "A"]}), news)
    assert texts == ["Title B", "Title A Abstract A"]


def test_pre_target_group_mean_excludes_target_day_and_falls_back():
    from recsys.features.build import pre_target_group_mean

    emb = pd.DataFrame([[1.0, 0.0], [9.0, 9.0], [0.0, 4.0]], index=["A", "T", "N"])
    keys = pd.Series(["s1", "s1", "s_new"], index=emb.index)
    pre_target = pd.Series([True, False, False], index=emb.index)  # T and N are target-day only
    means = pre_target_group_mean(emb, keys, pre_target)
    assert means.loc["s1"].tolist() == [1.0, 0.0]      # target-day article T excluded
    assert means.loc["s_new"].tolist() == [0.0, 4.0]   # no pre-target article: fallback
    assert list(means.index) == ["s1", "s_new"]
