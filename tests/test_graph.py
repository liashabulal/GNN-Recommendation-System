"""Graph construction tests: co-click edges must never use target-day information."""

import pandas as pd
import pytest

from recsys.data.splits import EvalSplit
from recsys.graph.build import build_graph, coclick_edges

COLUMNS = ["impression_id", "user_id", "time", "history", "candidates", "labels"]
TRAIN_DAY, TARGET_DAY = pd.Timestamp("2019-11-09"), pd.Timestamp("2019-11-14")


@pytest.fixture
def tables():
    train = pd.DataFrame([
        (1, "U1", TRAIN_DAY, ["H1"], ["A", "B"], [1, 0]),  # U1: H1 (history) + A (click)
        (2, "U2", TRAIN_DAY, ["H1"], ["A", "C"], [1, 1]),  # U2: H1 + A + C
    ], columns=COLUMNS)
    target = pd.DataFrame([
        # Known user U1 clicks B and X on the target day: must not create edges.
        (10, "U1", TARGET_DAY, ["H1"], ["B", "X"], [1, 1]),
        # Target-only user U9 has history [P, Q] and clicks Y: must not create edges.
        (11, "U9", TARGET_DAY, ["P", "Q"], ["Y", "A"], [1, 0]),
    ], columns=COLUMNS)
    ids = ["A", "B", "C", "H1", "P", "Q", "X", "Y"]
    news = pd.DataFrame({"news_id": ids, "category": "news", "subcategory": "newsus", "title": ids})
    mentions = pd.DataFrame({"news_id": ["X"], "entity_id": ["Q1"], "label": ["E"], "mind_type": ["P"],
                             "confidence": [1.0], "source": ["title"]})
    return build_graph(EvalSplit(name="toy", train=train, target=target), news, mentions)


def _pairs(edges):
    return {(s, d): w for s, d, w in edges[["src", "dst", "weight"]].itertuples(index=False)}


def test_coclick_uses_training_clicks_and_histories_only(tables):
    assert _pairs(tables["edges_article_coclicked_article"]) == {
        ("A", "H1"): 2,  # both users
        ("A", "C"): 1,
        ("C", "H1"): 1,
    }


def test_no_coclick_edge_touches_target_day_information(tables):
    edges = tables["edges_article_coclicked_article"]
    touched = set(edges["src"]) | set(edges["dst"])
    # B was only shown (not clicked) in training; X, Y only clicked on the target day;
    # P, Q only in a target-only user's history.
    assert touched.isdisjoint({"B", "X", "Y", "P", "Q"})


def test_unseen_articles_still_get_content_edges(tables):
    articles = tables["nodes_article"].set_index("news_id")
    assert not articles.loc["X", "seen_in_training"]
    assert "X" in set(tables["edges_article_in_subcategory"]["news_id"])
    assert "X" in set(tables["edges_article_mentions_entity"]["news_id"])


def test_coclick_weight_counts_users_not_clicks():
    # The same user clicking the same pair twice still counts once.
    clicks = pd.DataFrame({"user_id": ["U1", "U1", "U2"], "news_id": ["A", "B", "A"]})
    clicks = pd.concat([clicks, clicks]).drop_duplicates()
    assert _pairs(coclick_edges(clicks)) == {("A", "B"): 1}
