"""HeteroData conversion tests: a hand-built toy graph, plus the real saved graphs when present."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from recsys.graph.build import load_graph
from recsys.graph.pyg import PYG_DIR, load_heterodata, to_heterodata


def _edge_set(data, edge_type):
    return set(map(tuple, data[edge_type].edge_index.t().tolist()))


def assert_every_edge_type_has_reverse(data):
    for src, rel, dst in data.edge_types:
        reverses = [et for et in data.edge_types if et[0] == dst and et[2] == src and
                    _edge_set(data, et) == {(b, a) for a, b in _edge_set(data, (src, rel, dst))}]
        assert reverses, f"no reverse edge type for {(src, rel, dst)}"


def assert_no_coclick_on_unseen(data):
    unseen = (~data["article"].seen_in_training).nonzero().flatten()
    touched = data["article", "coclicked", "article"].edge_index.unique()
    assert not torch.isin(touched, unseen).any()


@pytest.fixture
def toy():
    tables = {
        "nodes_article": pd.DataFrame({"news_id": ["A", "B", "C", "U"], "seen_in_training": [True, True, True, False]}),
        "nodes_subcategory": pd.DataFrame({"subcategory_id": ["news/us"]}),
        "nodes_category": pd.DataFrame({"category": ["news"]}),
        "nodes_entity": pd.DataFrame({"entity_id": ["Q1", "Q2"]}),
        "nodes_entity_type": pd.DataFrame({"entity_type": ["person", "place", "organisation", "event",
                                                           "creative work", "other"]}),
        "edges_article_in_subcategory": pd.DataFrame({"news_id": ["A", "B", "C", "U"], "subcategory_id": "news/us"}),
        "edges_subcategory_in_category": pd.DataFrame({"subcategory_id": ["news/us"], "category": ["news"]}),
        # Q1 is mentioned by three articles (A twice: title and abstract), Q2 by one.
        "edges_article_mentions_entity": pd.DataFrame({
            "news_id": ["A", "A", "B", "U", "C"], "entity_id": ["Q1", "Q1", "Q1", "Q1", "Q2"],
            "source": ["title", "abstract", "title", "title", "title"], "confidence": 1.0}),
        "edges_entity_is_type": pd.DataFrame({"entity_id": ["Q1", "Q2"], "entity_type": ["person", "place"]}),
        "edges_article_coclicked_article_pruned": pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"],
                                                                "weight": [3, 2], "norm_weight": [0.5, 0.25]}),
    }
    features = {
        "article": pd.DataFrame(np.eye(4, 3, dtype=np.float32), index=["A", "B", "C", "U"]),
        "subcategory": pd.DataFrame(np.ones((1, 3), dtype=np.float32), index=["news/us"]),
        "category": pd.DataFrame(np.ones((1, 3), dtype=np.float32), index=["news"]),
        "entity": pd.DataFrame(np.ones((2, 2), dtype=np.float32), index=["Q1", "Q2"]),
    }
    return tables, features


def test_toy_node_counts_and_features(toy):
    tables, features = toy
    data, ids = to_heterodata(tables, features)
    assert data["article"].num_nodes == 4 and data["entity_type"].num_nodes == 6
    assert ids["article"] == ["A", "B", "C", "U"]
    assert torch.equal(data["article"].x, torch.tensor(np.eye(4, 3, dtype=np.float32)))


def test_toy_reverse_edges_and_symmetric_coclick(toy):
    data, _ = to_heterodata(*toy)
    assert_every_edge_type_has_reverse(data)
    cc = _edge_set(data, ("article", "coclicked", "article"))
    assert cc == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert data["article", "coclicked", "article"].edge_weight.tolist() == [0.5, 0.25, 0.5, 0.25]


def test_toy_entity_edge_weight_uses_article_degree(toy):
    data, _ = to_heterodata(*toy)
    et = ("entity", "mentioned_in", "article")
    # A's title + abstract mentions of Q1 collapse to one edge, so Q1 has 3 edges, Q2 has 1.
    assert data[et].edge_index.shape[1] == 4
    weights = dict(zip(map(tuple, data[et].edge_index.t().tolist()), data[et].edge_weight.tolist()))
    assert weights[(0, 0)] == pytest.approx(1 / math.log(1 + 3))  # Q1 -> A
    assert weights[(1, 2)] == pytest.approx(1 / math.log(1 + 1))  # Q2 -> C


def test_toy_no_coclick_on_unseen(toy):
    assert_no_coclick_on_unseen(to_heterodata(*toy)[0])


def test_missing_features_raise(toy):
    tables, features = toy
    features["article"] = features["article"].drop(index="U")
    with pytest.raises(ValueError, match="missing starting features"):
        to_heterodata(tables, features)


@pytest.mark.parametrize("setup", ["validation", "test"])
def test_real_graph(setup):
    if not (PYG_DIR / f"{setup}.pt").exists():
        pytest.skip("run `python -m recsys.graph.pyg` first")
    data, ids = load_heterodata(setup)
    tables = load_graph(setup)
    expected = {"article": "nodes_article", "subcategory": "nodes_subcategory", "category": "nodes_category",
                "entity": "nodes_entity", "entity_type": "nodes_entity_type"}
    for node_type, table in expected.items():
        assert data[node_type].num_nodes == len(tables[table]), node_type
    assert_every_edge_type_has_reverse(data)
    assert_no_coclick_on_unseen(data)
    assert data["article", "coclicked", "article"].num_edges == 2 * len(tables["edges_article_coclicked_article_pruned"])
