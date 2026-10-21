"""GNN component tests on a tiny hand-built graph (forward pass only)."""

import numpy as np
import pandas as pd
import pytest
import torch

from recsys.graph.pyg import to_heterodata
from recsys.models.gnn import GNNEncoder, UserEncoder, article_loader, encode_all_articles, score
from recsys.models.history import history_tensor, impression_histories

HIDDEN = 8
TYPES = ["person", "place", "organisation", "event", "creative work", "other"]


def tiny_tables(drop_unseen_content: bool = False) -> dict:
    """Articles A, B, C (seen; co-clicked) and U (unseen: content edges only)."""
    articles = ["A", "B", "C", "U"]
    in_sub = pd.DataFrame({"news_id": articles, "subcategory_id": ["news/us"] * 3 + ["news/world"]})
    mentions = pd.DataFrame({"news_id": ["A", "B", "U", "C"], "entity_id": ["Q1", "Q1", "Q1", "Q2"],
                             "source": "title", "confidence": 1.0})
    if drop_unseen_content:  # U keeps its node but loses every edge
        in_sub = in_sub[in_sub["news_id"] != "U"]
        mentions = mentions[mentions["news_id"] != "U"]
    return {
        "nodes_article": pd.DataFrame({"news_id": articles, "seen_in_training": [True, True, True, False]}),
        "nodes_subcategory": pd.DataFrame({"subcategory_id": ["news/us", "news/world"]}),
        "nodes_category": pd.DataFrame({"category": ["news"]}),
        "nodes_entity": pd.DataFrame({"entity_id": ["Q1", "Q2"]}),
        "nodes_entity_type": pd.DataFrame({"entity_type": TYPES}),
        "edges_article_in_subcategory": in_sub,
        "edges_subcategory_in_category": pd.DataFrame({"subcategory_id": ["news/us", "news/world"],
                                                       "category": ["news", "news"]}),
        "edges_article_mentions_entity": mentions,
        "edges_entity_is_type": pd.DataFrame({"entity_id": ["Q1", "Q2"], "entity_type": ["person", "place"]}),
        "edges_article_coclicked_article_pruned": pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"],
                                                                "weight": [3, 2], "norm_weight": [0.5, 0.25]}),
    }


FEATURES = {
    # U's own input features are all zero, so anything non-zero about it comes from the graph.
    "article": pd.DataFrame(np.eye(4, 3, dtype=np.float32), index=["A", "B", "C", "U"]),
    "subcategory": pd.DataFrame([[1, 0, 0], [0, 0, 1]], dtype=np.float32, index=["news/us", "news/world"]),
    "category": pd.DataFrame([[1, 1, 1]], dtype=np.float32, index=["news"]),
    "entity": pd.DataFrame([[1, 0], [0, 1]], dtype=np.float32, index=["Q1", "Q2"]),
}
GNN_CFG = {"gnn": {"hidden_dim": HIDDEN, "num_layers": 2}, "features": {"entity_type_dim": 4}}


@pytest.fixture
def data():
    return to_heterodata(tiny_tables(), FEATURES)[0]


@pytest.fixture
def encoder(data):
    torch.manual_seed(0)
    return GNNEncoder.from_data(data, GNN_CFG).eval()


def test_output_shapes(data, encoder):
    out = encoder(data)
    expected = {"article": 4, "subcategory": 2, "category": 1, "entity": 2, "entity_type": 6}
    assert {nt: tuple(x.shape) for nt, x in out.items()} == {nt: (n, HIDDEN) for nt, n in expected.items()}


def test_article_vectors_have_unit_length(data, encoder):
    norms = encoder(data)["article"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-6)


def test_unseen_article_gets_vector_from_content_edges(data, encoder):
    u = 3
    assert not data["article"].seen_in_training[u]
    cc = data["article", "coclicked", "article"].edge_index
    assert not (cc == u).any()  # no co-click edges

    with_content = encoder(data)["article"][u]
    no_content = encoder(to_heterodata(tiny_tables(drop_unseen_content=True), FEATURES)[0])["article"][u]
    assert with_content.abs().sum() > 0
    # Its vector changes when its subcategory/entity edges are removed: messages do reach it.
    assert not torch.allclose(with_content, no_content, atol=1e-4)


def test_full_neighbourhood_sampling_matches_full_graph(data, encoder):
    full = encoder(data)["article"]
    sampled = encode_all_articles(encoder, data, batch_size=2, fanout=[-1, -1])  # -1 = all neighbours
    assert torch.allclose(sampled, full, atol=1e-5)


def test_fanout_limits_sampled_neighbours(data):
    # Seed B has two co-click neighbours (A and C); with fanout 1 only one is sampled.
    seed = 1
    full = data["article", "coclicked", "article"].edge_index
    assert (full[1] == seed).sum() == 2
    batch = next(iter(article_loader(data, torch.tensor([seed]), batch_size=1, fanout=[1, 1])))
    cc = batch["article", "coclicked", "article"].edge_index
    assert (cc[1] == 0).sum() == 1  # the seed is local node 0 in the sampled subgraph


def test_empty_history_gets_default_vector():
    torch.manual_seed(0)
    users = UserEncoder(HIDDEN)
    articles = torch.nn.functional.normalize(torch.randn(5, HIDDEN), dim=-1)
    history = torch.tensor([[0, 2, -1], [-1, -1, -1]])
    out = users(articles, history)
    assert torch.allclose(out[0], (articles[0] + articles[2]) / 2)
    assert torch.allclose(out[1], users.default)


def test_score_is_dot_product():
    u = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    c = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    assert score(u, c).tolist() == [11.0, 6.0]


def test_impression_histories_order_dedup_cutoff():
    t = pd.Timestamp
    train = pd.DataFrame([
        (1, "U1", t("2019-11-09 10:00"), ["A", "B"], ["C", "X"], [1, 0]),  # clicks C at 10:00
        (2, "U1", t("2019-11-10 10:00"), ["A", "B"], ["A", "Y"], [1, 0]),  # clicks A again at day 2
    ], columns=["impression_id", "user_id", "time", "history", "candidates", "labels"])
    behaviors = pd.DataFrame([
        (5, "U1", t("2019-11-09 10:00"), ["A", "B"]),  # same time as the C click: excluded
        (6, "U1", t("2019-11-09 12:00"), ["A", "B"]),  # after C
        (7, "U1", t("2019-11-11 09:00"), ["A", "B"]),  # after C and the repeat click on A
        (8, "U9", t("2019-11-11 09:00"), []),          # nothing at all
    ], columns=["impression_id", "user_id", "time", "history"])
    h = impression_histories(behaviors, train, max_history=3)
    assert h[5] == ["A", "B"]
    assert h[6] == ["A", "B", "C"]
    assert h[7] == ["B", "C", "A"]  # A moved to its most recent position
    assert h[8] == []
    assert impression_histories(behaviors, train, max_history=2)[7] == ["C", "A"]  # last 2 = most recent

    rows = pd.Series({"A": 0, "B": 1, "C": 2})
    assert history_tensor(h.loc[[6, 8]], rows, max_history=4).tolist() == [[0, 1, 2, -1], [-1, -1, -1, -1]]


def test_train_step_runs_and_gradients_flow(data):
    from recsys.training.train_gnn import ClickExamples, Recommender, _sampler, train_step

    torch.manual_seed(0)
    cfg = {**GNN_CFG, "gnn": {**GNN_CFG["gnn"], "dropout": 0.0}}
    model = Recommender(data, cfg)
    row = pd.Series(range(4), index=["A", "B", "C", "U"])
    examples = pd.DataFrame({"impression_id": [1, 2], "candidates": [["A", "B", "U"], ["C", "U"]],
                             "labels": [[1, 0, 0], [0, 1]]})
    hist = torch.tensor([[0, 2, -1], [-1, -1, -1]])  # impression 2: empty history -> default vector
    ex = ClickExamples(examples, row, hist)
    assert len(ex) == 2 and ex.pos.tolist() == [0, 3]

    rng = np.random.default_rng(0)
    negs = ex.negatives(np.array([1]), k=4, rng=rng)  # only one negative available: with replacement
    assert negs.tolist() == [[2, 2, 2, 2]]

    loss = train_step(model, _sampler(data, [-1, -1]), ex, np.array([0, 1]), k=4, rng=rng)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.users.default.grad is not None and model.users.default.grad.abs().sum() > 0
    assert model.encoder.proj["article"].weight.grad.abs().sum() > 0
