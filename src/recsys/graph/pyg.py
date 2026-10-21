"""Convert the parquet graph tables + starting features into a PyG HeteroData per setup.

Run with `uv run python -m recsys.graph.pyg`; writes data/processed/pyg/<setup>.pt.
Parquet stays the source of truth (Neo4j is never read).

Node types and features (ADR 006):
    article      x [n, 384]  frozen text embedding;  seen_in_training [n] bool
    subcategory  x [n, 384]  mean article text embedding (per setup)
    category     x [n, 384]  mean article text embedding (per setup)
    entity       x [n, 101]  MIND vector (zeros if missing) + missing flag
    entity_type  num_nodes = 6, no x (learnable embedding in the model)

Edge types, each relation stored in both directions as separate edge types:
    article -in-> subcategory              subcategory -contains-> article
    subcategory -in-> category             category -contains-> subcategory
    article -mentions-> entity             entity -mentioned_in-> article  (edge_weight = 1 / log(1 + degree))
    entity -is-> entity_type               entity_type -has-> entity
    article -coclicked-> article           (edge_weight = pruned norm_weight)

Co-click is symmetric and stored once per pair in parquet with an arbitrary src < dst
order, so it becomes a single article->article edge type containing both directions: it
is its own reverse. Two separate types would learn two weight matrices for one relation,
split by ID order.

An article can mention the same entity in both title and abstract; for message passing
that is one edge (the parquet table keeps both rows with their source and confidence).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from recsys.features.build import FEATURES_DIR
from recsys.graph.build import load_graph
from recsys.paths import PROCESSED_DIR

PYG_DIR = PROCESSED_DIR / "pyg"
SETUPS = ("validation", "test")

# (src type, relation, dst type) -> reverse relation name
REVERSE = {
    ("article", "in", "subcategory"): "contains",
    ("subcategory", "in", "category"): "contains",
    ("article", "mentions", "entity"): "mentioned_in",
    ("entity", "is", "entity_type"): "has",
}


def load_features(setup: str) -> dict[str, pd.DataFrame]:
    """Starting features as DataFrames indexed by node id."""
    def frame(npy, ids_file, id_col):
        return pd.DataFrame(np.load(npy), index=pd.read_parquet(ids_file)[id_col])

    entity_ids = pd.read_parquet(FEATURES_DIR / "entity_ids.parquet")
    entity = pd.DataFrame(np.load(FEATURES_DIR / "entity_vectors.npy"), index=entity_ids["entity_id"])
    entity["missing"] = entity_ids["missing"].to_numpy(dtype=np.float32)
    return {
        "article": frame(FEATURES_DIR / "article_text.npy", FEATURES_DIR / "article_ids.parquet", "news_id"),
        "subcategory": frame(FEATURES_DIR / setup / "subcategory_text.npy",
                             FEATURES_DIR / setup / "subcategory_ids.parquet", "subcategory_id"),
        "category": frame(FEATURES_DIR / setup / "category_text.npy",
                          FEATURES_DIR / setup / "category_ids.parquet", "category_id"),
        "entity": entity,
    }


def _index(ids) -> pd.Series:
    return pd.Series(np.arange(len(ids)), index=pd.Index(ids))


def _edge_index(src_ids, dst_ids, src_map: pd.Series, dst_map: pd.Series) -> torch.Tensor:
    return torch.tensor(np.stack([src_map.loc[src_ids].to_numpy(), dst_map.loc[dst_ids].to_numpy()]),
                        dtype=torch.long)


def to_heterodata(tables: dict[str, pd.DataFrame], features: dict[str, pd.DataFrame]) -> tuple[HeteroData, dict]:
    """Build HeteroData from graph tables and features. Returns (data, node id lists per type)."""
    ids = {
        "article": tables["nodes_article"]["news_id"].tolist(),
        "subcategory": tables["nodes_subcategory"]["subcategory_id"].tolist(),
        "category": tables["nodes_category"]["category"].tolist(),
        "entity": tables["nodes_entity"]["entity_id"].tolist(),
        "entity_type": tables["nodes_entity_type"]["entity_type"].tolist(),
    }
    maps = {t: _index(v) for t, v in ids.items()}

    data = HeteroData()
    for t in ("article", "subcategory", "category", "entity"):
        x = features[t].reindex(ids[t])
        if x.isna().any().any():
            raise ValueError(f"missing starting features for {x.isna().any(axis=1).sum()} {t} nodes")
        data[t].x = torch.tensor(x.to_numpy(dtype=np.float32))
    data["entity_type"].num_nodes = len(ids["entity_type"])
    data["article"].seen_in_training = torch.tensor(tables["nodes_article"]["seen_in_training"].to_numpy(dtype=bool))

    mentions = tables["edges_article_mentions_entity"].drop_duplicates(["news_id", "entity_id"])
    edges = {
        ("article", "in", "subcategory"): (tables["edges_article_in_subcategory"], "news_id", "subcategory_id"),
        ("subcategory", "in", "category"): (tables["edges_subcategory_in_category"], "subcategory_id", "category"),
        ("article", "mentions", "entity"): (mentions, "news_id", "entity_id"),
        ("entity", "is", "entity_type"): (tables["edges_entity_is_type"], "entity_id", "entity_type"),
    }
    for (src, rel, dst), (df, src_col, dst_col) in edges.items():
        ei = _edge_index(df[src_col], df[dst_col], maps[src], maps[dst])
        data[src, rel, dst].edge_index = ei
        data[dst, REVERSE[(src, rel, dst)], src].edge_index = ei.flip(0)

    # entity -> article messages are down-weighted for entities mentioned by many articles.
    degree = mentions.groupby("entity_id")["news_id"].nunique()
    data["entity", "mentioned_in", "article"].edge_weight = torch.tensor(
        (1 / np.log1p(degree.loc[mentions["entity_id"]].to_numpy())), dtype=torch.float32)

    # Co-click: pruned edges, both directions in one symmetric edge type.
    cc = tables["edges_article_coclicked_article_pruned"]
    ei = _edge_index(cc["src"], cc["dst"], maps["article"], maps["article"])
    w = torch.tensor(cc["norm_weight"].to_numpy(dtype=np.float32))
    data["article", "coclicked", "article"].edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    data["article", "coclicked", "article"].edge_weight = torch.cat([w, w])

    data.validate(raise_on_error=True)
    return data, ids


def build(setup: str) -> tuple[HeteroData, dict]:
    return to_heterodata(load_graph(setup), load_features(setup))


def load_heterodata(setup: str) -> tuple[HeteroData, dict]:
    saved = torch.load(PYG_DIR / f"{setup}.pt", weights_only=False)
    return saved["data"], saved["ids"]


def main() -> None:
    PYG_DIR.mkdir(parents=True, exist_ok=True)
    for setup in SETUPS:
        data, ids = build(setup)
        torch.save({"data": data, "ids": ids}, PYG_DIR / f"{setup}.pt")
        print(f"--- {setup}\n{data}")


if __name__ == "__main__":
    main()
