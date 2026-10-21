"""Graphs built for an arbitrary *graph period*, separate from a setup's training period (ADR 008).

A period graph is defined by:
    graph period   [graph_start, graph_end): clicks that build co-click edges and seen flags
    target period  [target_start, target_end): impressions the graph serves (training
                   examples or a scored day); their articles become nodes via content edges

Everything that depends on the graph period is recomputed from it: seen/unseen flags,
co-click edges and pruning (ADR 004), entity degree weights, and subcategory/category means
over articles that exist before the target period (ADR 006).

Run `uv run python -m recsys.graph.periods [name ...]` to build the named periods below.
Outputs: parquet tables in data/processed/graph/periods/<name>/ (source of truth) and
the HeteroData in data/processed/pyg/periods/<name>.pt.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from recsys.data.mind import load_behaviors, load_entity_mentions, load_news
from recsys.data.splits import EvalSplit
from recsys.data.wikidata import ENTITY_TYPES_FILE
from recsys.features.build import FEATURES_DIR, pre_target_group_mean
from recsys.graph.build import GRAPH_DIR, build_graph
from recsys.graph.pyg import PYG_DIR, to_heterodata

D = pd.Timestamp


@dataclass(frozen=True)
class Period:
    graph_start: pd.Timestamp
    graph_end: pd.Timestamp  # exclusive
    target_start: pd.Timestamp
    target_end: pd.Timestamp  # exclusive


PERIODS = {
    # Validation setup: train on a 9-11 Nov graph with 12-13 Nov examples; score 14 Nov.
    "validation_train": Period(D("2019-11-09"), D("2019-11-12"), D("2019-11-12"), D("2019-11-14")),
    "validation_score": Period(D("2019-11-09"), D("2019-11-14"), D("2019-11-14"), D("2019-11-15")),
    # One-off check: score 14 Nov with the (older) training graph.
    "validation_score_on_train_graph": Period(D("2019-11-09"), D("2019-11-12"), D("2019-11-14"), D("2019-11-15")),
    # Test setup: train on a 9-12 Nov graph with 13-14 Nov examples; score 15 Nov.
    "test_train": Period(D("2019-11-09"), D("2019-11-13"), D("2019-11-13"), D("2019-11-15")),
    "test_score": Period(D("2019-11-09"), D("2019-11-15"), D("2019-11-15"), D("2019-11-16")),
}


def all_behaviors() -> pd.DataFrame:
    """MIND train (9-14 Nov) and dev (15 Nov) impressions; impression IDs made unique."""
    train, dev = load_behaviors("train"), load_behaviors("dev")
    dev = dev.assign(impression_id=dev["impression_id"] + 1_000_000)  # dev IDs restart at 1
    return pd.concat([train, dev], ignore_index=True)


def period_split(name: str, behaviors: pd.DataFrame | None = None) -> EvalSplit:
    """EvalSplit whose `train` is the graph period and `target` the target period."""
    p = PERIODS[name]
    b = all_behaviors() if behaviors is None else behaviors
    graph = b[(b["time"] >= p.graph_start) & (b["time"] < p.graph_end)].reset_index(drop=True)
    target = b[(b["time"] >= p.target_start) & (b["time"] < p.target_end)].reset_index(drop=True)
    assert graph["time"].max() < target["time"].min(), "graph period overlaps target period"
    return EvalSplit(name=name, train=graph, target=target)


def period_features(tables: dict, split: EvalSplit) -> dict[str, pd.DataFrame]:
    """Starting features for a period graph (ADR 006), group means recomputed for the period."""
    article_emb = pd.DataFrame(np.load(FEATURES_DIR / "article_text.npy"),
                               index=pd.read_parquet(FEATURES_DIR / "article_ids.parquet")["news_id"])
    entity_ids = pd.read_parquet(FEATURES_DIR / "entity_ids.parquet")
    entity = pd.DataFrame(np.load(FEATURES_DIR / "entity_vectors.npy"), index=entity_ids["entity_id"])
    entity["missing"] = entity_ids["missing"].to_numpy(dtype=np.float32)

    nodes = tables["nodes_article"].set_index("news_id")
    setup_emb = article_emb.loc[nodes.index]
    known = split.seen_news | set(split.target["history"].explode().dropna())
    pre_target = pd.Series(nodes.index.isin(known), index=nodes.index)
    return {
        "article": article_emb,
        "entity": entity,
        "subcategory": pre_target_group_mean(setup_emb, nodes["subcategory_id"], pre_target),
        "category": pre_target_group_mean(setup_emb, nodes["category"], pre_target),
    }


def build_period(name: str, behaviors=None, news=None, mentions=None, entity_types=None):
    split = period_split(name, behaviors)
    tables = build_graph(split, news, mentions, entity_types=entity_types)
    out = GRAPH_DIR / "periods" / name
    out.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        df.to_parquet(out / f"{table}.parquet", index=False)
    data, ids = to_heterodata(tables, period_features(tables, split))
    (PYG_DIR / "periods").mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "ids": ids}, PYG_DIR / "periods" / f"{name}.pt")
    return data, ids


def load_period(name: str):
    saved = torch.load(PYG_DIR / "periods" / f"{name}.pt", weights_only=False)
    return saved["data"], saved["ids"]


def main(names: list[str]) -> None:
    behaviors = all_behaviors()
    news = pd.concat([load_news("train"), load_news("dev")]).drop_duplicates("news_id")
    mentions = pd.concat([load_entity_mentions("train"), load_entity_mentions("dev")])
    mentions = mentions.drop_duplicates(["news_id", "entity_id", "source"])
    entity_types = pd.read_parquet(ENTITY_TYPES_FILE)
    for name in names or PERIODS:
        data, _ = build_period(name, behaviors, news, mentions, entity_types)
        a = data["article"]
        print(f"{name:<33} articles {a.num_nodes:>6,} (unseen {(~a.seen_in_training).sum().item():>6,}) "
              f"co-click edges {data['article', 'coclicked', 'article'].num_edges // 2:>7,}")


if __name__ == "__main__":
    main(sys.argv[1:])
