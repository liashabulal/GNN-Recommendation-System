"""Build leakage-safe graph tables for one evaluation setup (see docs/decisions/).

Article-centred graph with no user nodes (ADR 001):

    nodes  article, subcategory, category, entity
    edges  article -in-> subcategory -in-> category
           article -mentions-> entity        (confidence, source = title/abstract)
           article -coclicked- article       (undirected, weight = number of users)
           ... plus a pruned, normalised co-click set (ADR 004, settings in configs/graph.toml)
           entity -is-> entity_type          (coarse Wikidata type, if the enrichment cache exists)

Content edges (subcategory, entity) come from article text and metadata, which exist
as soon as an article is published, so they cover every article in the setup, including
unseen target-day ones. Co-click edges use *interactions*, so they come only from the
training period (ADR 002): the clicks in training-period impressions and the history
snapshots of training-period users. Target impressions, including their labels and
the histories of target-only users, are never used for co-click edges.

Run `uv run python -m recsys.graph.build` to write both setups to data/processed/graph/.
"""

import tomllib

import numpy as np
import pandas as pd

from recsys.data.mind import explode_impressions, load_entity_mentions, load_news
from recsys.data.splits import EvalSplit, load_split
from recsys.data.wikidata import COARSE_TYPES, ENTITY_TYPES_FILE
from recsys.paths import CONFIGS_DIR, PROCESSED_DIR

GRAPH_DIR = PROCESSED_DIR / "graph"


def subcategory_id(category: pd.Series, subcategory: pd.Series) -> pd.Series:
    # Subcategory names repeat across categories (e.g. "news" under both news and video).
    return category + "/" + subcategory


def training_clicks(train: pd.DataFrame) -> pd.DataFrame:
    """Distinct (user_id, news_id) clicks from training-period users only.

    Sources: each training user's history snapshot (taken before the log window)
    and their clicks in training-period impressions.
    """
    history = (train.drop_duplicates("user_id")[["user_id", "history"]]
               .explode("history").dropna().rename(columns={"history": "news_id"}))
    cands = explode_impressions(train)
    clicked = cands.loc[cands["clicked"] == 1, ["user_id", "news_id"]]
    return pd.concat([history, clicked]).drop_duplicates().reset_index(drop=True)


def coclick_edges(clicks: pd.DataFrame) -> pd.DataFrame:
    """Undirected article pairs clicked by the same user; weight = number of such users.

    Each pair is stored once with src < dst.
    """
    ids, codes = np.unique(clicks["news_id"].to_numpy(dtype=str), return_inverse=True)
    clicks = clicks.assign(code=codes).sort_values(["user_id", "code"])
    user = clicks["user_id"].to_numpy()
    code = clicks["code"].to_numpy(dtype=np.int64)

    # For each user, every pair (i < j) of their clicked articles, encoded as one int64.
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    ends = np.r_[starts[1:], len(user)]
    pair_keys = []
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        i, j = np.triu_indices(end - start, k=1)
        items = code[start:end]  # sorted, so items[i] < items[j]
        pair_keys.append(items[i] * len(ids) + items[j])

    keys, weights = np.unique(np.concatenate(pair_keys), return_counts=True)
    return pd.DataFrame({
        "src": ids[keys // len(ids)],
        "dst": ids[keys % len(ids)],
        "weight": weights.astype(np.int32),
    })


def load_pruning_config() -> dict:
    with open(CONFIGS_DIR / "graph.toml", "rb") as f:
        return tomllib.load(f)["coclick_pruning"]


def prune_coclick_edges(edges: pd.DataFrame, readers: pd.Series, min_users: int, top_k: int) -> pd.DataFrame:
    """Pruned, normalised co-click edges (ADR 004).

    1. Keep edges shared by at least `min_users` users.
    2. norm_weight = weight / sqrt(readers_src * readers_dst): a cosine between the two
       articles' reader sets, so popular articles stop dominating every neighbourhood.
    3. Keep an edge if it is among the `top_k` highest norm_weight edges of either endpoint.
    """
    e = edges[edges["weight"] >= min_users].copy()
    e["norm_weight"] = e["weight"] / np.sqrt(e["src"].map(readers).to_numpy() * e["dst"].map(readers).to_numpy())

    # Look at each edge from both endpoints and rank neighbours per article.
    both = pd.concat([
        e.assign(article=e["src"], edge=np.arange(len(e))),
        e.assign(article=e["dst"], edge=np.arange(len(e))),
    ])
    # Deterministic ties: higher raw weight first, then edge order.
    both = both.sort_values(["article", "norm_weight", "weight", "edge"], ascending=[True, False, False, True])
    keep = both.loc[both.groupby("article").cumcount() < top_k, "edge"].unique()
    return e.iloc[np.sort(keep)].reset_index(drop=True)


def build_graph(split: EvalSplit, news: pd.DataFrame, mentions: pd.DataFrame,
                pruning: dict | None = None, entity_types: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """All node and edge tables for one setup, keyed by table name.

    `entity_types` (from recsys.data.wikidata) adds entity-type nodes and edges.
    """
    pruning = pruning or load_pruning_config()
    clicks = training_clicks(split.train)
    readers = clicks.groupby("news_id")["user_id"].nunique()  # training users per article
    raw_coclick = coclick_edges(clicks)

    seen = split.seen_news
    in_setup = seen | set(split.target["history"].explode().dropna()) | set(split.target["candidates"].explode())

    articles = news[news["news_id"].isin(in_setup)].copy()
    articles["subcategory_id"] = subcategory_id(articles["category"], articles["subcategory"])
    articles["seen_in_training"] = articles["news_id"].isin(seen)

    subcats = (articles[["subcategory_id", "subcategory", "category"]]
               .drop_duplicates("subcategory_id").rename(columns={"subcategory": "name"}))

    article_mentions = mentions[mentions["news_id"].isin(in_setup)]
    entities = (article_mentions.groupby("entity_id")
                .agg(label=("label", "first"), mind_type=("mind_type", lambda t: t.mode().iloc[0]))
                .reset_index())

    tables = {
        "nodes_article": articles[["news_id", "category", "subcategory_id", "title", "seen_in_training"]]
        .reset_index(drop=True),
        "nodes_subcategory": subcats.reset_index(drop=True),
        "nodes_category": pd.DataFrame({"category": sorted(articles["category"].unique())}),
        "nodes_entity": entities,
        "edges_article_in_subcategory": articles[["news_id", "subcategory_id"]].reset_index(drop=True),
        "edges_subcategory_in_category": subcats[["subcategory_id", "category"]].reset_index(drop=True),
        "edges_article_mentions_entity": article_mentions[["news_id", "entity_id", "source", "confidence"]]
        .reset_index(drop=True),
        "edges_article_coclicked_article": raw_coclick,
        "edges_article_coclicked_article_pruned": prune_coclick_edges(raw_coclick, readers, **pruning),
    }
    if entity_types is not None:
        tables.update(entity_type_tables(tables["nodes_entity"], entity_types))
    return tables


def entity_type_tables(entities: pd.DataFrame, entity_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wikidata label on entity nodes, plus entity-type nodes and entity-is-type edges."""
    typed = entities.merge(entity_types[["entity_id", "wikidata_label", "coarse_type"]],
                           on="entity_id", how="left", validate="one_to_one")
    typed["coarse_type"] = typed["coarse_type"].fillna("other")
    return {
        "nodes_entity": typed.drop(columns="coarse_type"),
        "nodes_entity_type": pd.DataFrame({"entity_type": COARSE_TYPES}),
        "edges_entity_is_type": typed[["entity_id", "coarse_type"]].rename(columns={"coarse_type": "entity_type"}),
    }


def load_graph(setup: str) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted((GRAPH_DIR / setup).glob("*.parquet"))}


def main() -> None:
    news = pd.concat([load_news("train"), load_news("dev")]).drop_duplicates("news_id")
    mentions = pd.concat([load_entity_mentions("train"), load_entity_mentions("dev")])
    mentions = mentions.drop_duplicates(["news_id", "entity_id", "source"])

    entity_types = pd.read_parquet(ENTITY_TYPES_FILE) if ENTITY_TYPES_FILE.exists() else None
    if entity_types is None:
        print("No Wikidata cache found: building without entity types (run recsys.data.wikidata)")

    for setup in ("validation", "test"):
        tables = build_graph(load_split(setup), news, mentions, entity_types=entity_types)
        out = GRAPH_DIR / setup
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{setup}: " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))


if __name__ == "__main__":
    main()
