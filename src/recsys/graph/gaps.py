"""Milestone 3, Part A: measure feature gaps before choosing starting features.

Run with `uv run python -m recsys.graph.gaps`. Writes reports/milestone3_gaps.md.

1. Subcategories / categories that appear in target-day candidate articles but in no
   training-period article (so no training signal will have touched them).
2. Entities without a MIND entity vector (entity_embedding.vec), and how much of the
   graph and of the unseen target-day articles depend on them.
"""

import pandas as pd

from recsys.data.mind import explode_impressions
from recsys.data.splits import load_split
from recsys.graph.build import load_graph
from recsys.graph.report import fmt
from recsys.paths import REPORTS_DIR, mind_split_dir

SETUPS = ("validation", "test")


def mind_vector_ids() -> set[str]:
    """Entity IDs with a MIND entity vector, from both splits' entity_embedding.vec.

    The vectors are pre-trained on the Wikidata knowledge graph (content, not clicks),
    so using the union of both files is leakage-safe for either setup.
    """
    ids = set()
    for split in ("train", "dev"):
        with open(mind_split_dir(split) / "entity_embedding.vec") as f:
            ids |= {line.split("\t", 1)[0] for line in f}
    return ids


def unseen_taxonomy(graph: dict, split) -> tuple[pd.Series, str]:
    articles = graph["nodes_article"].set_index("news_id")
    seen = articles[articles["seen_in_training"]]
    cands = explode_impressions(split.target)[["impression_id", "news_id", "clicked"]]
    cands = cands.join(articles[["category", "subcategory_id"]], on="news_id")

    new_cats = sorted(set(cands["category"]) - set(seen["category"]))
    new_subcats = sorted(set(cands["subcategory_id"]) - set(seen["subcategory_id"]))
    in_new_sub = cands["subcategory_id"].isin(new_subcats)
    in_new_cat = cands["category"].isin(new_cats)

    per_sub = (cands[in_new_sub].groupby("subcategory_id")
               .agg(articles=("news_id", "nunique"), impressions=("impression_id", "nunique"),
                    clicks=("clicked", "sum")))
    summary = pd.Series({
        "new categories": len(new_cats),
        "new subcategories": len(new_subcats),
        "target impressions with a candidate in a new subcategory": cands.loc[in_new_sub, "impression_id"].nunique(),
        "share of target impressions": cands.loc[in_new_sub, "impression_id"].nunique()
        / cands["impression_id"].nunique(),
        "candidate rows in a new subcategory": int(in_new_sub.sum()),
        "clicks on articles in a new subcategory": int(cands.loc[in_new_sub, "clicked"].sum()),
        "target impressions with a candidate in a new category": cands.loc[in_new_cat, "impression_id"].nunique(),
    })
    names = (f"New categories: {', '.join(f'`{c}`' for c in new_cats) or 'none'}.\n\n"
             + ("New subcategories (articles / impressions / clicks):\n\n"
                + "\n".join(f"- `{s}`: {r.articles} / {r.impressions:,} / {r.clicks}"
                            for s, r in per_sub.sort_values("impressions", ascending=False).iterrows())
                if len(per_sub) else "New subcategories: none."))
    return summary, names


def missing_entity_vectors(graph: dict, split, vector_ids: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    entities = graph["nodes_entity"].set_index("entity_id")
    mentions = graph["edges_article_mentions_entity"]
    missing = set(entities.index) - vector_ids

    # Unseen target-day candidates whose entities (at least one) are all missing a vector.
    unseen = set(split.target["candidates"].explode()) - split.seen_news
    m = mentions[mentions["news_id"].isin(unseen)].assign(missing=lambda d: d["entity_id"].isin(missing))
    per_article = m.groupby("news_id")["missing"].agg(["all", "any"])

    summary = pd.Series({
        "entities": len(entities),
        "entities without a MIND vector": len(missing),
        "share of entities": len(missing) / len(entities),
        "article-entity edges": len(mentions),
        "edges involving a missing entity": int(mentions["entity_id"].isin(missing).sum()),
        "share of edges": mentions["entity_id"].isin(missing).mean(),
        "unseen target candidates with >= 1 entity": len(per_article),
        "... whose entities are ALL missing a vector": int(per_article["all"].sum()),
        "... with at least one missing entity": int(per_article["any"].sum()),
    })
    top = (mentions[mentions["entity_id"].isin(missing)].groupby("entity_id")["news_id"].nunique()
           .sort_values(ascending=False).head(20).rename("articles").to_frame()
           .join(entities[["wikidata_label", "label", "mind_type"]]))
    types = graph["edges_entity_is_type"].set_index("entity_id")["entity_type"]
    top["coarse_type"] = types.reindex(top.index)
    top["label"] = top["wikidata_label"].fillna(top["label"])
    return summary, top.drop(columns="wikidata_label")


HISTORY_ORDER_SOURCE = "https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md"


def history_position_freshness() -> pd.Series:
    """Share of history items also shown as candidates in 9-15 Nov, by position decile.

    News goes stale within days, so recent clicks are far more likely to still be shown in
    the log window. A share rising from the start to the end of the list means oldest -> newest.
    Users with at least 10 history items; one snapshot per user.
    """
    from recsys.data.mind import load_behaviors

    behaviors = pd.concat([load_behaviors("train"), load_behaviors("dev")])
    shown = set(explode_impressions(behaviors)["news_id"])
    histories = behaviors.drop_duplicates("user_id")["history"]
    histories = histories[histories.map(len) >= 10]
    rows = [(min(10 * i // len(h), 9), a in shown) for h in histories for i, a in enumerate(h)]
    return pd.DataFrame(rows, columns=["decile", "shown"]).groupby("decile")["shown"].mean()


def main() -> None:
    vector_ids = mind_vector_ids()
    sections = ["# Milestone 3 · feature gaps\n\nGenerated by `uv run python -m recsys.graph.gaps`. "
                "\"Training-period articles\" are the articles seen in the setup's training period "
                "(in a training history or candidate list); \"target-day\" means the setup's scored impressions."]

    tax = {s: unseen_taxonomy(load_graph(s), load_split(s)) for s in SETUPS}
    sections.append("## 1. Categories and subcategories absent from training-period articles\n\n"
                    + fmt(pd.DataFrame({s: v[0] for s, v in tax.items()}))
                    + "".join(f"\n\n**{s}.** {v[1]}" for s, v in tax.items()))

    ent = {s: missing_entity_vectors(load_graph(s), load_split(s), vector_ids) for s in SETUPS}
    sections.append(f"## 2. Entities without a MIND entity vector\n\nVectors: union of both splits' "
                    f"`entity_embedding.vec` ({len(vector_ids):,} IDs; pre-trained on Wikidata, so content-only).\n\n"
                    + fmt(pd.DataFrame({s: v[0] for s, v in ent.items()})))
    top = ent["test"][1]
    sections.append("**20 most-mentioned entities without a vector (test setup, by distinct articles):**\n\n"
                    "| entity | label | coarse type | MIND type | articles |\n|---|---|---|---|---|\n"
                    + "\n".join(f"| {i} | {r.label} | {r.coarse_type} | {r.mind_type} | {r.articles} |"
                                for i, r in top.iterrows()))

    sections.append(
        "## Data quality: entity-linking errors in MIND\n\n"
        "MIND's entity links were produced automatically, and some point at the wrong Wikidata item. "
        "Seen among the most-mentioned entities without a vector (test setup):\n\n"
        "- **Q48834885 \"October 2007 California wildfires\"** is linked from 29 articles published in "
        "October–November 2019, which are about the 2019 fires. The surface text (e.g. \"California wildfires\") "
        "was resolved to a 2007 event.\n"
        "- **Q6841333 \"Middle East Economic Survey\"** (a trade publication) is linked from 53 articles, "
        "far more than such a niche publication plausibly gets, which suggests an ambiguous surface form "
        "resolved to the wrong item.\n\n"
        "Other suspicious links in the same list: \"50S\" (Q1088533) in 28 articles, and "
        "\"Democratic Party presidential debates, 2008\" (Q5255734) in 22 articles from 2019.\n\n"
        "Impact: these entities lack MIND vectors and get the zero vector + missing flag (ADR 006), so the "
        "wrong identity mostly matters through their edges. They connect unrelated articles only if many "
        "articles share the same wrong link, which the 1 / log(1 + degree) edge weight dampens. Not corrected; "
        "worth revisiting if the entity ablation shows entities matter.")

    fresh = history_position_freshness()
    sections.append(
        "## 3. Is the history column in time order?\n\n"
        "**Yes, oldest to newest.** The MIND documentation states the order but not the direction:\n\n"
        "> History: The news click history (ID list of clicked news) of this user before this impression. "
        "The clicked news articles are ordered by time.\n\n"
        f"Source: MIND dataset introduction, `behaviors.tsv` section ({HISTORY_ORDER_SOURCE}).\n\n"
        "**Direction, checked empirically.** A user's history is one fixed snapshot taken before the log "
        "window: it is identical in train and dev for all 5,943 users in both. So the direction is inferred "
        "from freshness instead. Recent clicks are far more likely to still be shown as candidates in "
        "9–15 Nov, and that share rises steadily along the list:\n\n"
        "| position in history (decile, 0 = first item) | " + " | ".join(str(i) for i in fresh.index) + " |\n"
        + ("|---" * (len(fresh) + 1)) + "|\n"
        + "| share also shown in 9–15 Nov | " + " | ".join(f"{v:.0%}" for v in fresh) + " |\n\n"
        f"The last decile is {fresh.iloc[-1] / fresh.iloc[0]:.0f}× the first, so the **last items are the most recent**. "
        "\"50 most recent\" therefore means the last 50 history items, followed by any training-period clicks "
        "made before the impression time.")

    (REPORTS_DIR / "milestone3_gaps.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
