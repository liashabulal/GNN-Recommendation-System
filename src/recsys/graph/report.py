"""Summarise the graph tables. Run with `uv run python -m recsys.graph.report`.

Writes reports/graph_stats.md: counts per node/edge type, the co-click weight
distribution, and cold-start coverage of unseen test articles.
"""

import pandas as pd

from recsys.data.splits import load_split
from recsys.graph.build import load_graph, load_pruning_config, training_clicks
from recsys.paths import REPORTS_DIR


def counts_table() -> pd.DataFrame:
    return pd.DataFrame({setup: {k: len(v) for k, v in load_graph(setup).items()}
                         for setup in ("validation", "test")})


def weight_distribution(edges: pd.DataFrame) -> pd.Series:
    w = edges["weight"]
    return pd.Series({
        "edges": len(w),
        "median": w.median(), "p90": w.quantile(0.9), "p99": w.quantile(0.99), "max": w.max(),
        "share weight = 1": (w == 1).mean(),
        "share weight >= 5": (w >= 5).mean(),
    })


def unseen_coverage(graph: dict, split) -> pd.Series:
    """Unseen target candidate articles: which edge types connect them to the graph?"""
    unseen = set(split.target["candidates"].explode()) - split.seen_news
    cc = graph["edges_article_coclicked_article"]
    has_cc = unseen & (set(cc["src"]) | set(cc["dst"]))
    has_entity = unseen & set(graph["edges_article_mentions_entity"]["news_id"])
    has_subcat = unseen & set(graph["edges_article_in_subcategory"]["news_id"])
    no_cc = unseen - has_cc
    return pd.Series({
        "unseen target candidate articles": len(unseen),
        "with >= 1 co-click edge": len(has_cc),
        "zero co-click but >= 1 entity or subcategory edge": len(no_cc & (has_entity | has_subcat)),
        "zero co-click, >= 1 entity edge": len(no_cc & has_entity),
        "zero co-click, subcategory edge only (no entity)": len(no_cc & has_subcat - has_entity),
        "fully isolated": len(no_cc - has_entity - has_subcat),
    })


def pair_concentration(split) -> pd.Series:
    """Share of raw co-click pair contributions coming from the users with the longest histories.

    A user with n distinct clicked articles contributes n(n-1)/2 pairs (one unit of weight each).
    """
    clicks = training_clicks(split.train)
    n = clicks.groupby("user_id").size()
    pairs = n * (n - 1) / 2
    history_len = split.train.drop_duplicates("user_id").set_index("user_id")["history"].map(len)
    ranked = pairs.reindex(history_len.sort_values(ascending=False, kind="stable").index)
    total = ranked.sum()
    return pd.Series({
        "pair contributions": total,
        "top 1% of users by history length": ranked.iloc[: round(len(ranked) * 0.01)].sum() / total,
        "top 10% of users by history length": ranked.iloc[: round(len(ranked) * 0.10)].sum() / total,
    })


def pruning_coverage(graph: dict) -> pd.Series:
    articles = graph["nodes_article"]
    def share(edges):
        return articles["news_id"].isin(set(edges["src"]) | set(edges["dst"])).mean()
    seen = articles[articles["seen_in_training"]]
    def share_seen(edges):
        return seen["news_id"].isin(set(edges["src"]) | set(edges["dst"])).mean()
    raw, pruned = graph["edges_article_coclicked_article"], graph["edges_article_coclicked_article_pruned"]
    return pd.Series({
        "raw edges": len(raw),
        "pruned edges": len(pruned),
        "articles with >= 1 co-click edge (raw)": share(raw),
        "articles with >= 1 co-click edge (pruned)": share(pruned),
        "seen articles with >= 1 co-click edge (raw)": share_seen(raw),
        "seen articles with >= 1 co-click edge (pruned)": share_seen(pruned),
    })


def neighbour_examples(graph: dict, categories=("news", "sports", "foodanddrink"), n=5) -> str:
    """Top-n neighbours by raw weight vs by normalised weight (pruned) for one article per category.

    Example article per category: the one with the most pruned edges among articles whose
    reader count is closest to the category's 90th percentile (popular but not the top hit).
    """
    articles = graph["nodes_article"].set_index("news_id")
    raw, pruned = graph["edges_article_coclicked_article"], graph["edges_article_coclicked_article_pruned"]

    def neighbours(edges, article, by):
        e = edges[(edges["src"] == article) | (edges["dst"] == article)]
        e = e.assign(other=e["src"].where(e["dst"] == article, e["dst"]))
        e = e.sort_values([by, "weight"], ascending=False).head(n)
        return [f"{articles.loc[o, 'title']} *({articles.loc[o, 'subcategory_id']}, {by}={w:.3g})*"
                for o, w in zip(e["other"], e[by])]

    degree = pd.concat([pruned["src"], pruned["dst"]]).value_counts()
    raw_degree = pd.concat([raw["src"], raw["dst"]]).value_counts()
    out = []
    for cat in categories:
        candidates = articles[(articles["category"] == cat) & articles.index.isin(degree.index)].copy()
        candidates["raw_degree"] = raw_degree.reindex(candidates.index).fillna(0)
        target = candidates["raw_degree"].quantile(0.9)
        pick = candidates.assign(dist=(candidates["raw_degree"] - target).abs(),
                                 deg=degree.reindex(candidates.index)).sort_values(["dist", "deg"],
                                                                                  ascending=[True, False]).index[0]
        out.append(f"### {articles.loc[pick, 'title']}  \n`{pick}` · {articles.loc[pick, 'subcategory_id']}\n")
        out.append("**Top by raw weight (shared users):**\n" + "\n".join(f"{i}. {t}" for i, t in
                                                                        enumerate(neighbours(raw, pick, 'weight'), 1)))
        out.append("**Top by normalised weight (pruned set):**\n" + "\n".join(
            f"{i}. {t}" for i, t in enumerate(neighbours(pruned, pick, 'norm_weight'), 1)))
    return "\n\n".join(out)


def fmt(df: pd.DataFrame) -> str:
    """Markdown table; shares as percentages, everything else as integers."""
    cell = lambda v: f"{v:.1%}" if 0 < v < 1 else f"{v:,.0f}"
    lines = ["| | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    lines += [f"| {idx} | " + " | ".join(cell(v) for v in row) + " |" for idx, row in df.iterrows()]
    return "\n".join(lines)


def main() -> None:
    counts = counts_table()
    weights = pd.DataFrame({s: weight_distribution(load_graph(s)["edges_article_coclicked_article"])
                            for s in ("validation", "test")})
    graphs = {s: load_graph(s) for s in ("validation", "test")}
    splits = {s: load_split(s) for s in ("validation", "test")}
    coverage = pd.DataFrame({s: unseen_coverage(graphs[s], splits[s]) for s in graphs})
    concentration = pd.DataFrame({s: pair_concentration(splits[s]) for s in graphs})
    pruning = pd.DataFrame({s: pruning_coverage(graphs[s]) for s in graphs})
    cfg = load_pruning_config()

    text = "\n\n".join([
        "# Graph statistics\n\nGenerated by `uv run python -m recsys.graph.report`.",
        "## Node and edge counts\n\n" + fmt(counts),
        "## Co-click weight (number of users who clicked both articles)\n\n" + fmt(weights),
        "## Unseen target-day candidate articles\n\n" + fmt(coverage),
        "## Who creates the co-click pairs?\n\nShare of raw pair contributions (each user adds "
        "n(n-1)/2 for n clicked articles) from the users with the longest histories.\n\n" + fmt(concentration),
        f"## Pruned co-click edges (ADR 004: min_users={cfg['min_users']}, top_k={cfg['top_k']})\n\n"
        + fmt(pruning),
        "## Example neighbourhoods (test setup)\n\n" + neighbour_examples(graphs["test"]),
    ]) + "\n"
    (REPORTS_DIR / "graph_stats.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
