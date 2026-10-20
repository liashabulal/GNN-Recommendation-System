# 004 · Pruned, normalised co-click edges

**Status:** accepted (Milestone 2); the raw edge table is kept alongside

## Context

The raw co-click table (ADR 002) has 16.1M edges in the test setup. 71% are shared by a
single user, and 74% of all pair contributions come from the top 10% of users by
history length (31% from the top 1%). A single heavy reader with 500 clicks adds about
125k pairs. Ranking neighbours by raw shared-user counts surfaces the same viral
articles everywhere (e.g. one celebrity story is a top-5 neighbour of a history
column, an NFL trade and a cornbread recipe). See `reports/graph_stats.md`.

## Decision

Build `edges_article_coclicked_article_pruned` from the raw table:

1. **min_users = 2**: drop pairs shared by only one user, which are mostly noise from heavy readers.
2. **Normalise:** `norm_weight = weight / sqrt(readers_a * readers_b)`, where readers are the
   training-period users who clicked the article or had it in their history (cosine
   similarity of reader sets).
3. **top_k = 20**: keep an edge if it is among the 20 highest `norm_weight` edges of
   *either* endpoint, so every article that has any qualifying edge keeps at least one.

`min_users` and `top_k` live in `configs/graph.toml`. Both the raw and the pruned tables are written.

## Alternatives considered

- **Raw counts with a top-k only.** Still dominated by popular articles; the examples showed
  off-topic viral neighbours.
- **PMI / Jaccard.** Similar intent. PMI over-rewards rare pairs, and Jaccard needs a union
  size per pair. Cosine is simple and standard for item–item collaborative filtering.
- **Down-weight heavy users (e.g. 1/log(n) per user) instead of normalising articles.** Worth
  trying later as an alternative or an addition; it addresses the concentration directly.
- **Keep an edge only if it is in the top-k of both endpoints (mutual kNN).** Sparser, and
  it would cut off long-tail articles whose neighbours are popular.

## Consequences

- Test setup: 16.1M → 229k edges (≈70× smaller), which is easy to load into Neo4j and to
  message-pass over.
- The share of seen articles with at least one co-click edge drops from 78% to 42%. This is
  entirely due to `min_users = 2` (the either-endpoint top-k never removes an article's last
  edge). Those articles still connect through subcategory and entity edges, like unseen ones.
- `min_users`, `top_k` and raw-vs-pruned are hyperparameters to tune on validation with the
  test-mix metric once a graph model exists.
