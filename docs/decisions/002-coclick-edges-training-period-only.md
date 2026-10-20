# 002 · Co-click edges from training-period clicks only

**Status:** accepted (Milestone 2)

## Context

Co-click edges (two articles clicked by the same user) carry collaborative signal into an
article-only graph (ADR 001). They are derived from interactions, so they can leak
target-day information: target labels directly, or target users' histories, which would
make the graph depend on who happens to be scored.

## Decision

For each setup, co-click edges are built only from:

1. clicks in **training-period impressions** (validation: 9–13 Nov; test: 9–14 Nov), and
2. the **history snapshots of training-period users** (users with at least one training
   impression). These snapshots predate the log window.

The weight of an edge is the **number of distinct users** who clicked both articles. Target
users' histories are used only to build their profiles at scoring time; target labels are
never used. Unit tests in `tests/test_graph.py` check both rules.

## Alternatives considered

- **Also use target users' histories for edges.** These are pre-window and technically not
  label leakage, but the graph would change with the set of users being scored, and test
  numbers would not reflect a graph built before the test day.
- **Weight by click count, or by PMI / Jaccard.** Raw user counts are simpler and
  transparent. Normalised weights can be computed from these tables later without
  rebuilding.
- **Drop weight-1 edges at build time.** Deferred: 71% of the 16.1M test-setup edges have
  weight 1. The full table is kept, and pruning is a modelling or loading choice (e.g. the
  Neo4j load), not a storage one.

## Consequences

- Unseen target articles have **zero** co-click edges by construction (test: all 2,442).
  They depend entirely on entity and subcategory edges, which every one of them has
  (2,149 with at least one entity; 293 subcategory only).
- The validation and test graphs differ (the test graph includes 14 Nov), so every model is
  refit on the test graph, as with the baselines.
