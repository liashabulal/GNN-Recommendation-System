# 001 · Article-centred inductive graph with no user nodes

**Status:** accepted (Milestone 2)

## Context

On the test split, 88% of impressions come from users with no activity in the
training period (9–14 Nov), while 97% of users have a non-empty history snapshot.
A user node would have no edges and an untrained embedding for almost every test user.
45% of test candidate articles are also unseen in training, so articles must be
representable from content alone.

## Decision

The graph contains only content-side nodes: **article, subcategory, category, entity**
(and entity types from Wikidata in Part B). Users are not nodes. A user is represented at
scoring time by aggregating the learned representations of the articles in their history
snapshot, which works the same way for known and new users.

## Alternatives considered

- **User–article bipartite graph (LightGCN / PinSage style).** This is the strongest choice
  for warm users, but it is transductive: new users have no node, which covers 88% of test
  impressions. Fold-in tricks would still depend on the history, which is what we use anyway.
- **User nodes added at inference time from the history.** This is equivalent to aggregating
  history articles, but adds message-passing cost and implementation complexity for no
  information gain.

## Consequences

- Known users and new users are scored by the same code path, so the gap between their
  segments measures only data effects.
- No per-user parameters, so individual preferences beyond what the history shows cannot be
  learned. If the known-user segment lags behind ID-based models later, revisit this.
- Collaborative signal enters only through article–article co-click edges (ADR 002).
