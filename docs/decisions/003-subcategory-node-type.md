# 003 · Subcategory as a node type

**Status:** accepted, pending ablation

## Context

Every article has exactly one category (17–18 values) and one subcategory (~290 values).
Categories are too coarse to separate articles (`news` and `sports` cover 59% of them).
Subcategories are fine-grained (e.g. `football_nfl`, `newspolitics`). For the 293 unseen
test articles without any entity, the subcategory is their only link to the rest of the graph.

## Decision

Model subcategory as its own node type: article → subcategory → category. Subcategory IDs
are `category/subcategory`, because names repeat across categories (e.g. `news` appears
under both `news` and `video`).

## Alternatives considered

- **Subcategory as an article feature (one-hot / embedding) instead of a node.** This is
  simpler and avoids high-degree hub nodes (the largest subcategories have thousands of
  articles). But it gives no message-passing path between articles in the same subcategory.
- **Category nodes only.** Too coarse to be useful as a bridge.

## Consequences

- Subcategory nodes are hubs (thousands of neighbours), so GNN layers need neighbour
  sampling or degree normalisation.
- **To check later:** an ablation comparing (a) subcategory node, (b) subcategory as a
  feature only, (c) neither. Report the difference with the paired bootstrap, with
  particular attention to the unseen-article segment. Revisit this decision if (b) is not
  significantly worse.
- **Judge the ablation on the unseen-article segment.** 293 unseen test articles have no
  co-click and no entity edges, so they are linked to the graph *only* through their
  subcategory. Overall metrics would hide the effect on them.
