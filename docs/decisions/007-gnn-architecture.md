# 007 · GNN architecture

**Status:** accepted (Milestone 3)

## Context

ADR 001 fixes an article-centred inductive graph with no user nodes. ADR 006 fixes the
starting features. This record covers the model that turns them into scores. Code:
`src/recsys/models/gnn.py`, `src/recsys/models/history.py`; settings in `configs/model.toml`.

## Decision

**Message-passing layer: a custom `WeightedMeanConv`.** For one edge type it computes
`Σ w·x_j / Σ w` over the (sampled) neighbours, followed by a linear map. When the edge type
has no weights, all `w = 1`, which gives a plain mean. Weights are normalised per destination
node on whatever subgraph was sampled. The layer has no self term.

None of PyG 2.8.1's built-in layers fitted:

| Layer | Problem |
|---|---|
| `GraphConv` | Accepts `edge_weight`, but its "mean" divides `Σ w·x` by the neighbour *count*, which is not a weighted mean. It also adds its own root (self) weight for *every* edge type. |
| `SAGEConv` | Mean aggregation and an optional root weight, but no `edge_weight`. |
| `GCNConv` | Accepts `edge_weight`, but works only within one node type, so it can't be used inside `HeteroConv` for article→entity and similar edges. |

**Layer structure (`HeteroLayer`).** One `WeightedMeanConv` per edge type, outputs summed per
destination node type (`HeteroConv`, `aggr="sum"`), plus the node's own previous vector
through a separate weight matrix, one per node type, added once.

**Encoder.**
1. A linear projection per node type to **128** dimensions. Entity types get a learnable
   64-dim embedding first.
2. **Two** `HeteroLayer`s, with **ReLU after layer 1 only**.
3. **L2-normalised article vectors**, so the score is a cosine between the user profile
   direction and the article.

The encoder has about 630k parameters.

**Neighbour sampling.** Up to **10** neighbours per edge type at hop 1 and **5** at hop 2, using
PyG's `NeighborLoader`. Edge weights travel with the sampled edges. A test checks that
sampling with all neighbours reproduces the full-graph vectors exactly.

**User encoder.** For an impression at time t, the history is the **last 50 items** of the
history column (which runs oldest → newest; `reports/milestone3_gaps.md`) followed by the
user's **training-period clicks made before t**. Each article appears once, at its most
recent position. The user vector is the **mean of those articles' GNN vectors**. A user with
nothing gets a **learned default vector** (2.7% of test impressions).

**Scorer.** The dot product of the user vector and the candidate article vector.

**Environment pin.** The neighbour sampler needs `pyg-lib` or `torch-sparse`. `pyg-lib`
publishes macOS arm64 wheels only up to PyTorch 2.12 (version 0.7.0), so **torch is pinned to
2.12.\*** and `pyg-lib==0.7.0` comes from PyG's wheel index (`pyproject.toml`). The article
text embeddings were computed under torch 2.14. Re-encoding 500 random articles under 2.12
gave a cosine of 1.000000 against the saved vectors, so the feature files were kept.

## Alternatives considered

- **`GraphConv` with weights pre-normalised per destination.** This works on the full graph but
  not after sampling, where the kept neighbours' weights no longer sum to 1. It would still
  add a root term per edge type.
- **Attention (`GATConv`, `HGTConv`).** More expressive, but no `edge_weight` input, and harder to
  attribute gains. Attention *pooling* for the user encoder is ablation A8 (ADR 009).
- **A pure-PyTorch sampler** instead of pinning torch: more code to own and test. Compiling
  `torch-sparse` from source: slow and fragile on macOS.
- **User nodes:** rejected in ADR 001.

## Consequences

- An edge type with no incoming edges for a node contributes zeros. Unseen articles therefore
  get their vector from their own text, subcategory and entity edges only (tested).
- Upgrading torch requires a matching `pyg-lib` wheel. Check
  `https://data.pyg.org/whl/torch-<version>+cpu.html` and update the index URL and pins together.
