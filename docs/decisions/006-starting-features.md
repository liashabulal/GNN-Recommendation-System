# 006 · Starting node features

**Status:** accepted (Milestone 3)

## Context

The GNN needs an input vector for every node type before message passing. Every choice
must work for nodes never seen in the training period (45% of test candidate articles,
and one subcategory that is new on the test day) and must not leak target-day clicks.
Gap measurements are in `reports/milestone3_gaps.md`.

## Decision

| Node type | Starting feature | Dim | Computed |
|---|---|---|---|
| article | `all-MiniLM-L6-v2` sentence embedding of `title + " " + abstract`, L2-normalised | 384 | once, frozen |
| entity | MIND entity vector (TransE on Wikidata) **+ binary `missing` flag** | 100 + 1 | once, frozen |
| subcategory | mean article text embedding over the setup's articles in it | 384 | once per setup, frozen |
| category | mean article text embedding over the setup's articles in it | 384 | once per setup, frozen |
| entity type | learnable embedding | 64 | trained |

Encoder name and dimensions live in `configs/model.toml`. Files are in
`data/processed/features/` (built by `uv run python -m recsys.features.build`).

**Why each choice**

- **Article: a frozen, small, pre-trained text encoder.** Content exists at publication, so
  unseen articles get a meaningful vector immediately; that is the TF-IDF baseline's
  strength, in a denser form. MiniLM-L6 (384 dimensions) encodes all 65,238 articles in about
  2 minutes on a laptop. Freezing it keeps the GNN the only thing being trained, so its
  contribution can be measured, and avoids overfitting a 22M-parameter encoder on about
  230k clicks. A sanity check shows topical neighbours (e.g. a Kenyan Drake trade story's
  nearest articles are other reports of the same trade, at cosine 0.83–0.93).
- **Entity: MIND's vectors.** They are pre-trained on the Wikidata graph (content, not clicks)
  and cover 96.9% of entities. The train and dev `.vec` files give identical vectors for
  all 18,346 shared IDs, so their union is used for both setups.
- **Missing entity vectors, option A: zero vector + `missing` flag.** 1,021 entities (3.1%) have
  no vector. They touch **1.1% of article–entity edges**, and **only 10 unseen test articles**
  have nothing but such entities. With an impact this small, a more elaborate imputation
  can't move the metrics, while an explicit flag lets the model learn to discount these
  inputs instead of trusting an estimate. The entity still gets its coarse type through its
  `IS_TYPE` edge and context from its articles through message passing. **Option B (mean
  vector of co-mentioned entities) is a cheap upgrade if the entity ablation shows these
  entities matter.**
- **Subcategory and category: mean article text embedding.** This gives them a vector in the same
  space as articles and works for the subcategory that is new on the test day
  (`news/newstechnology`, whose nearest subcategory is `news/newsscienceandtechnology`).
  It is leakage-safe because only article content is averaged, never clicks.
- **Entity type: learnable, 64 dimensions.** Only six types, all present in training, so a
  lookup table is fully trained and needs no pre-computed input.

## Alternatives considered

- **Fine-tune the text encoder** (NRMS/UNBERT style): stronger in the MIND literature but
  expensive, and it would blur whether gains come from the graph or the encoder. Possible later.
- **Larger encoders** (e.g. `all-mpnet-base-v2`, 768 dimensions): about 5× slower. Revisit if the
  text features turn out to be the bottleneck.
- **Learnable ID embeddings for articles or subcategories:** transductive. New articles and
  subcategories would get untrained vectors.
- **Option B or C for missing entities** (co-mention mean, type mean): see above. C mostly
  duplicates the entity-type node.

## Consequences

- Each node type has its own input dimension (384 / 101 / 384 / 384 / 64), so the model
  needs a per-type linear projection to the shared hidden size (Part D).
- Changing the text encoder means re-running `recsys.features.build`; the graph tables are
  unaffected.
- **Assumption: subcategory and category features average *all* articles in the setup,
  including target-day articles.** This uses content, not clicks, so no labels leak. But a live
  system would only have the articles published up to the moment of each impression, so this
  is mildly optimistic.
  - **Expected effect:** small for the large subcategories that hold most articles. Those
    average hundreds to thousands of articles, so a day's new articles shift the mean little.
  - **Less so for the long tail:** the median subcategory has only 14 articles, and 130 of 291
    (test setup) have fewer than 10. Articles outside the training period make up 21% of the
    average subcategory mean, weighted by articles (8% at the median), in the test setup.
    That count includes dev history-only articles, not just target-day ones; 7% / 2% in the
    validation setup.
  - **If a model leans heavily on small subcategories,** recompute these means from
    training-period articles only (plus each new article itself) and compare.
- The validation and test feature files differ for the same reason.
