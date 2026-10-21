# 008 · Training design

**Status:** accepted (Milestone 3); dropout and weight-decay values to be tuned on validation

## Context

The GNN (ADR 007) is trained from clicks. The co-click edges are also built from clicks
(ADR 002). If the same clicks both create edges and serve as training labels, the model can
learn to read the answer off the graph. That shortcut is unavailable on the target day, where
the clicked articles are mostly unseen and have no co-click edges.

## Decision

**Loss.** Softmax cross-entropy over **1 clicked article and 4 non-clicked articles from the
same impression** (the standard MIND "1 + K" setup, K = 4).
- Each click is one example, so an impression with several clicks yields several examples.
- An impression with fewer than 4 non-clicked candidates samples them with replacement.

**Graph period vs training-example period.** Training uses an earlier graph than the
impressions it learns from:

| Setup | Training graph (edges, seen flags) | Training examples | Scoring graph for the target day | Target day |
|---|---|---|---|---|
| validation | 9–11 Nov | impressions of 12–13 Nov | 9–13 Nov | 14 Nov |
| test | 9–12 Nov | impressions of 13–14 Nov | 9–14 Nov | 15 Nov |

- **The leakage this prevents.** A training click must not create the co-click edge used to
  predict it. If the 13 Nov click on article X were in the graph, X would already be linked to
  the user's other clicks, and the model would learn to trust co-click edges that can't exist
  for target-day articles.
- **The cold-start practice it gives.** Articles that first appear in the example period have
  no co-click edges in the training graph, exactly like unseen target-day articles. So the
  model learns to score articles from content edges alone, on real clicks.
- **Scoring graph.** For the target day, the graph is rebuilt from **all days before it**
  (9–13 Nov for validation, 9–14 Nov for test), as a deployed system would rebuild its graph
  daily. Part F also scores validation once with the 9–11 Nov training graph, to measure what
  the daily rebuild is worth.
- **Recomputed for each graph period:** seen/unseen flags, co-click edges and their pruning
  (ADR 004), the 1 / log(1 + degree) entity weights, and the subcategory/category means
  (ADR 006: articles that exist before the period's end).
- **User histories** for training examples follow ADR 007. They use the history snapshot plus
  all clicks from 9 Nov (graph period and earlier example-period impressions) made strictly
  before the impression time, which is the same rule the scoring side uses. The history
  snapshot predates 9 Nov, so it is safe for every period.

**Temperature.** Article vectors have unit length and user vectors are means of them, so
dot-product scores stay within about ±1. That range is too narrow for a 5-way softmax to give
useful gradients. The loss therefore uses `exp(s) · score` with a **learnable log-scale `s`**
(initialised at log 10). Every score is multiplied by the same positive number, so
**rankings, and therefore all evaluation metrics, are unchanged**; only the training signal is.

**Optimisation.**
- Adam, learning rate 0.001.
- Dropout: starting value 0.2, applied after the input projection and after layer 1.
- Weight decay: starting value 1e-5.
- Both values are tuned on validation.

**Early stopping.**
- Evaluate on the validation day after every epoch, using the **test-mix weighted AUC**
  (ADR 009). Stop with **patience 2** and keep the best epoch.
- The test refit has no held-out day, so it trains for **exactly the best validation epoch
  count**. Its example period (13–14 Nov, about 62k impressions) is close in size to
  validation's (12–13 Nov, about 65k), so the epoch count transfers.

**Seeds.** Each final configuration is trained with **3 seeds**.

## Alternatives considered

- **One period for both graph and examples:** this is ablation A6, expected to look better in
  training and worse on the target day.
- **Random edge masking** (hold out a share of co-click edges as targets): this simulates
  missing edges but not the *time* structure. The target day's problem is articles that are
  new, not edges that are hidden.
- **BPR (pairwise) loss:** works, but softmax over impression negatives matches the evaluation
  (ranking within an impression) and is the MIND standard.
- **Random negatives from the whole catalogue:** easier negatives, and they don't reflect what
  was shown together.

## Consequences

- Each setup needs two graphs (training and scoring) built from different date ranges, so the
  graph build gets a "graph period" parameter separate from the setup's training period.
- The training graph is shorter (3 or 4 days instead of 5 or 6), so it is sparser than the
  scoring graph. The one-off comparison in Part F shows how much the daily rebuild matters.
- Results are reported as mean ± spread over 3 seeds, and wins follow ADR 009.
