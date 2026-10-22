# 009 · Evaluation plan for the GNN milestone

**Status:** accepted (Milestone 3)

## Context

The protocol from Step 2 still applies: time-based splits, per-impression metrics, the paired
bootstrap, and the test-mix weighted row. This record fixes *which* comparisons answer
*which* question, what counts as a win, and how ablations are judged, before any GNN result
is seen.

## Decision

**Comparisons**

| Comparison | Question it answers |
|---|---|
| embedding baseline vs TF-IDF | What is better text representation worth? Frozen MiniLM vectors, user = mean of history embeddings, cosine score, no graph. |
| **GNN vs A0 (trained no-graph model)** | **What is the graph worth? Primary.** Same architecture, loss, training and user encoder; A0 has all edges removed. |
| GNN vs embedding baseline | Secondary: graph *and* training on clicks together. Both use the same frozen text features and the same user-history rule. |
| A0 vs embedding baseline | What is training on clicks worth, without any graph? |

**Why A0 was added (before Part G).** The GNN-vs-embedding comparison is **confounded**: the GNN
is trained on clicks, while the embedding baseline is untrained (cosine of frozen vectors). Its
gain therefore mixes "the graph" with "training on clicks". That may explain why removing
co-click edges (ablation A1) has no measurable effect even on seen articles, where the GNN's
whole gain over the embedding baseline sits. A0 is the same model with all edges removed: each
article vector comes only from its own projected text embedding (through the same
per-node-type layers, L2-normalised), trained identically. It separates the two effects.
**In Part G, A0 is trained on test with 5 seeds (0–4), and GNN vs A0 is the primary comparison
for "the graph helps"**, under the same win criterion. GNN vs embedding stays as a secondary
comparison.

**Win criterion (test, final models only; revised before Part G).** The final GNN is trained
with **5 seeds (0–4)**. Each seed's GNN is bootstrapped *separately* against the baseline
(paired, 1,000 resamples). A model wins only if **every seed's 95% interval for the AUC
difference excludes zero** (all on the same side). Report the **mean difference across seeds,
with the minimum and maximum seed values**, next to each seed's interval.

- **Why revised:** on validation, the reference GNN's AUC varied by **0.016 on the test-mix
  weighted row** across 3 training seeds (0.651 / 0.635 / 0.640), larger than most ablation
  effects. The bootstrap captures data noise (which impressions are sampled) only, not
  training variance. The earlier rule relied on seed 0's interval, which would have rested
  on a single draw from that spread; seed 0 also turned out to be the strongest of three.
  Requiring all five intervals to exclude zero covers both sources of noise.
- **Seed 0** remains the designated deployable model, but its number is reported alongside the
  seed mean, minimum and maximum, not instead of them.

AUC is primary. **nDCG@10 is secondary**: reported with the same bootstrap, but it can't on its
own turn a loss into a win. **MRR and nDCG@5 are reported only.**

**Segments, always reported:**
- all
- new users and known users
- seen articles and unseen articles
- empty history

On validation, the **test-mix weighted** row is added (ADR 008 tuning metric).

**Test-set discipline.** Test is touched **only for final models**. All tuning, early stopping
and ablations are judged on validation; no decision is revisited after looking at test.

**Ablations (validation only), each with the segment that decides it.** Each ablation is
first run with **seed 0 only** and compared with the seed-0 full model (paired bootstrap on its
deciding segment). Let the 95% interval be [lo, hi] with width w = hi − lo:
- **Clear effect, the result stands:** the interval excludes zero, and its nearest end is more
  than w / 4 from zero.
- **Clearly no effect, the result stands:** the interval includes zero, and zero lies in its
  middle half (lo + w/4 ≤ 0 ≤ hi − w/4).
- **Borderline:** anything else (just excluding or just including zero). Rerun with 3 seeds
  and apply the full win criterion above.

| ID | Ablation | Deciding segment | Why that segment |
|---|---|---|---|
| A1 | remove co-click edges | seen articles | co-click edges exist only for seen articles |
| A2 | remove entity edges | unseen articles | for new articles, entities are the main graph link (ADR 002) |
| A3 | remove subcategory nodes | unseen articles | 293 unseen test articles connect only via subcategory (ADR 003) |
| A4 | raw instead of pruned co-click edges | test-mix weighted | global effect of noisy, popularity-heavy edges (ADR 004) |
| A5 | no degree down-weighting on entity → article | test-mix weighted | global effect of hub entities |
| A6 | no graph/example split in training | test-mix weighted | the leakage and cold-start effect of ADR 008 |
| A7 | don't add training-period clicks to history | known users | only known users have such clicks |
| A8 | attention pooling instead of mean in the user encoder | test-mix weighted | global change to every user vector |

**Ablation training budget (revised after round 1).** Ablations train for a **fixed 9 epochs**
(the reference's best epoch) with **no early stopping**, and are compared with the reference at
epoch 9. Every variant then has the same training budget, and only the component differs.
- **Why:** in round 1, early stopping with patience 2 ended 7 of 9 ablations at epochs 2–6.
  At those epochs the reference itself was up to 0.019 AUC below its best, so the
  differences mixed the component with the training length (`reports/gnn_ablations.md`,
  round 1).
- **Known cost:** a variant that needs longer training to converge is disadvantaged at
  9 epochs. If a variant is within noise or borderline at 9 epochs but trained longer under
  early stopping, it is also compared at its own epoch count, against the reference trained
  for the same number of epochs.
- **3 seeds:** a borderline result is rerun with seeds 1 and 2 for *both* the variant and the
  reference (each at 9 epochs), and seed k is compared with seed k.

**Decision rules for the final configuration** (applied to each ablation's deciding segment):
- **Alternative clearly better** (interval excludes zero in its favour): adopt it.
- **Clearly worse:** keep the current design.
- **Within noise:** keep the current design and record "no measurable effect", unless the
  component is costly to run (then consider removing it).
- **R1 (residual text connection):** adopt only if it clearly helps unseen articles **and**
  does not clearly hurt the test-mix weighted row.

**Beyond-accuracy metrics** (defined and computed in Part G, on test, for final models):
- **Freshness exposure:** share of top-5 slots given to unseen articles, compared with their
  share of clicks.
- **Coverage:** share of candidate articles that appear in at least one user's top 5.
- **Diversity:** mean number of distinct subcategories in each user's top 5.

**Future work: attention pooling (A8).** Attention pooling in the user encoder was within noise
at 9 epochs (+0.002 test-mix AUC, interval including zero) and borderline positive at its own
early-stopped length of 13 epochs (+0.005 [+0.001, +0.009], against a 13-epoch reference). It
is the only variant that leaned positive at every training length, so it is a promising
direction. No further runs now; revisit with longer training and several seeds.

## Alternatives considered

- **Bootstrap on seed-averaged scores** (the mean of the 3 seeds' scores per candidate):
  rejected. Averaging seeds builds an *ensemble*, which is usually stronger than any single
  model, so its interval would describe a model we would not deploy. Bootstrapping each seed
  separately measures the deployable single model, and the sign check across seeds covers
  training variance.
- **A single seed without the sign check:** ignores training variance, which is the larger
  source of noise for a neural model.
- **Seed 0's interval plus a sign check across 3 seeds** (the original rule): replaced before
  Part G. With a seed spread of 0.016, one seed's interval says little about the model
  family, and a sign check doesn't bound the size of the effect. The ablation rule (seed 0
  first, 3 seeds when borderline) is kept for validation-only decisions, where the cost of a
  wrong call is lower.
- **3 seeds for every ablation:** about 3× the compute for results that are usually clear-cut.
  The borderline rule spends the extra seeds only where they can change the conclusion.
- **Judging ablations on "all":** would hide effects that only exist for a segment (e.g. A3 on
  the 293 subcategory-only articles).

## Consequences

- Every final number on test comes with a CI and a per-seed sign check, and segment results
  are always shown next to the overall ones.
- An ablation "matters" only on its deciding segment, so a change in other segments is
  reported but doesn't decide.
- Because test is reserved, the validation test-mix row is the only guide during development.
  Its agreement with test was good for the baselines (Step 2).
