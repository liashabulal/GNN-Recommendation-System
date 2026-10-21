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
| GNN vs embedding baseline | What is the graph worth? Both use the same frozen text features and the same user-history rule. |

**Win criterion (test, final models only).** The GNN is trained with 3 seeds. **Seed 0 is
designated in advance** as the model we would deploy. Each seed's GNN is bootstrapped
*separately* against the baseline (paired, 1,000 resamples), and **all three intervals are
reported**. A model wins only if both hold:
1. The **AUC difference has the same sign in all three seeds**.
2. **Seed 0's 95% interval for the AUC difference excludes zero.**

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

## Alternatives considered

- **Bootstrap on seed-averaged scores** (the mean of the 3 seeds' scores per candidate):
  rejected. Averaging seeds builds an *ensemble*, which is usually stronger than any single
  model, so its interval would describe a model we would not deploy. Bootstrapping each seed
  separately measures the deployable single model, and the sign check across seeds covers
  training variance.
- **A single seed without the sign check:** ignores training variance, which is the larger
  source of noise for a neural model.
- **Requiring all three per-seed intervals to exclude zero:** stricter than needed. Seed 0 is
  fixed in advance, so its interval is a fair test of the deployable model, and the other two
  seeds only need to agree in sign.
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
