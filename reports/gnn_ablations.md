# GNN ablations (validation, 14 Nov)

## Round 3: trained no-graph model (A0) and seen flag (A9)

**A0** has the GNN's architecture, loss, training (fixed 9 epochs) and user encoder, but every edge type is empty, so each article vector comes only from its own projected text embedding (through the same per-node-type layers, L2-normalised). It separates *the graph* from *training on clicks* (ADR 009).

**GNN − A0** (AUC; per seed k: model seed k vs model seed k)

| segment | mean | min to max | seeds whose CI excludes 0 | per seed: difference [95% CI] |
|---|---|---|---|---|
| all | **-0.0032** | -0.0114 to +0.0088 | 3/3 | +0.0088 [+0.0065, +0.0112] · -0.0068 [-0.0093, -0.0043] · -0.0114 [-0.0138, -0.0090] |
| test-mix weighted | **-0.0001** | -0.0067 to +0.0103 | 2/3 | +0.0103 [+0.0050, +0.0156] · -0.0038 [-0.0093, +0.0016] · -0.0067 [-0.0121, -0.0013] |
| user=known | **-0.0039** | -0.0125 to +0.0084 | 3/3 | +0.0084 [+0.0060, +0.0110] · -0.0075 [-0.0101, -0.0050] · -0.0125 [-0.0150, -0.0099] |
| user=new | **+0.0005** | -0.0059 to +0.0106 | 1/3 | +0.0106 [+0.0045, +0.0166] · -0.0033 [-0.0094, +0.0029] · -0.0059 [-0.0120, +0.0002] |
| history=empty | **+0.0109** | +0.0003 to +0.0222 | 0/3 | +0.0222 [-0.0128, +0.0561] · +0.0101 [-0.0268, +0.0512] · +0.0003 [-0.0349, +0.0348] |
| article=seen | **-0.0046** | -0.0087 to +0.0006 | 2/3 | +0.0006 [-0.0022, +0.0036] · -0.0057 [-0.0087, -0.0028] · -0.0087 [-0.0117, -0.0057] |
| article=unseen | **+0.0012** | -0.0076 to +0.0148 | 2/3 | +0.0148 [+0.0104, +0.0189] · -0.0076 [-0.0120, -0.0031] · -0.0036 [-0.0077, +0.0004] |

*Value of the graph* (primary comparison in Part G).

**A0 − embedding baseline** (AUC; per seed k: model seed k vs model seed k)

| segment | mean | min to max | seeds whose CI excludes 0 | per seed: difference [95% CI] |
|---|---|---|---|---|
| all | **+0.0160** | +0.0134 to +0.0200 | 3/3 (same side) | +0.0134 [+0.0099, +0.0166] · +0.0146 [+0.0113, +0.0179] · +0.0200 [+0.0165, +0.0233] |
| test-mix weighted | **+0.0153** | +0.0121 to +0.0200 | 3/3 (same side) | +0.0139 [+0.0062, +0.0213] · +0.0121 [+0.0045, +0.0191] · +0.0200 [+0.0125, +0.0275] |
| user=known | **+0.0162** | +0.0133 to +0.0200 | 3/3 (same side) | +0.0133 [+0.0097, +0.0174] · +0.0152 [+0.0116, +0.0191] · +0.0200 [+0.0164, +0.0239] |
| user=new | **+0.0152** | +0.0117 to +0.0200 | 3/3 (same side) | +0.0140 [+0.0052, +0.0224] · +0.0117 [+0.0030, +0.0199] · +0.0200 [+0.0115, +0.0287] |
| history=empty | **+0.0406** | +0.0312 to +0.0467 | 1/3 | +0.0439 [-0.0014, +0.0895] · +0.0312 [-0.0151, +0.0807] · +0.0467 [+0.0014, +0.0940] |
| article=seen | **+0.0370** | +0.0356 to +0.0389 | 3/3 (same side) | +0.0366 [+0.0327, +0.0406] · +0.0356 [+0.0317, +0.0396] · +0.0389 [+0.0351, +0.0429] |
| article=unseen | **-0.0129** | -0.0175 to -0.0063 | 3/3 (same side) | -0.0175 [-0.0226, -0.0119] · -0.0148 [-0.0201, -0.0095] · -0.0063 [-0.0114, -0.0010] |

*Value of training on clicks, without a graph.*

**What A0 shows.**
- **The graph adds nothing measurable on validation.** GNN − A0 is −0.000 on the test-mix weighted row on average, with seed 0 favouring the graph (+0.010) and seeds 1–2 favouring A0 (−0.004, −0.007). Only seed 0 showed a graph advantage, and seed 0 was already known to be the strongest reference run.
- **All of the GNN's gain over the embedding baseline comes from training on clicks.** A0 − embedding is +0.015 test-mix, +0.037 on seen articles, all 3 seeds clear of zero. This explains A1: removing co-click edges can't matter when the edges as a whole don't.
- **The unseen-article deficit also comes from training on clicks,** not from message passing: A0 − embedding is −0.013 on unseen articles (all 3 seeds below zero).
- **Implication for Part G:** with GNN vs A0 as the primary comparison and the 5-seed criterion, the expected test outcome is *no measurable graph effect*.

**A9: seen flag as an article input** (seed 0, fixed 9 epochs, vs the reference). The flag was never a model input (the encoder reads only node features), so A9 *adds* it: each article's features get a 0/1 seen-in-the-graph-period flag.

| segment | difference | 95% CI | verdict |
|---|---|---|---|
| test-mix weighted (deciding) | -0.0066 | [-0.0102, -0.0032] | clear effect |
| article=seen | -0.0002 | [-0.0019, +0.0015] | no effect |
| article=unseen | -0.0069 | [-0.0106, -0.0034] | clear effect |

## Round 2: fixed 9 epochs, no early stopping (ADR 009)

Every variant trains for exactly 9 epochs and is compared with the reference at epoch 9 (paired bootstrap, AUC, 1,000 resamples; difference = variant − reference). Borderline deciding results are rerun with seeds 1 and 2 for both variant and reference; *3 seeds* lists the seed 0 / 1 / 2 differences. Win criterion: same sign in all three seeds and seed 0's interval excludes zero.

| id | ablation | segment | difference (seed 0) | 95% CI | verdict | 3 seeds |
|---|---|---|---|---|---|---|
| A1 | no co-click edges | article=seen (deciding) | -0.0052 | [-0.0074, -0.0030] | clear effect | -0.0052 / +0.0113 / +0.0053 |
| A1 | no co-click edges | test-mix weighted | -0.0093 | [-0.0134, -0.0050] | clear effect |  |
| A2 | no entity edges | article=unseen (deciding) | -0.0040 | [-0.0073, -0.0006] | borderline | -0.0040 / +0.0198 / +0.0052 |
| A2 | no entity edges | test-mix weighted | -0.0031 | [-0.0074, +0.0013] | borderline |  |
| A3 | no subcategory nodes | article=unseen (deciding) | -0.0032 | [-0.0063, -0.0001] | borderline | -0.0032 / +0.0156 / -0.0079 |
| A3 | no subcategory nodes | test-mix weighted | -0.0066 | [-0.0107, -0.0025] | clear effect |  |
| A4 | raw co-click edges | test-mix weighted (deciding) | -0.0085 | [-0.0125, -0.0047] | clear effect | -0.0085 / +0.0101 / +0.0072 |
| A5 | no degree down-weighting | test-mix weighted (deciding) | -0.0041 | [-0.0073, -0.0011] | borderline | -0.0041 / +0.0089 / +0.0018 |
| A6 | no graph/example split | test-mix weighted (deciding) | -0.0243 | [-0.0291, -0.0195] | clear effect |  |
| A7 | no training clicks in history | user=known (deciding) | -0.0111 | [-0.0132, -0.0090] | clear effect |  |
| A7 | no training clicks in history | test-mix weighted | -0.0079 | [-0.0120, -0.0041] | clear effect |  |
| A8 | attention pooling | test-mix weighted (deciding) | +0.0017 | [-0.0021, +0.0057] | no effect |  |
| R1 | residual text connection | article=unseen (deciding) | -0.0035 | [-0.0077, +0.0004] | borderline | -0.0035 / +0.0142 / +0.0069 |
| R1 | residual text connection | test-mix weighted (deciding) | -0.0078 | [-0.0133, -0.0024] | borderline | -0.0078 / +0.0110 / +0.0045 |

### A8 at its own length (13 epochs)

A8 (round-1 run, best epoch 13) against the reference trained for a fixed 13 epochs, test-mix weighted AUC: **+0.0053** [+0.0012, +0.0091], borderline.

### Training-seed variance

| model | test-mix weighted | article=unseen | article=seen | all |
|---|---|---|---|---|
| reference seed 0 | 0.6510 | 0.6598 | 0.6674 | 0.6623 |
| reference seed 1 | 0.6351 | 0.6401 | 0.6601 | 0.6478 |
| reference seed 2 | 0.6401 | 0.6526 | 0.6604 | 0.6486 |
| **reference mean** | **0.6421** | **0.6508** | **0.6626** | **0.6529** |
| reference range (max − min) | 0.0159 | 0.0198 | 0.0073 | 0.0145 |
| A1 mean of 3 seeds | 0.6424 | 0.6506 | 0.6665 | 0.6528 |
| A2 mean of 3 seeds | 0.6454 | 0.6579 | 0.6679 | 0.6577 |
| A3 mean of 3 seeds | 0.6439 | 0.6523 | 0.6632 | 0.6557 |
| A4 mean of 3 seeds | 0.6450 | 0.6501 | 0.6695 | 0.6559 |
| A5 mean of 3 seeds | 0.6443 | 0.6570 | 0.6652 | 0.6559 |
| R1 mean of 3 seeds | 0.6447 | 0.6567 | 0.6738 | 0.6621 |

The reference's spread across 3 seeds is **0.0159 AUC on the test-mix weighted row**, larger than most ablation effects. **Seed 0 is the strongest of the three reference runs**, so seed-0 comparisons are made against an unusually good reference and are biased against the variants. The bootstrap interval covers impression sampling only, not training variance, so a seed-0 "clear effect" smaller than about this spread may be partly seed luck. That is what happened to A1 (seen articles) and A4 (test-mix): clear effects at seed 0, but positive at seeds 1 and 2, so within noise over 3 seeds. A3's seed-0 effect on test-mix (−0.007) was not rerun and should be read the same way. A6 (−0.024) and A7 (−0.011 on known users, consistently negative in round 1 too) are well beyond it.

### Decisions (ADR 009 rules)

| id | ablation | decision |
|---|---|---|
| A1 | no co-click edges | within noise: keep current design (no measurable effect) |
| A2 | no entity edges | within noise: keep current design (no measurable effect) |
| A3 | no subcategory nodes | within noise: keep current design (no measurable effect) |
| A4 | raw co-click edges | within noise: keep current design (no measurable effect) |
| A5 | no degree down-weighting | within noise: keep current design (no measurable effect) |
| A6 | no graph/example split | clearly worse: keep current design |
| A7 | no training clicks in history | clearly worse: keep current design |
| A8 | attention pooling | within noise: keep current design (no measurable effect) |
| R1 | residual text connection | keep current design (unseen: noise, test-mix: noise) |

**Notes on the decisions.**
- **A1 (co-click edges) is within noise on seen articles over 3 seeds,** although the GNN beats the embedding baseline on seen articles by +0.030 to +0.037 in every seed. So the seen-article gain does not come from co-click edges on their own; more likely it comes from what training on clicks teaches the model, with any single graph component replaceable. Not tested directly.
- **A8 (attention pooling) is recorded as future work** (ADR 009): within noise at 9 epochs, borderline positive at 13. No further runs now.
- **What is robust:** the graph/example split (A6) and adding training-period clicks to histories (A7).

## Round 1: early stopping (seed 0)

Reference: the current GNN (MLflow run `58c44743ac894c7ea918f1c43d266500`, best epoch 9). Each ablation is trained with early stopping and compared with the reference by paired bootstrap (AUC, 1,000 resamples). **difference = ablation − reference**: negative means the removed or changed component was helping.

Verdict (ADR 009), with w the interval width: **clear effect** = interval excludes 0 by more than w/4; **no effect** = 0 lies in the middle half of the interval; **borderline** = anything else, which gets 3 seeds before a decision.

| id | ablation | segment | best epoch | reference at that epoch vs its best | difference | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| A1 | no co-click edges | article=seen (deciding) | 6 | -0.0086 | -0.0016 | [-0.0042, +0.0007] | borderline |
| A1 | no co-click edges | test-mix weighted | 6 | -0.0086 | -0.0086 | [-0.0131, -0.0039] | clear effect |
| A2 | no entity edges | article=unseen (deciding) | 2 | -0.0189 | +0.0003 | [-0.0034, +0.0041] | no effect |
| A2 | no entity edges | test-mix weighted | 2 | -0.0189 | -0.0100 | [-0.0154, -0.0051] | clear effect |
| A3 | no subcategory nodes | article=unseen (deciding) | 3 | -0.0154 | -0.0161 | [-0.0201, -0.0125] | clear effect |
| A3 | no subcategory nodes | test-mix weighted | 3 | -0.0154 | -0.0093 | [-0.0139, -0.0045] | clear effect |
| A4 | raw co-click edges | test-mix weighted (deciding) | 2 | -0.0189 | -0.0198 | [-0.0251, -0.0145] | clear effect |
| A5 | no degree down-weighting | test-mix weighted (deciding) | 9 | +0.0000 | -0.0029 | [-0.0059, +0.0002] | borderline |
| A6 | no graph/example split | test-mix weighted (deciding) | 2 | -0.0189 | -0.0220 | [-0.0277, -0.0163] | clear effect |
| A7 | no training clicks in history | user=known (deciding) | 3 | -0.0154 | -0.0240 | [-0.0265, -0.0214] | clear effect |
| A7 | no training clicks in history | test-mix weighted | 3 | -0.0154 | -0.0180 | [-0.0231, -0.0131] | clear effect |
| A8 | attention pooling | test-mix weighted (deciding) | 13 | n/a | +0.0046 | [+0.0000, +0.0090] | borderline |
| R1 | residual text connection | article=unseen (deciding) | 3 | -0.0154 | -0.0108 | [-0.0148, -0.0068] | clear effect |
| R1 | residual text connection | test-mix weighted (deciding) | 3 | -0.0154 | -0.0058 | [-0.0107, -0.0006] | borderline |

**Borderline on a deciding segment (to rerun with 3 seeds):** A1 (article=seen), A5 (test-mix weighted), A8 (test-mix weighted), R1 (test-mix weighted)

## Caveat: early stopping confounds several results

Validation AUC is noisy from epoch to epoch, and with patience 2 a single dip ends a run. 7 of 9 ablations stopped with their best epoch at 2, 3, 6 (the reference's best epoch is 9). The column *reference at that epoch vs its best* shows how far below its own final score the reference was at the same epoch (test-mix weighted AUC). Where that gap is as large as the ablation's difference, the comparison mixes *component removed* with *trained for fewer epochs*. Affected: A1 (best epoch 6, reference gap -0.0086), A2 (best epoch 2, reference gap -0.0189), A3 (best epoch 3, reference gap -0.0154), A4 (best epoch 2, reference gap -0.0189), A6 (best epoch 2, reference gap -0.0189), A7 (best epoch 3, reference gap -0.0154), R1 (best epoch 3, reference gap -0.0154).

These verdicts, especially on the test-mix weighted row, should not be used to choose the final configuration until the runs are made comparable in training length.

All segments and nDCG@10 for round 1: `reports/gnn_ablations.csv`.
