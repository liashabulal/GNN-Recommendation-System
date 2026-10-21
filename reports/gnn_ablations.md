# GNN ablations (validation, 14 Nov, seed 0)

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

All segments and nDCG@10: `reports/gnn_ablations.csv`.
