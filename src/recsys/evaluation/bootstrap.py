"""Paired bootstrap confidence intervals for the difference between two models.

Both models are scored on the same impressions, so we work with the per-impression
difference d_i = metric_A(i) - metric_B(i). Each resample draws impressions with
replacement and takes the mean of d over the draw; the 2.5th and 97.5th percentiles
of those means give the 95% interval. Pairing removes the impression-to-impression
variation both models share (easy vs hard impressions), so intervals are much tighter
than comparing two independent CIs.

Per segment, impressions are resampled within that segment. The test-mix weighted
row resamples known-user and new-user impressions separately (stratified) and
combines the two resampled means with the split's weight.
"""

import numpy as np
import pandas as pd

from recsys.data.splits import EvalSplit
from recsys.evaluation.harness import segment_metrics


def resampled_means(diffs: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Mean of `diffs` over `n_resamples` bootstrap draws (with replacement); NaN if empty."""
    n = len(diffs)
    if n == 0:
        return np.full(n_resamples, np.nan)
    rng = np.random.default_rng(seed)
    return np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])


def _paired_diffs(per_imp_a: pd.DataFrame, per_imp_b: pd.DataFrame, metric: str) -> np.ndarray:
    a = per_imp_a.set_index("impression_id")[metric].sort_index()
    b = per_imp_b.set_index("impression_id")[metric].sort_index()
    if not a.index.equals(b.index):
        raise ValueError("models were not evaluated on the same impressions")
    # AUC is NaN where a (sub-)list has no non-click; that depends only on labels,
    # so the same impressions are dropped for both models.
    return (a - b).dropna().to_numpy()


def paired_bootstrap(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    split: EvalSplit,
    model_a: str = "a",
    model_b: str = "b",
    metrics: tuple[str, ...] = ("auc", "ndcg@10"),
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Difference (A - B) per segment and metric, with a (1 - alpha) bootstrap interval."""
    seg_a, seg_b = segment_metrics(scores_a, split), segment_metrics(scores_b, split)
    quantiles = [alpha / 2, 1 - alpha / 2]

    rows = []
    for metric in metrics:
        means_by_segment = {}
        for segment in seg_a:
            diffs = _paired_diffs(seg_a[segment], seg_b[segment], metric)
            means_by_segment[segment] = resampled_means(diffs, n_resamples, seed)
            rows.append({"segment": segment, "metric": metric, "n_impressions": len(diffs),
                         "diff": diffs.mean() if len(diffs) else np.nan,
                         "_means": means_by_segment[segment]})

        if split.reweight_known_share is not None:
            w = split.reweight_known_share
            known, new = (next(r for r in rows if r["segment"] == s and r["metric"] == metric)
                          for s in ("user=known", "user=new"))
            rows.append({
                "segment": "test-mix weighted", "metric": metric,
                "n_impressions": known["n_impressions"] + new["n_impressions"],
                "diff": w * known["diff"] + (1 - w) * new["diff"],
                "_means": w * means_by_segment["user=known"] + (1 - w) * means_by_segment["user=new"],
            })

    for row in rows:
        row["ci_low"], row["ci_high"] = np.quantile(row.pop("_means"), quantiles)
        row["significant"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    out = pd.DataFrame(rows)
    out.insert(0, "model_b", model_b)
    out.insert(0, "model_a", model_a)
    out.insert(0, "split", split.name)
    return out
