"""Per-impression ranking metrics used by MIND: AUC, MRR, nDCG@k.

Definitions follow the official MIND scorer (binary click labels):
    AUC     - probability a random clicked article outscores a random non-clicked one
    MRR     - mean of 1/rank over *all* clicked articles (MIND's variant, not first-click only)
    nDCG@k  - DCG over the top k (gain 2^label - 1, discount log2(rank + 1)), divided by the ideal DCG

Ties: the official scorer breaks ties by array position, which lets candidate order leak
into the result and matters for models that emit many equal scores (e.g. popularity,
where every unseen article scores 0). Here every metric is the exact *expected* value
under a uniformly random ordering of tied candidates. Without ties, the results are
identical to the official scorer.
"""

import numpy as np


def _rank_ties(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank candidates by descending score, averaging labels within tie groups.

    Returns, for ranks 1..n:
        expected_label - probability the article at that rank is a click
        mean_rank      - average rank of the tie group occupying that rank
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary (0/1)")

    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]

    # A new tie group starts wherever the score changes.
    group_starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    group_sizes = np.diff(np.r_[group_starts, len(scores)])
    clicks_per_group = np.add.reduceat(sorted_labels, group_starts)

    expected_label = np.repeat(clicks_per_group / group_sizes, group_sizes)
    mean_rank = np.repeat(group_starts + (group_sizes + 1) / 2, group_sizes)
    return expected_label, mean_rank


def _check_has_click(labels) -> int:
    n_clicks = int(np.sum(labels))
    if n_clicks == 0:
        raise ValueError("metric undefined: impression has no clicked article")
    return n_clicks


def auc(labels, scores) -> float:
    """Area under the ROC curve; NaN when the impression has no non-clicked article."""
    n_pos = _check_has_click(labels)
    n_neg = len(labels) - n_pos
    if n_neg == 0:
        return float("nan")
    expected_label, mean_rank = _rank_ties(labels, scores)
    # Mann-Whitney U with average ranks; ascending rank = n + 1 - descending rank.
    ascending_rank = len(labels) + 1 - mean_rank
    rank_sum_pos = np.sum(expected_label * ascending_rank)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mrr(labels, scores) -> float:
    """MIND's MRR: sum of 1/rank over clicked articles, divided by the number of clicks."""
    n_pos = _check_has_click(labels)
    expected_label, _ = _rank_ties(labels, scores)
    ranks = np.arange(1, len(labels) + 1)
    return float(np.sum(expected_label / ranks) / n_pos)


def ndcg(labels, scores, k: int) -> float:
    """Normalised discounted cumulative gain over the top k ranks."""
    n_pos = _check_has_click(labels)
    expected_label, _ = _rank_ties(labels, scores)
    discounts = 1 / np.log2(np.arange(2, len(labels) + 2))  # rank r -> 1 / log2(r + 1)
    dcg = np.sum(expected_label[:k] * discounts[:k])  # binary labels: gain 2^y - 1 = y
    ideal_dcg = np.sum(discounts[: min(n_pos, k)])
    return float(dcg / ideal_dcg)


def impression_metrics(labels, scores) -> dict[str, float]:
    """All four MIND metrics for one impression."""
    return {
        "auc": auc(labels, scores),
        "mrr": mrr(labels, scores),
        "ndcg@5": ndcg(labels, scores, k=5),
        "ndcg@10": ndcg(labels, scores, k=10),
    }
