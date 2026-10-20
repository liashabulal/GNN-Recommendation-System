"""Score any model on an EvalSplit and return a results table with segment breakdowns.

A model is evaluated purely through its output: a DataFrame with one row per
(impression_id, news_id) candidate in the split's target impressions and a `score`
column (higher = more likely to be clicked).

Segments
    all              every target impression
    user=known/new   impressions whose user did / did not appear in the training period
    article=seen/unseen
                     each impression's candidate list is restricted to articles that were /
                     were not seen in the training period, then ranked among themselves.
                     Impressions left with no click in that sub-list are dropped.
    test-mix weighted
                     only when the split sets `reweight_known_share` (validation): the
                     user=known and user=new scores averaged with weights w and 1 - w, so the
                     user mix matches the test split. This is the primary tuning metric.

All metrics are computed per impression and then averaged. AUC is additionally
undefined when a (sub-)list has no non-clicked article; such impressions are left out
of the AUC average only.

Compare models *within* a segment, not segments with each other: the article
segments use shorter candidate lists, which inflates MRR/nDCG even for random scores.
"""

import numpy as np
import pandas as pd

from recsys.data.mind import explode_impressions
from recsys.data.splits import EvalSplit
from recsys.evaluation.metrics import impression_metrics

METRICS = ["auc", "mrr", "ndcg@5", "ndcg@10"]
KEY = ["impression_id", "news_id"]


def candidate_table(split: EvalSplit) -> pd.DataFrame:
    """One row per target candidate, with click label and segment flags."""
    cands = explode_impressions(split.target)[["impression_id", "user_id", "news_id", "clicked"]]
    cands["known_user"] = cands["user_id"].isin(split.known_users)
    cands["seen_news"] = cands["news_id"].isin(split.seen_news)
    return cands


def attach_scores(cands: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join model scores onto the candidate table, refusing incomplete or extra scores."""
    if not set(KEY + ["score"]) <= set(scores.columns):
        raise ValueError(f"scores must have columns {KEY + ['score']}")
    if scores.duplicated(KEY).any():
        raise ValueError("scores contain duplicate (impression_id, news_id) pairs")

    merged = cands.merge(scores[KEY + ["score"]], on=KEY, how="left", validate="one_to_one")
    n_missing = merged["score"].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing:,} candidates have no score (or a NaN score)")
    if len(scores) != len(cands):
        raise ValueError(f"scores has {len(scores) - len(cands):,} rows that are not target candidates")
    if not np.isfinite(merged["score"].to_numpy(dtype=float)).all():
        raise ValueError("scores must be finite")
    return merged


def per_impression_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Metrics for each impression in `scored` that has at least one click."""
    scored = scored.sort_values("impression_id", kind="stable")
    imp_ids = scored["impression_id"].to_numpy()
    labels = scored["clicked"].to_numpy()
    scores = scored["score"].to_numpy(dtype=float)

    # Slice the sorted arrays per impression instead of a (slow) groupby-apply.
    starts = np.flatnonzero(np.r_[True, imp_ids[1:] != imp_ids[:-1]])
    ends = np.r_[starts[1:], len(imp_ids)]
    rows = []
    for start, end in zip(starts, ends):
        if labels[start:end].sum() == 0:
            continue
        rows.append({"impression_id": imp_ids[start], **impression_metrics(labels[start:end], scores[start:end])})
    return pd.DataFrame(rows, columns=["impression_id", *METRICS])


def _summarise(per_imp: pd.DataFrame, segment: str) -> dict:
    return {"segment": segment, "n_impressions": len(per_imp), **per_imp[METRICS].mean().to_dict()}


def segment_metrics(scores: pd.DataFrame, split: EvalSplit) -> dict[str, pd.DataFrame]:
    """Per-impression metrics for each segment, keyed by segment name."""
    scored = attach_scores(candidate_table(split), scores)

    full = per_impression_metrics(scored)
    known_imps = set(scored.loc[scored["known_user"], "impression_id"])
    is_known = full["impression_id"].isin(known_imps)
    return {
        "all": full,
        "user=known": full[is_known],
        "user=new": full[~is_known],
        "article=seen": per_impression_metrics(scored[scored["seen_news"]]),
        "article=unseen": per_impression_metrics(scored[~scored["seen_news"]]),
    }


def evaluate(scores: pd.DataFrame, split: EvalSplit, model: str) -> pd.DataFrame:
    """Evaluate one model's scores on a split. Returns one row per segment."""
    per_segment = segment_metrics(scores, split)
    rows = [_summarise(per_imp, segment) for segment, per_imp in per_segment.items()]
    known, new = (next(r for r in rows if r["segment"] == s) for s in ("user=known", "user=new"))

    if split.reweight_known_share is not None:
        w = split.reweight_known_share
        rows.append({
            "segment": "test-mix weighted",
            "n_impressions": len(per_segment["all"]),
            **{m: w * known[m] + (1 - w) * new[m] for m in METRICS},
        })
    results = pd.DataFrame(rows)
    results.insert(0, "split", split.name)
    results.insert(0, "model", model)
    return results
