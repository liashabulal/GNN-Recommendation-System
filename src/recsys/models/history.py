"""The article history a user encoder sees for each impression.

For an impression of user u at time t, the history is:
    u's history column (a snapshot taken before the log window, oldest -> newest)
    + u's clicks in training-period impressions made strictly before t (oldest -> newest)
Each article is kept once, at its most recent position, and only the last `max_history`
items are used: the most recent ones (see reports/milestone3_gaps.md for the order).

Target-day labels are never used: only `train` (training-period) clicks are appended.
"""

from bisect import bisect_left

import numpy as np
import pandas as pd
import torch

from recsys.data.mind import explode_impressions


def impression_histories(behaviors: pd.DataFrame, train: pd.DataFrame, max_history: int) -> pd.Series:
    """Most recent history (list of news IDs) per impression_id of `behaviors`."""
    cands = explode_impressions(train)
    clicks = cands[cands["clicked"] == 1].sort_values("time", kind="stable")
    by_user = {u: (g["time"].tolist(), g["news_id"].tolist()) for u, g in clicks.groupby("user_id")}

    out = {}
    for imp, user, time, history in behaviors[["impression_id", "user_id", "time", "history"]].itertuples(index=False):
        times, news = by_user.get(user, ([], []))
        sequence = list(history) + news[: bisect_left(times, time)]  # clicks strictly before t
        # Keep each article once, at its most recent position.
        latest = list(dict.fromkeys(reversed(sequence)))[::-1]
        out[imp] = latest[-max_history:]
    return pd.Series(out, name="history")


def history_tensor(histories: pd.Series, article_row: pd.Series, max_history: int) -> torch.Tensor:
    """[n_impressions, max_history] article row indices, padded with -1 (order = histories.index)."""
    out = np.full((len(histories), max_history), -1, dtype=np.int64)
    for i, items in enumerate(histories):
        rows = article_row.reindex(items).dropna().astype(int).to_numpy()
        out[i, : len(rows)] = rows
    return torch.from_numpy(out)
