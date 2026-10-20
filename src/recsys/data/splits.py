"""Time-based evaluation splits for MIND-small.

    training period : MIND train impressions, 9-13 Nov 2019   (fit models / features here)
    validation      : MIND train impressions, 14 Nov 2019     (tuning only)
    test            : MIND dev impressions,   15 Nov 2019     (final reporting only)

Anything a model learns from interactions (click counts, user/article IDs it has
seen, co-click edges, ...) must come from `EvalSplit.train` only. The `target`
impressions may be used at scoring time exactly as a live system would see them:
the user's history snapshot and the list of candidates, but never the click labels.

Each user's MIND history is a fixed snapshot taken before the log window (it is
identical across all of a user's impressions in a split), so it never contains
clicks from the validation or test day.
"""

from dataclasses import dataclass

import pandas as pd

from recsys.data.mind import load_behaviors

TRAINING_START = pd.Timestamp("2019-11-09")
VALIDATION_DAY = pd.Timestamp("2019-11-14")  # training period ends just before this
TEST_DAY = pd.Timestamp("2019-11-15")

SPLIT_NAMES = ("validation", "test")


@dataclass(frozen=True)
class EvalSplit:
    """One evaluation setting: what a model may learn from, and what it is scored on."""

    name: str
    train: pd.DataFrame  # training-period behaviors (impressions with labels)
    target: pd.DataFrame  # behaviors to score

    @property
    def known_users(self) -> set[str]:
        """Users with at least one impression in the training period."""
        return set(self.train["user_id"])

    @property
    def seen_news(self) -> set[str]:
        """Articles that appear in a training-period history or candidate list."""
        return set(self.train["history"].explode().dropna()) | set(self.train["candidates"].explode())


def _training_period(mind_train: pd.DataFrame) -> pd.DataFrame:
    return mind_train[mind_train["time"] < VALIDATION_DAY].reset_index(drop=True)


def load_split(name: str) -> EvalSplit:
    """Load the "validation" or "test" split. Both use the same 9-13 Nov training period."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {name!r} (expected one of {SPLIT_NAMES})")

    mind_train = load_behaviors("train")
    train = _training_period(mind_train)

    if name == "validation":
        target = mind_train[mind_train["time"] >= VALIDATION_DAY].reset_index(drop=True)
    else:
        target = load_behaviors("dev")

    # Guard against accidental overlap if the boundaries above are ever edited.
    assert train["time"].max() < target["time"].min(), "training period overlaps target"
    return EvalSplit(name=name, train=train, target=target)
