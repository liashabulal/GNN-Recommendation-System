"""Time-based evaluation splits for MIND-small.

    split       fit on (training period)          scored on
    validation  MIND train impressions, 9-13 Nov  MIND train impressions, 14 Nov  (tuning only)
    test        MIND train impressions, 9-14 Nov  MIND dev impressions,   15 Nov  (final reporting only)

Protocol: tune hyperparameters on validation, freeze them, then refit on 9-14 Nov for
the single test run. "Known users" and "seen articles" are always defined against the
split's own training period.

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
VALIDATION_DAY = pd.Timestamp("2019-11-14")  # validation's training period ends just before this
TEST_DAY = pd.Timestamp("2019-11-15")  # test's training period ends just before this

SPLIT_NAMES = ("validation", "test")


@dataclass(frozen=True)
class EvalSplit:
    """One evaluation setting: what a model may learn from, and what it is scored on."""

    name: str
    train: pd.DataFrame  # training-period behaviors (impressions with labels)
    target: pd.DataFrame  # behaviors to score
    # If set, the harness adds a row re-weighting the known/new-user segments so that
    # known users make up this share of impressions (used to mimic test on validation).
    reweight_known_share: float | None = None

    @property
    def known_users(self) -> set[str]:
        """Users with at least one impression in the training period."""
        return set(self.train["user_id"])

    @property
    def seen_news(self) -> set[str]:
        """Articles that appear in a training-period history or candidate list."""
        return set(self.train["history"].explode().dropna()) | set(self.train["candidates"].explode())


def known_user_share(train: pd.DataFrame, target: pd.DataFrame) -> float:
    """Share of target impressions whose user has an impression in `train`."""
    return float(target["user_id"].isin(set(train["user_id"])).mean())


def load_split(name: str) -> EvalSplit:
    """Load the "validation" or "test" split (see module docstring for the periods)."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {name!r} (expected one of {SPLIT_NAMES})")

    mind_train = load_behaviors("train")
    mind_dev = load_behaviors("dev")

    # Test: refit on the whole of MIND train (9-14 Nov), score MIND dev.
    test_train = mind_train[mind_train["time"] < TEST_DAY].reset_index(drop=True)
    if name == "test":
        split = EvalSplit(name="test", train=test_train, target=mind_dev)
    else:
        split = EvalSplit(
            name="validation",
            train=mind_train[mind_train["time"] < VALIDATION_DAY].reset_index(drop=True),
            target=mind_train[mind_train["time"] >= VALIDATION_DAY].reset_index(drop=True),
            # Only user IDs are read from the test split here, never its labels.
            reweight_known_share=known_user_share(test_train, mind_dev),
        )

    # Guard against accidental overlap if the boundaries above are ever edited.
    assert split.train["time"].max() < split.target["time"].min(), "training period overlaps target"
    return split
