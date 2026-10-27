"""Command-line dispatch of recsys.graph.periods (no data needed: builders are replaced)."""

import pytest

from recsys.graph import periods


@pytest.fixture
def calls(monkeypatch):
    seen = {"raw": [], "periods": []}
    monkeypatch.setattr(periods, "build_raw_coclick", lambda name: seen["raw"].append(name))
    monkeypatch.setattr(periods, "build_periods", lambda names: seen["periods"].append(list(names)))
    return seen


def test_raw_coclick_defaults_to_validation_periods(calls):
    periods.main(["raw-coclick"])
    assert calls["raw"] == ["validation_train", "validation_score"]
    assert calls["periods"] == []


def test_raw_coclick_named_periods(calls):
    periods.main(["raw-coclick", "test_train"])
    assert calls["raw"] == ["test_train"]


def test_raw_coclick_rejects_unknown_period(calls):
    with pytest.raises(SystemExit, match="unknown period"):
        periods.main(["raw-coclick", "nonsense"])
    assert calls["raw"] == []


def test_plain_names_build_periods(calls):
    periods.main(["validation_score"])
    assert calls["periods"] == [["validation_score"]] and calls["raw"] == []
