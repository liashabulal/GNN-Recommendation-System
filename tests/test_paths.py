import pytest

from recsys import paths


def test_project_root_contains_pyproject():
    assert (paths.PROJECT_ROOT / "pyproject.toml").exists()


def test_mind_split_dir():
    assert paths.mind_split_dir("train") == paths.MIND_RAW_DIR / "train"
    with pytest.raises(ValueError):
        paths.mind_split_dir("test")
