"""Central place for project paths, so notebooks and scripts never hard-code them."""

from pathlib import Path

# src/recsys/paths.py -> project root is two levels above the package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MIND-small: one folder per split, each with behaviors.tsv, news.tsv and the
# entity/relation embedding files.
MIND_RAW_DIR = RAW_DIR / "mind"


def mind_split_dir(split: str) -> Path:
    """Return the raw folder for a MIND split ("train" or "dev")."""
    if split not in {"train", "dev"}:
        raise ValueError(f"Unknown MIND split: {split!r}")
    return MIND_RAW_DIR / split
