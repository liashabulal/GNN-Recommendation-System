"""Loaders for the raw MIND (Microsoft News Dataset) TSV files.

File formats (no header row, tab-separated):

behaviors.tsv - one row per impression (a page view showing a list of articles)
    impression_id, user_id, time, history, impressions
    - history:     space-separated news IDs the user clicked *before* this split
    - impressions: space-separated "NewsID-label" pairs, label 1 = clicked, 0 = not

news.tsv - one row per article
    news_id, category, subcategory, title, abstract, url,
    title_entities, abstract_entities
    - *_entities: JSON lists of linked Wikidata entities, each with a "WikidataId"
"""

import json

import pandas as pd

from recsys.paths import mind_split_dir

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def _split_ids(value: str) -> list[str]:
    """Split a space-separated ID string; missing or empty values become []."""
    return value.split() if isinstance(value, str) else []


def _wikidata_ids(entities_json: str) -> list[str]:
    """Extract Wikidata IDs from one *_entities JSON cell."""
    if not isinstance(entities_json, str) or not entities_json:
        return []
    return [entity["WikidataId"] for entity in json.loads(entities_json)]


def load_behaviors(split: str) -> pd.DataFrame:
    """Load behaviors.tsv for a split, with parsed time and list columns.

    Adds:
        history    -> list[str] of previously clicked news IDs
        candidates -> list[str] of news IDs shown in the impression
        labels     -> list[int] of click labels aligned with `candidates`
    """
    df = pd.read_csv(
        mind_split_dir(split) / "behaviors.tsv",
        sep="\t",
        header=None,
        names=BEHAVIOR_COLUMNS,
        dtype={"impression_id": "int64", "user_id": "string"},
    )
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].map(_split_ids)

    # "N123-1 N456-0" -> candidates ["N123", "N456"], labels [1, 0]
    pairs = df.pop("impressions").map(lambda s: [p.rsplit("-", 1) for p in s.split()])
    df["candidates"] = pairs.map(lambda ps: [news_id for news_id, _ in ps])
    df["labels"] = pairs.map(lambda ps: [int(label) for _, label in ps])
    return df


def load_news(split: str) -> pd.DataFrame:
    """Load news.tsv for a split, with entity JSON reduced to lists of Wikidata IDs."""
    df = pd.read_csv(
        mind_split_dir(split) / "news.tsv",
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        quoting=3,  # csv.QUOTE_NONE: titles contain bare quote characters
        dtype="string",
    )
    df["title_entity_ids"] = df.pop("title_entities").map(_wikidata_ids)
    df["abstract_entity_ids"] = df.pop("abstract_entities").map(_wikidata_ids)
    return df


def explode_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """One row per (impression, candidate article) with its click label."""
    return (
        behaviors[["impression_id", "user_id", "time", "candidates", "labels"]]
        .explode(["candidates", "labels"])
        .rename(columns={"candidates": "news_id", "labels": "clicked"})
        .astype({"clicked": "int8"})
        .reset_index(drop=True)
    )
