"""Enrich MIND entities with Wikidata labels and coarse types via SPARQL.

Run with `uv run python -m recsys.data.wikidata`. Two cached passes:

1. Entities: English label and direct "instance of" (P31) types for every entity ID.
2. Types: each distinct P31 type is checked against a few root classes via
   "subclass of" (P279*). Classifying ~thousands of types is much cheaper than
   walking subclass chains for ~30k entities.

Raw query results are cached as JSON lines in data/interim/wikidata/, one line per
entity or type. Re-running only queries IDs not yet in the cache, so an interrupted
run resumes and a finished run never hits the endpoint again.

Coarse type mapping: each P31 type is checked for being a subclass (P279*) of these
root classes, and an entity takes the first rule, in this order, that any of its types meets:

    1. person        Q5 human
    2. place         Q56061 administrative territorial entity     (countries, states, cities)
    3. organisation  Q43229 organization
    4. place         Q2221906 geographic location, Q618123 geographical feature, Q41176 building
    5. event         Q1656682 event, Q1190554 occurrence
    6. creative work Q17537576 creative work
    7. other         anything else, including entities with no P31

Why two place rules: Wikidata classes sovereign states as organisations too, so territorial
places must outrank organisation (rule 2). But many companies and universities also carry
a type that reaches "building" or "geographical feature" (a chain's outlets, a campus), so
generic place evidence must rank *below* organisation (rule 4). The broader root
"work" (Q386724) is deliberately not used: it also covers buildings and stadiums.
"""

import json
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.paths import CONFIGS_DIR, INTERIM_DIR

CACHE_DIR = INTERIM_DIR / "wikidata"
ENTITY_CACHE = CACHE_DIR / "entities.jsonl"
TYPE_CACHE = CACHE_DIR / "types.jsonl"
ENTITY_TYPES_FILE = CACHE_DIR / "entity_types.parquet"

COARSE_RULES = [  # (coarse type, root classes), in priority order
    ("person", ["Q5"]),
    ("place", ["Q56061"]),
    ("organisation", ["Q43229"]),
    ("place", ["Q2221906", "Q618123", "Q41176"]),
    ("event", ["Q1656682", "Q1190554"]),
    ("creative work", ["Q17537576"]),
]
# The type cache was queried with these roots (incl. Q386724 "work", now unused).
QUERY_ROOTS = sorted({q for _, qs in COARSE_RULES for q in qs} | {"Q386724"})
COARSE_TYPES = ["person", "place", "organisation", "event", "creative work", "other"]


def load_config() -> dict:
    with open(CONFIGS_DIR / "wikidata.toml", "rb") as f:
        return tomllib.load(f)


def sparql(query: str, cfg: dict) -> list[dict]:
    """POST a SPARQL query; retry on rate limits and server errors."""
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/sparql-results+json"}
    for attempt in range(cfg["max_retries"] + 1):
        try:
            req = urllib.request.Request(cfg["endpoint"], data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=cfg["timeout_seconds"]) as resp:
                return json.load(resp)["results"]["bindings"]
        except urllib.error.HTTPError as err:
            if err.code != 429 and err.code < 500 or attempt == cfg["max_retries"]:
                raise
            wait = float(err.headers.get("Retry-After") or 2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == cfg["max_retries"]:
                raise
            wait = 2 ** (attempt + 1)
        print(f"    retry {attempt + 1} in {wait:.0f}s")
        time.sleep(wait)


def _qid(binding: dict, key: str) -> str | None:
    value = binding.get(key, {}).get("value")
    return value.rsplit("/", 1)[-1] if value else None


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {row["id"]: row for row in map(json.loads, f)}


def _fetch_batches(ids: list[str], batch_size: int, cache: Path, make_query, parse, cfg: dict, what: str):
    """Query uncached IDs in batches, appending each batch's results to the cache."""
    done = _read_cache(cache)
    todo = [i for i in ids if i not in done]
    print(f"{what}: {len(ids) - len(todo):,} cached, {len(todo):,} to query")
    cache.parent.mkdir(parents=True, exist_ok=True)
    for n, start in enumerate(range(0, len(todo), batch_size), 1):
        batch = todo[start : start + batch_size]
        rows = parse(batch, sparql(make_query(batch), cfg))
        with open(cache, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        if n % 20 == 0:
            print(f"  {start + len(batch):,} / {len(todo):,}")
        time.sleep(cfg["pause_seconds"])
    return _read_cache(cache)


def _entity_query(batch: list[str]) -> str:
    values = " ".join(f"wd:{q}" for q in batch)
    return f"""
SELECT ?item ?itemLabel ?type WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P31 ?type . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def _parse_entities(batch: list[str], bindings: list[dict]) -> list[dict]:
    # Every requested ID gets a row, even if Wikidata returned nothing for it.
    rows = {q: {"id": q, "label": None, "p31": []} for q in batch}
    for b in bindings:
        row = rows[_qid(b, "item")]
        label = b.get("itemLabel", {}).get("value")
        if label and label != row["id"]:  # the label service falls back to the bare QID
            row["label"] = label
        if (t := _qid(b, "type")) and t not in row["p31"]:
            row["p31"].append(t)
    return list(rows.values())


def _type_query(batch: list[str]) -> str:
    values = " ".join(f"wd:{q}" for q in batch)
    roots = " ".join(f"wd:{q}" for q in QUERY_ROOTS)
    return f"""
SELECT ?type ?typeLabel ?root WHERE {{
  VALUES ?type {{ {values} }}
  OPTIONAL {{ VALUES ?root {{ {roots} }} ?type wdt:P279* ?root . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def _parse_types(batch: list[str], bindings: list[dict]) -> list[dict]:
    rows = {q: {"id": q, "label": None, "roots": []} for q in batch}
    for b in bindings:
        row = rows[_qid(b, "type")]
        row["label"] = b.get("typeLabel", {}).get("value")
        if (r := _qid(b, "root")) and r not in row["roots"]:
            row["roots"].append(r)
    return list(rows.values())


def coarse_type(p31: list[str], type_roots: dict[str, list[str]]) -> str:
    """Highest-priority coarse type reachable from any of an entity's P31 types."""
    roots = {r for t in p31 for r in type_roots.get(t, [])}
    for coarse, qids in COARSE_RULES:
        if roots & set(qids):
            return coarse
    return "other"


def enrich(entity_ids: list[str]) -> pd.DataFrame:
    """Fetch (or read from cache) labels and types; return one row per entity."""
    cfg = load_config()
    entities = _fetch_batches(sorted(set(entity_ids)), cfg["entity_batch_size"], ENTITY_CACHE,
                              _entity_query, _parse_entities, cfg, "entities")
    type_ids = sorted({t for e in entities.values() for t in e["p31"]})
    types = _fetch_batches(type_ids, cfg["type_batch_size"], TYPE_CACHE, _type_query, _parse_types, cfg, "types")
    type_roots = {t: row["roots"] for t, row in types.items()}

    out = pd.DataFrame([
        {
            "entity_id": q,
            "wikidata_label": entities[q]["label"],
            "p31": entities[q]["p31"],
            "p31_labels": [types[t]["label"] for t in entities[q]["p31"]],
            "coarse_type": coarse_type(entities[q]["p31"], type_roots),
        }
        for q in sorted(set(entity_ids))
    ])
    out.to_parquet(ENTITY_TYPES_FILE, index=False)
    return out


# MIND's one-letter codes are undocumented; only these three have an unambiguous reading.
MIND_CODE_EXPECTED = {"P": "person", "G": "place", "O": "organisation"}


def _md_table(df: pd.DataFrame) -> str:
    lines = ["| | " + " | ".join(map(str, df.columns)) + " |", "|---" * (len(df.columns) + 1) + "|"]
    lines += [f"| {i} | " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row) + " |"
              for i, row in df.iterrows()]
    return "\n".join(lines)


def compare_with_mind(typed: pd.DataFrame) -> str:
    """Markdown report: coarse Wikidata types vs MIND's own entity type codes."""
    counts = typed["coarse_type"].value_counts().reindex(COARSE_TYPES).to_frame("entities")
    cross = pd.crosstab(typed["mind_type"], typed["coarse_type"]).reindex(columns=COARSE_TYPES, fill_value=0)
    cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).index]

    agree_rows, examples = [], []
    for code, expected in MIND_CODE_EXPECTED.items():
        sub = typed[typed["mind_type"] == code]
        agree_rows.append({"MIND code": code, "expected": expected, "entities": len(sub),
                           "agreement": f"{(sub['coarse_type'] == expected).mean():.1%}"})
        wrong = sub[sub["coarse_type"] != expected]
        examples.append(f"- **{code} → not {expected}** ({len(wrong):,}): " + "; ".join(
            f"{r.wikidata_label if pd.notna(r.wikidata_label) else r.entity_id} ({r.coarse_type})" for r in wrong.head(6).itertuples()))
    agree = pd.DataFrame(agree_rows).set_index("MIND code")
    p, g, o = (typed["mind_type"] == c for c in "PGO")
    pgo = typed[p | g | o]
    overall = (pgo["coarse_type"] == pgo["mind_type"].map(MIND_CODE_EXPECTED)).mean()

    return "\n\n".join([
        "# Wikidata entity types\n\nGenerated by `uv run python -m recsys.data.wikidata` "
        "(reads the cache in data/interim/wikidata/; mapping documented in the module docstring).",
        f"Entities: {len(typed):,} · without an English label: {typed['wikidata_label'].isna().sum():,} · "
        f"without any P31 type: {(typed['p31'].map(len) == 0).sum():,}",
        "## Coarse type counts\n\n" + _md_table(counts),
        "## Agreement with MIND's type codes\n\nMIND's codes are undocumented. P, G and O read "
        f"unambiguously as person, location and organisation; overall agreement on those: **{overall:.1%}**.\n\n"
        + _md_table(agree),
        "Examples of disagreement:\n\n" + "\n".join(examples),
        "## Most common raw P31 types among entities mapped to \"other\"\n\n"
        "Counts entities (an entity with several P31 types counts once per type). Reviewed for obvious "
        "mapping gaps: none is a large group that clearly belongs to an existing coarse type (see ADR 005).\n\n"
        + _md_table(typed.loc[typed["coarse_type"] == "other", "p31_labels"].explode().value_counts()
                    .head(20).rename_axis("P31 type").to_frame("entities")),
        "## All MIND codes vs coarse types (entity counts)\n\n"
        "Other codes appear to be finer classes (e.g. W = works, F = facilities, S = structures/streets, "
        "L = locations, U = miscellaneous), inferred from the table below and sample labels.\n\n" + _md_table(cross),
    ]) + "\n"


def main() -> None:
    from recsys.graph.build import load_graph
    from recsys.paths import REPORTS_DIR

    # The test setup's entities are a superset of the validation setup's.
    entities = load_graph("test")["nodes_entity"][["entity_id", "mind_type"]]
    typed = entities.merge(enrich(entities["entity_id"].tolist()), on="entity_id")
    report = compare_with_mind(typed)
    (REPORTS_DIR / "wikidata_types.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
