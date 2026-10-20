"""Offline tests for Wikidata parsing and the coarse type mapping (no network)."""

import pandas as pd

from recsys.data.wikidata import _parse_entities, coarse_type
from recsys.graph.build import entity_type_tables

URI = "http://www.wikidata.org/entity/"


def test_parse_entities_keeps_every_requested_id():
    bindings = [
        {"item": {"value": URI + "Q1"}, "itemLabel": {"value": "One"}, "type": {"value": URI + "Q5"}},
        {"item": {"value": URI + "Q1"}, "itemLabel": {"value": "One"}, "type": {"value": URI + "Q215627"}},
        # The label service returns the bare QID when there is no English label.
        {"item": {"value": URI + "Q2"}, "itemLabel": {"value": "Q2"}},
    ]
    rows = {r["id"]: r for r in _parse_entities(["Q1", "Q2", "Q3"], bindings)}
    assert rows["Q1"] == {"id": "Q1", "label": "One", "p31": ["Q5", "Q215627"]}
    assert rows["Q2"]["label"] is None and rows["Q2"]["p31"] == []
    assert rows["Q3"] == {"id": "Q3", "label": None, "p31": []}  # nothing returned


def test_coarse_type_priority():
    type_roots = {
        "Q5": ["Q5"],                    # human
        "Q3624078": ["Q56061", "Q43229", "Q2221906"],  # sovereign state: territorial + organisation
        "Q4830453": ["Q43229"],          # business
        "Q41176": ["Q41176"],            # building
        "Q11424": ["Q17537576"],         # film
        "Q999": [],                      # unrelated class
    }
    assert coarse_type(["Q5"], type_roots) == "person"
    assert coarse_type(["Q3624078"], type_roots) == "place"  # territorial place outranks organisation
    assert coarse_type(["Q4830453", "Q41176"], type_roots) == "organisation"  # organisation outranks building
    assert coarse_type(["Q41176"], type_roots) == "place"
    assert coarse_type(["Q4830453"], type_roots) == "organisation"
    assert coarse_type(["Q11424", "Q4830453"], type_roots) == "organisation"  # organisation outranks creative work
    assert coarse_type(["Q999"], type_roots) == "other"
    assert coarse_type([], type_roots) == "other"


def test_entity_type_tables_default_to_other():
    entities = pd.DataFrame({"entity_id": ["Q1", "Q2"], "label": ["a", "b"], "mind_type": ["P", "G"]})
    types = pd.DataFrame({"entity_id": ["Q1"], "wikidata_label": ["A"], "coarse_type": ["person"]})
    tables = entity_type_tables(entities, types)
    edges = dict(tables["edges_entity_is_type"].itertuples(index=False))
    assert edges == {"Q1": "person", "Q2": "other"}
    assert len(tables["nodes_entity_type"]) == 6
