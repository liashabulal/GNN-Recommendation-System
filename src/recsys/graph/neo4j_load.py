"""Load a graph setup into Neo4j. Run with `uv run python -m recsys.graph.neo4j_load [setup]`.

Requires the Neo4j container (`colima start --memory 4 && docker compose up -d`).
Connection settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, defaulting to the
local docker-compose values. Loading wipes the database first, so it is repeatable.

Property graph schema (test setup by default):

    (:Article {id, title, category, seen_in_training})
    (:Subcategory {id, name, category})   (:Category {id})
    (:Entity {id, label, wikidata_label, mind_type})   (:EntityType {id})

    (:Article)-[:IN_SUBCATEGORY]->(:Subcategory)-[:IN_CATEGORY]->(:Category)
    (:Article)-[:MENTIONS {source, confidence}]->(:Entity)-[:IS_TYPE]->(:EntityType)
    (:Article)-[:COCLICKED {weight, norm_weight}]->(:Article)

COCLICKED is undirected in meaning; it is stored once (src -> dst) and should be matched
without a direction: (a)-[:COCLICKED]-(b). Only the *pruned* co-click edges are loaded
(ADR 004); the raw 16M-edge table stays in parquet.
"""

import os
import sys
import time

import pandas as pd
from neo4j import GraphDatabase

from recsys.graph.build import load_graph

BATCH_SIZE = 10_000

NODE_LABELS = ["Article", "Subcategory", "Category", "Entity", "EntityType"]


def connect():
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "recsys-local-dev")),
    )


def _records(df: pd.DataFrame) -> list[dict]:
    # Parquet/pandas missing values become None so Neo4j stores no property.
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _write(session, query: str, df: pd.DataFrame, what: str) -> None:
    rows = _records(df)
    start = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[i : i + BATCH_SIZE]).consume()
    print(f"  {what:<28} {len(rows):>9,}  {time.time() - start:5.1f}s")


def reset(session) -> None:
    """Delete everything, in batches so large graphs don't exhaust memory."""
    session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
    for label in NODE_LABELS:
        session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()


def load(setup: str = "test") -> None:
    g = load_graph(setup)
    with connect() as driver, driver.session() as s:
        print(f"Loading the {setup} setup into Neo4j")
        reset(s)

        # Nodes
        _write(s, "UNWIND $rows AS r MERGE (n:Category {id: r.category})", g["nodes_category"], "Category")
        _write(s, """UNWIND $rows AS r MERGE (n:Subcategory {id: r.subcategory_id})
                     SET n.name = r.name, n.category = r.category""", g["nodes_subcategory"], "Subcategory")
        _write(s, """UNWIND $rows AS r MERGE (n:Article {id: r.news_id})
                     SET n.title = r.title, n.category = r.category, n.seen_in_training = r.seen_in_training""",
               g["nodes_article"], "Article")
        _write(s, """UNWIND $rows AS r MERGE (n:Entity {id: r.entity_id})
                     SET n.label = r.label, n.wikidata_label = r.wikidata_label, n.mind_type = r.mind_type""",
               g["nodes_entity"], "Entity")
        _write(s, "UNWIND $rows AS r MERGE (n:EntityType {id: r.entity_type})", g["nodes_entity_type"], "EntityType")

        # Edges (MATCH on the constrained id is an index lookup)
        _write(s, """UNWIND $rows AS r MATCH (a:Article {id: r.news_id}), (b:Subcategory {id: r.subcategory_id})
                     CREATE (a)-[:IN_SUBCATEGORY]->(b)""", g["edges_article_in_subcategory"], "IN_SUBCATEGORY")
        _write(s, """UNWIND $rows AS r MATCH (a:Subcategory {id: r.subcategory_id}), (b:Category {id: r.category})
                     CREATE (a)-[:IN_CATEGORY]->(b)""", g["edges_subcategory_in_category"], "IN_CATEGORY")
        _write(s, """UNWIND $rows AS r MATCH (a:Article {id: r.news_id}), (e:Entity {id: r.entity_id})
                     CREATE (a)-[:MENTIONS {source: r.source, confidence: r.confidence}]->(e)""",
               g["edges_article_mentions_entity"], "MENTIONS")
        _write(s, """UNWIND $rows AS r MATCH (e:Entity {id: r.entity_id}), (t:EntityType {id: r.entity_type})
                     CREATE (e)-[:IS_TYPE]->(t)""", g["edges_entity_is_type"], "IS_TYPE")
        _write(s, """UNWIND $rows AS r MATCH (a:Article {id: r.src}), (b:Article {id: r.dst})
                     CREATE (a)-[:COCLICKED {weight: r.weight, norm_weight: r.norm_weight}]->(b)""",
               g["edges_article_coclicked_article_pruned"], "COCLICKED (pruned)")


if __name__ == "__main__":
    load(sys.argv[1] if len(sys.argv) > 1 else "test")
