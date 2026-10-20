# RDF vs property graph in this project

Two graph storage models appear in this project. Each was used for the job it suits;
neither is the source of truth. The canonical graph is the set of parquet tables in
`data/processed/graph/`, which models read directly.

## RDF (Wikidata, queried with SPARQL)

**Used for:** enriching MIND's 32,472 entity IDs with English labels and "instance of"
types, then classifying 5,427 types into coarse classes (`src/recsys/data/wikidata.py`, ADR 005).

**Why RDF fits here**

- **Shared global identifiers.** MIND already links entities to Wikidata IDs (`Q22686`). In RDF
  an ID is a URI, so the same entity can be looked up without any mapping step. Last.fm
  artists can be linked to the same vocabulary later, which is what will connect the news
  and music domains.
- **Schema and data in one graph.** Classes are themselves nodes, so a single query can walk
  the class hierarchy: `?type wdt:P279* wd:Q43229` ("is this type a kind of organisation?").
  This transitive reasoning over an open, community-maintained ontology is what RDF and
  SPARQL are built for.
- **Open world.** Data is incomplete and evolving: 2,142 entities have no type, and labels
  moved to the language-neutral `mul` tag. RDF tolerates missing statements, and queries
  use `OPTIONAL`.

**Costs we hit:** a public endpoint with rate limits (hence batching, backoff and a local
cache), and silent changes in conventions (the `mul` labels).

## Property graph (Neo4j, queried with Cypher)

**Used for:** loading the derived test-setup graph (articles, subcategories, categories,
entities, entity types, *pruned* co-click edges) for interactive exploration and sanity checks
(`src/recsys/graph/neo4j_load.py`, `notebooks/03_graph_exploration.ipynb`).

**Why a property graph fits here**

- **Properties on edges.** Our edges carry data: co-click `weight` and `norm_weight`, and mention
  `source` and `confidence`. In a property graph these are plain relationship properties. In
  RDF they need reification or RDF-star, which makes every edge query more awkward.
- **Path and neighbourhood queries read naturally.** "Articles sharing entities with this
  unseen article, weighted by entity degree" is a few lines of Cypher pattern matching, and it
  mirrors the message passing a GNN will do.
- **A closed, application-specific schema.** Our node and edge types are fixed by the decision
  records; we don't need a shared ontology or global URIs inside this graph.

**Costs:** it is a second copy of the data (it must be reloaded after rebuilding the tables), it
needs a running service (Colima and Docker), and it is not where training happens.

## Summary

| | RDF / SPARQL (Wikidata) | Property graph / Cypher (Neo4j) |
|---|---|---|
| Role | external knowledge: labels, types, class hierarchy | exploration and sanity checks of our own graph |
| Strength used | global IDs, transitive `P279*` over an open ontology | edge properties, readable path queries |
| Data | read-only, remote, cached locally | derived from parquet, reloadable in about 15 s |
| Source of truth? | no (cached snapshot in `data/interim/`) | no (parquet is canonical) |
