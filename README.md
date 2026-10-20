# Multi-domain graph-based recommender

A personal project to build a recommender system that models users, items and
knowledge-graph entities as one graph, across more than one content "brand".

**Version 1 covers two domains:**

- **News**: [Microsoft MIND-small](https://msnews.github.io/), later enriched with Wikidata entities
- **Music**: Last.fm (added in a later stage)

**Goal:** learn how far graph structure (shared users, shared entities, cross-domain
links) improves ranking quality and cold-start behaviour over strong non-graph
baselines. Later stages bring in PySpark, Neo4j, PyTorch Geometric, LightGBM and a
vector database.

## Non-goals

- **No live A/B testing.** There is no production traffic.
- **All evaluation is offline**, on held-out splits of the public datasets
  (e.g. AUC, MRR, nDCG@k on MIND's dev impressions).
- Not a production serving system. Latency and infrastructure only matter where they
  shape modelling choices.

## Layout

```
configs/       experiment and pipeline configs
data/          raw/, interim/, processed/ (git-ignored, never committed)
notebooks/     exploration and audits (01_mind_audit.ipynb, ...)
reports/       written findings; figures/ for saved plots
src/recsys/    the importable package
tests/         pytest tests
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

```bash
uv sync                      # create .venv and install dependencies
uv run pytest                # run tests
uv run jupyter lab           # open the notebooks
```

### Data

Download MIND-small and place it as:

```
data/raw/mind/train/{behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec}
data/raw/mind/dev/{behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec}
```
