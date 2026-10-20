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

## Evaluation protocol (news / MIND-small)

Splits are by time, so every evaluation predicts days the model has not seen:

| Split | Fit on (training period) | Scored on | Use |
|---|---|---|---|
| Validation | MIND train, 9–13 Nov 2019 | MIND train, 14 Nov 2019 | tuning and model selection only |
| Test | MIND train, 9–14 Nov 2019 | MIND dev, 15 Nov 2019 | final reporting only, never for tuning |

- **Protocol:** tune on validation (fit on 9–13 Nov), freeze the hyperparameters, then
  refit on 9–14 Nov and score test once. Known users and seen articles for the test
  segments are recomputed against 9–14 Nov.
- **Primary tuning metric:** about 84% of validation impressions come from known users,
  versus about 12% on test. Validation results therefore include a **test-mix weighted**
  row, which reweights the known-user and new-user segment scores to the test split's
  exact known-user share. Tune on that row together with the new-user segment.
- Click counts, known user and article IDs, graph edges, and anything else learned from
  interactions come from the split's training period only. At scoring time a model may use what
  a live system would have: the user's history snapshot and the candidate list, never
  the labels. MIND histories are fixed snapshots taken before the log window, so they
  contain no validation- or test-day clicks.
- Metrics (as in the official MIND scorer): AUC, MRR, nDCG@5 and nDCG@10, computed per
  impression and averaged. Ties between equal scores are resolved by their exact expected
  value under random ordering, so candidate order cannot affect results.
- Segment breakdowns: **known vs new users** (whether the user has an impression in the
  training period), and **seen vs unseen articles** (whether the article appears in a
  training-period history or candidate list; each candidate list is restricted to that
  segment and re-ranked).

Code: `src/recsys/data/splits.py` and `src/recsys/evaluation/`.

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
