# GNN news recommender (MIND-small)

A news recommender that represents articles, subcategories, Wikidata entities and co-click links as
one graph and ranks articles with a two-layer graph neural network (PyTorch Geometric). It was evaluated
offline on Microsoft's [MIND-small](https://msnews.github.io/) dataset under an evaluation plan fixed
before the test day was scored. **Version 1 covers news only**; a second domain (music) is future work.

## Results (test: 15 Nov 2019, 73,152 impressions)

| Model | Test AUC (5-seed mean, min–max) | nDCG@10 |
|---|---|---|
| **GNN** | **0.6636** (0.6539–0.6710) | **0.4140** |
| A0: same model trained without the graph | 0.6583 (0.6549–0.6629) | 0.4048 |
| Embedding baseline: frozen MiniLM text vectors, untrained | 0.6369 | 0.3936 |
| TF-IDF content baseline | 0.5778 | 0.3577 |
| Popularity baseline | 0.5318 | 0.3095 |

Embedding, TF-IDF and popularity are deterministic, so they have a single value.

**Most important findings**

1. **Training on clicks gave most of the gain.** A0 beats the untrained embedding baseline by +0.021 AUC,
   clear of zero in all 5 seeds. That is about four-fifths of the GNN's total gain (+0.027).
2. **The graph did not measurably improve on a trained text-only model.** GNN vs A0 is +0.005 AUC on
   average, but one seed's 95% interval includes zero, so the pre-registered win criterion (every seed's
   interval excludes zero) is not met.
3. **Seed variance and a confound were caught before testing.**
   - Training seeds varied by 0.016 AUC on validation, more than most effects. The win criterion was
     tightened to 5 seeds before the test run.
   - Comparing a trained GNN with an untrained baseline mixed two effects, so the A0 control was added.
   - The order of these changes is visible in git history (see the write-up).

**Read more:**
- [Write-up](docs/writeup.md): the full results, findings, limitations and future work.
- [Design decisions (ADRs 001–009)](docs/decisions/).
- Notebooks:
  - [01 data audit](notebooks/01_mind_audit.ipynb)
  - [02 baselines](notebooks/02_baselines.ipynb)
  - [03 graph exploration in Neo4j](notebooks/03_graph_exploration.ipynb)
  - [04 GNN test results](notebooks/04_gnn_results.ipynb)
- Detailed reports in [`reports/`](reports/): validation results, ablations, graph statistics, Wikidata
  types and data gaps.

## Non-goals

- **No live A/B testing.** There is no production traffic.
- **All evaluation is offline**, on held-out days of the public dataset.
- Not a production serving system.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`. PyTorch is pinned to
2.12 so that PyG's neighbour sampler (`pyg-lib`) has a macOS arm64 wheel (ADR 007).

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

## Reproduce

Run these in order from the project root. Times are approximate, from a laptop CPU (Apple Silicon).
Outputs go to `data/` (git-ignored), `reports/` and `mlruns/` (MLflow, git-ignored).

**1. Data audit**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_mind_audit.ipynb   # ~30 s
```

**2. Baselines**

```bash
uv run python -m recsys.experiments.run_baselines        # random, popularity, TF-IDF (tuned) on validation and test, ~3 min
uv run python -m recsys.experiments.compare_baselines    # paired bootstrap TF-IDF vs popularity, ~1.5 min
```

**3. Knowledge graph**

```bash
uv run python -m recsys.data.wikidata                    # Wikidata labels/types via SPARQL, ~20 min first run (cached after)
uv run python -m recsys.graph.build                      # graph tables for both setups, ~45 s
uv run python -m recsys.graph.report                     # reports/graph_stats.md, ~2 min
uv run python -m recsys.graph.gaps                       # reports/milestone3_gaps.md, ~3 min
```

**4. Features and PyG graphs**

```bash
uv run python -m recsys.features.build                   # MiniLM text embeddings etc., ~3 min (downloads the model once)
uv run python -m recsys.graph.pyg                        # HeteroData per setup, ~10 s
uv run python -m recsys.graph.periods                    # training and scoring graphs per period (ADR 008), ~3 min
uv run python -m recsys.graph.periods raw-coclick         # raw co-click graphs for ablation A4, ~1 min
```

**5. Optional: Neo4j exploration** (see [Neo4j](#neo4j-optional-for-exploration) below)

```bash
colima start --memory 4 && docker compose up -d
uv run python -m recsys.graph.neo4j_load                 # ~15 s
```

**6. GNN on validation**

```bash
uv run python -m recsys.training.train_gnn --setup validation --seed 0   # early stopping, ~15 min; prints the MLflow run id
uv run python -m recsys.experiments.validation_gnn <run_id>              # all models on 14 Nov, ~10 min
```

**7. Ablations (validation)**

```bash
uv run python -m recsys.experiments.ablations reference <run_id>                                  # ~1 min
uv run python -m recsys.experiments.ablations train A1 A2 A3 A4 A5 A6 A7 A8 R1                    # round 1, early stopping, ~1 h
uv run python -m recsys.experiments.ablations train A0 A1 A2 A3 A4 A5 A6 A7 A8 R1 A9 --epochs 9    # round 2-3, seed 0, ~2 h
uv run python -m recsys.experiments.ablations train REF A0 A1 A2 A3 A4 A5 R1 --epochs 9 --seed 1   # 3-seed reruns, ~1.5 h
uv run python -m recsys.experiments.ablations train REF A0 A1 A2 A3 A4 A5 R1 --epochs 9 --seed 2   # ~1.5 h
uv run python -m recsys.experiments.ablations train REF --epochs 13                               # A8 at its own length, ~15 min
uv run python -m recsys.experiments.ablations report                                              # reports/gnn_ablations.md, ~25 min
```

**8. Final test run (ADR 009)**

```bash
uv run python -m recsys.experiments.final_test train     # GNN and A0, seeds 0-4, 9 epochs each, ~2 h
uv run python -m recsys.experiments.final_test report    # bootstraps, verdicts, beyond-accuracy, ~40 min
```

**9. Notebooks**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb          # ~10 s
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_graph_exploration.ipynb  # needs Neo4j, ~20 s
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_gnn_results.ipynb        # ~10 s
```

Results are reproducible to about 0.001 AUC, not exactly: the neighbour sampler is not deterministic
under a fixed seed (ADR 008).

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

- **Comparing models:** use a paired bootstrap (`recsys.evaluation.paired_bootstrap`).
  It resamples impressions with replacement (1,000 resamples, fixed seed) and reports the
  difference in AUC and nDCG@10 with a 95% interval, overall and per segment. Only claim
  an improvement when the interval excludes zero.

Code: `src/recsys/data/splits.py` and `src/recsys/evaluation/`.

## News graph

```bash
uv run python -m recsys.data.wikidata      # Wikidata labels/types (cached in data/interim/; runs once)
uv run python -m recsys.graph.build        # parquet graph tables for both setups -> data/processed/graph/
uv run python -m recsys.graph.report       # reports/graph_stats.md
```

Design decisions are recorded in `docs/decisions/` (ADRs 001–009).

### Neo4j (optional, for exploration)

Neo4j 5 Community runs in Docker via [Colima](https://github.com/abiosoft/colima).
**Colima does not start automatically after a reboot**, so start it first:

```bash
colima start --memory 4                    # Neo4j needs more than Colima's 2 GB default
docker compose up -d                       # Neo4j on localhost:7474 (browser) and :7687 (Bolt)
uv run python -m recsys.graph.neo4j_load   # wipe and load the test setup (pruned co-click edges only)
```

Default local credentials are `neo4j` / `recsys-local-dev`; override with `NEO4J_PASSWORD`.
Then open `notebooks/03_graph_exploration.ipynb`. See `docs/rdf_vs_property_graph.md` for
why both Wikidata (RDF) and Neo4j (property graph) are used.

## Layout

```
configs/       experiment and pipeline configs
docs/          write-up, decision records (decisions/), RDF vs property graph note
data/          raw/, interim/, processed/ (git-ignored, never committed)
notebooks/     exploration and audits (01_mind_audit.ipynb, ...)
reports/       written findings; figures/ for saved plots
src/recsys/    the importable package
tests/         pytest tests
```
