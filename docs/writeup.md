# Does a knowledge graph help news recommendation? A pre-registered test on MIND-small

**Summary.**
- **What was built:** a news recommender for MIND-small that represents articles, subcategories,
  Wikidata entities and co-click links as one graph, and scores articles with a two-layer GNN.
- **How it was tested:** against an untrained text-embedding baseline and against A0, the same
  model trained without the graph. The evaluation plan was fixed before the test day was scored.
- **Headline test numbers:** the GNN beats the embedding baseline by **+0.027 AUC** (5 seeds, all
  clear of zero), and beats A0 by **+0.005 AUC** (4 of 5 seeds clear of zero).
- **Conclusion:** under the pre-registered criterion, **the graph did not measurably improve on a
  trained text-only model.** Most of the gain came from training on clicks.

Version 1 of a graph-based recommender for news. This document summarises the problem, the
design, the evaluation protocol, the results and what they do and do not show. Every number
comes from the files in `reports/`; design decisions are recorded in `docs/decisions/` (ADRs).

## 1. Problem and data

**Task.** Given a user's reading history and an impression (a list of news articles shown at one
moment), rank the articles so that the ones the user clicks come first.

**Data: Microsoft MIND-small.** Six days of MSN News logs (9–14 Nov 2019) for training, plus one day
(15 Nov) released as the dev set. There are 50,000 users per split and 65,238 articles. Each
article has a title, an abstract, a category and subcategory, and entity links to Wikidata
(`notebooks/01_mind_audit.ipynb`).

Two properties of the data shape everything below:

- **Most test users are new.** 88% of test impressions come from users with no activity in the
  training days. But 97% have a non-empty reading history, taken as a snapshot before the log
  window.
- **Many test articles are new.** 45% of the articles shown on the test day never appear in the
  training days, and they are clicked more often than familiar ones (CTR 4.7% vs 3.9%).

**Approach.** A heterogeneous graph with no user nodes:
- **Nodes:** articles, subcategories, categories, Wikidata entities and coarse entity types.
- **Edges:** article–subcategory, subcategory–category, article–entity, entity–type, and pruned
  article–article co-click edges.

A two-layer GNN produces article vectors from frozen text embeddings and the graph. A user is
the mean of their recent articles' vectors, and the score is a dot product.

## 2. Key design decisions

| ADR | Decision |
|---|---|
| [001](decisions/001-article-centred-inductive-graph.md) | Article-centred inductive graph with no user nodes, because 88% of test users are unseen |
| [002](decisions/002-coclick-edges-training-period-only.md) | Co-click edges only from training-period clicks and training users' histories |
| [003](decisions/003-subcategory-node-type.md) | Subcategory as a node type (checked by ablation) |
| [004](decisions/004-coclick-pruning-and-normalisation.md) | Co-click edges: at least 2 shared users, cosine-normalised, top 20 neighbours per article |
| [005](decisions/005-entity-type-mapping.md) | Six coarse entity types from Wikidata via SPARQL, with priority rules and manual overrides |
| [006](decisions/006-starting-features.md) | Starting features: frozen MiniLM text embeddings, MIND entity vectors with a missing flag, pre-target subcategory means |
| [007](decisions/007-gnn-architecture.md) | Custom weighted-mean layer, 2 layers × 128 dimensions, neighbour sampling 10/5, mean-pooled user vector |
| [008](decisions/008-training-design.md) | Softmax over 1 click + 4 same-impression negatives; separate graph and training-example periods |
| [009](decisions/009-evaluation-plan.md) | Comparisons, the win criterion, segments and the ablation rules, fixed before the test run |

## 3. Evaluation protocol

**Time-based splits.** Every evaluation predicts a day the model has not seen.

| Split | Model fitted on | Scored on | Used for |
|---|---|---|---|
| validation | 9–13 Nov | 14 Nov | all tuning, early stopping and ablations |
| test | 9–14 Nov (refit) | 15 Nov (MIND dev) | the final models only, once |

**Graph period vs training-example period (ADR 008).** A model is trained on an earlier graph than
the impressions it learns from. For the test setup that is a 9–12 Nov graph with 13–14 Nov
training examples, and the 15 Nov test day is scored with a graph rebuilt from 9–14 Nov.
- **Why:** a training click cannot create the co-click edge used to predict it.
- **Side benefit:** articles that first appear in the example period give the model practice
  with new articles.

**Metrics.** AUC, MRR, nDCG@5 and nDCG@10 are computed per impression and averaged, as in the
official MIND scorer. Tied scores get their exact expected value under random ordering. Results
are reported for segments: known and new users, seen and unseen articles, and empty histories.

**Significance.** A paired bootstrap resamples impressions (1,000 resamples) and gives a 95% interval
for the difference between two models.

**Seeds and the win criterion (pre-registered in ADR 009 before the test run).**
- Each trained model was run with 5 seeds (0–4).
- A model "wins" only if **every seed's 95% interval for the AUC difference on all test
  impressions excludes zero**, on the same side.
- nDCG@10 is secondary and cannot turn a loss into a win.
- **The primary question, "does the graph help?",** is answered by the GNN against **A0**: the
  same model, trained the same way, with every edge removed.

## 4. Results (test, 15 Nov, 73,152 impressions)

**Accuracy on all test impressions.** Trained models show the mean over 5 seeds, with the min–max
range in brackets.

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **GNN** | **0.6636** (0.6539–0.6710) | **0.3182** | **0.3512** | **0.4140** (0.4078–0.4185) |
| A0: trained, no graph | 0.6583 (0.6549–0.6629) | 0.3110 | 0.3394 | 0.4048 (0.4018–0.4077) |
| Embedding baseline: frozen MiniLM, untrained | 0.6369 | 0.3058 | 0.3336 | 0.3936 |
| TF-IDF content baseline | 0.5778 | 0.2777 | 0.2963 | 0.3577 |
| Popularity: training-period clicks | 0.5318 | 0.2380 | 0.2457 | 0.3095 |
| Random | 0.5008 | 0.2189 | 0.2239 | 0.2868 |

TF-IDF, popularity and random come from Step 2 (`reports/baselines.csv`). They were fitted on the
same 9–14 Nov period and scored on the same impressions.

**Pre-registered comparisons.** AUC on all test impressions; per-seed paired bootstrap.

| Comparison | Mean difference | Min to max | Seeds whose interval excludes zero | Outcome |
|---|---|---|---|---|
| **GNN vs A0 (primary)** | **+0.0053** | −0.0010 to +0.0126 | 4 of 5 | **criterion not met** |
| GNN vs embedding | +0.0267 | +0.0170 to +0.0341 | 5 of 5 | GNN wins |
| A0 vs embedding | +0.0214 | +0.0180 to +0.0260 | 5 of 5 | A0 wins |

**Beyond accuracy.** Measured on each impression's top 5; trained models show the mean over 5 seeds.

| Model | Share of top-5 slots given to unseen articles | Coverage: candidates in any top 5 | Distinct subcategories in a top 5 |
|---|---|---|---|
| GNN | 0.285 | 0.488 | 3.91 |
| A0 | 0.247 | 0.470 | 4.18 |
| Embedding baseline | 0.222 | 0.522 | 4.08 |

For comparison, unseen articles receive 23.2% of test clicks and make up 20.0% of candidates.

Full tables are in `notebooks/04_gnn_results.ipynb` and `reports/gnn_test_*.csv`.

## 5. Findings

1. **Training on clicks gave most of the gain.**
   - A0, the model trained on clicks without any graph, beats the untrained embedding baseline by
     **+0.021 test AUC** (+0.018 to +0.026), and every one of the 5 seeds' intervals excludes zero.
   - That is about four-fifths of the GNN's total gain over the same baseline (+0.027).

2. **The graph's additional effect did not meet the pre-registered criterion.**
   - GNN vs A0 is **+0.005 AUC on average** and positive in 4 of 5 seeds.
   - Seed 3's interval includes zero (−0.0010, 95% CI −0.0024 to +0.0005), so under ADR 009 the
     graph's contribution is not established.
   - On nDCG@10, the GNN is ahead of A0 in all 5 seeds (+0.006 to +0.015, every interval excluding
     zero). nDCG@10 was pre-registered as the **secondary** metric, which cannot turn a loss into a
     win (ADR 009). Its result is consistent with a small graph effect but does not change the
     verdict.

3. **On unseen articles, validation and test disagree, and the test result is unexplained.**
   - **On validation,** training on clicks hurt unseen articles: A0 was 0.013 AUC below the embedding
     baseline there, with all 3 seeds below zero.
   - **On test,** the GNN was ahead of A0 on unseen articles in all 5 seeds (+0.012 to +0.019), and A0
     was ahead of the embedding baseline there too (+0.017).
   - **Why it's reported as unexplained:** this segment result was not pre-registered, and it
     contradicts validation.
   - **One untested hypothesis:** the test day is scored with a longer graph (9–14 Nov) that gives new
     articles more context.

4. **A confound was found and controlled before the test run.**
   - The original plan compared the GNN with the embedding baseline. That mixes two things: the GNN
     is trained on clicks, and the baseline is untrained.
   - The symptom on validation: removing co-click edges (ablation A1) made no measurable difference,
     even on seen articles, where the GNN's gain over the baseline appeared to be.
   - **The control:** A0 (same model, no edges) was added to separate the graph from training.
   - **The result:** on validation, GNN vs A0 was −0.000 on the test-mix weighted row, so the earlier
     gain came from training, not from the graph.
   - **What changed:** GNN vs A0 became the primary comparison before the test set was touched.

5. **Variation between training seeds was larger than most ablation effects, so the win criterion
   was tightened before the test run.**
   - **The spread:** on validation, three seeds of the same GNN scored 0.651, 0.635 and 0.640
     test-mix AUC, a spread of 0.016.
   - **The bootstrap misses it:** the bootstrap covers data noise only.
   - **Examples:** two ablations that looked clear at seed 0 (A1, A4) were within noise over
     3 seeds.
   - **The change:** the original criterion (seed 0's interval plus a sign check over 3 seeds) was
     replaced by "every one of 5 seeds' intervals excludes zero". Under the earlier rule, the
     test result for GNN vs A0 would have counted as a win: seed 0's interval excludes zero, and
     seeds 0–2 are all positive.
   - **The order can be checked in git history.** The rule change was made on validation evidence,
     before any test results existed:
     - commit `c5cfdbe` (21 Oct) adds the 5-seed criterion to ADR 009;
     - commit `7e9c630` (23 Oct) makes GNN vs A0 the primary comparison;
     - the test models were trained and scored only after that, and the results were committed in
       `961446c` (27 Oct).

     The test score files themselves are not tracked by git (data stays local), so git records the
     order of the commits, not the moment each file was written.

6. **Data-quality findings.**
   - **Popularity bias in raw co-click edges.** 74% of raw co-click pairs come from the 10% of
     users with the longest histories. Ranked by raw shared-user counts, one viral celebrity
     story was a top-5 neighbour of a history column, an NFL trade and a cornbread recipe.
     Cosine normalisation and pruning gave topical neighbours (ADR 004, `reports/graph_stats.md`).
   - **MIND entity-type errors caught by Wikidata.** On MIND's three unambiguous type codes,
     Wikidata types agree 96.3% of the time. Most disagreements are MIND errors, for example bands
     (Red Hot Chili Peppers, The Roots) coded as persons (ADR 005, `reports/wikidata_types.md`).
   - **Entity-linking errors.** Articles from 2019 are linked to "October 2007 California
     wildfires" (29 articles) and to "Middle East Economic Survey" (53 articles), most likely
     ambiguous surface forms resolved to the wrong item (`reports/milestone3_gaps.md`).
   - **Missing labels.** Wikidata labels stored only under the language-neutral `mul` tag were
     missed at first, and recovered with a fallback.

## 6. Limitations

- **Offline evaluation only.** No live traffic and no A/B test; clicks on shown impressions are the
  only signal.
- **One dataset.** MIND-small, one news provider and one region.
- **Six days of data.** Short graph periods; seasonality and longer-term interests are not represented.
- **Small effects are hard to detect.** With this much data, an effect of about 0.005 AUC is smaller
  than the variation between training seeds (about 0.016 on validation).
- **Not fully deterministic.** The neighbour sampler (`pyg-lib`) is not deterministic under a fixed
  seed; reruns of the same seed differ by about 0.001 AUC.
- **Validation and test disagree on unseen articles** (finding 3), which limits what the validation
  day can say about the test day for that segment.
- **No reading-time data.** A click is the only feedback, so there is no model of time spent or
  satisfaction.

## 7. Future work

1. **Version 2 on MIND-large:**
   - PySpark for data processing and co-click pair generation;
   - a fresh, untouched test set (the MIND-large dev split);
   - a re-test of whether the graph helps with more data, under the same pre-registered protocol.
2. **A LightGBM re-ranker** combining the GNN, A0 and text scores with co-click, freshness and
   popularity features.
3. **Diversity-aware re-ranking and near-duplicate removal.** The GNN's top-5 lists span fewer
   subcategories (3.91 vs 4.08–4.18), and the same story often appears as several articles.
4. **Dedicated cold-start retrieval for new articles,** given that unseen articles are 45% of test
   candidates and the evidence on how to rank them is mixed.
5. **Attention pooling in the user encoder (A8).** Within noise at 9 epochs, borderline positive at
   13 epochs (+0.005, interval +0.001 to +0.009); not tested further.
6. **A second domain with a different graph schema** (the original plan: music from Last.fm), to see
   whether the protocol and findings carry over.
