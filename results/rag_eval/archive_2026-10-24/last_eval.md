# RAG Evaluation Report

_Generated: 2026-10-08T19:03:17.226293+00:00_

## Summary

| Metric | Value |
|--------|-------|
| Questions (all / retrieval-scored) | 100 / 80 |
| Index papers / chunks | 10 / 1725 |
| Embedding | google/embeddinggemma-2 @ dim=None |
| Mean Recall@k | 0.85 |
| Mean MRR | 0.75 |
| Mean hit@1 | 0.6875 |
| Mean top score | 0.7931 |
| Refusal recall / precision | 0.45 / 0.45 |
| Refusal accuracy | 0.78 |
| Cache hit rate | 1.0 |
| Latency p50 / p95 (ms) | 66.7 / 73.9 |

## By question kind (retrieval-scored only)

| Kind | n | MRR | Recall@k | Hit@1 |
|------|---|-----|----------|-------|
| comparison | 5 | 0.5167 | 0.8 | 0.4 |
| definition | 12 | 0.6944 | 0.75 | 0.6667 |
| identifier | 10 | 0.4167 | 0.7 | 0.2 |
| method | 14 | 0.625 | 0.7143 | 0.5714 |
| multi_hop | 1 | 1.0 | 1.0 | 1.0 |
| open_problem | 13 | 0.9487 | 1.0 | 0.9231 |
| paraphrase | 25 | 0.9133 | 0.96 | 0.88 |

## By difficulty

| Difficulty | n | MRR | Recall@k | Hit@1 |
|------------|---|-----|----------|-------|
| easy | 21 | 0.8016 | 0.8571 | 0.7619 |
| hard | 24 | 0.7535 | 0.8333 | 0.7083 |
| medium | 35 | 0.7167 | 0.8571 | 0.6286 |

**95% bootstrap CI** — MRR (0.6562, 0.8375), Recall@k (0.7625, 0.925), Hit@1 (0.575, 0.7875)
**Failure hints:** `{'over_refuse': 5, 'missing_or_ranking': 12, 'ranking': 13, 'false_answer': 11}`

## Per-question

| ID | Kind | Recall@k | MRR | Hit@1 | Top | Coverage | Refuse OK | ms |
|----|------|----------|-----|-------|-----|----------|-----------|----|
| ntk_01 | definition | 1.0 | 1.0 | 1.0 | 0.8406 | high | True | 145.1 |
| ntk_02 | definition | 1.0 | 1.0 | 1.0 | 0.8178 | high | True | 70.0 |
| ntk_03 | definition | 1.0 | 1.0 | 1.0 | 0.8385 | high | True | 63.1 |
| ntk_04 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8383 | high | True | 72.6 |
| ntk_05 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8332 | high | True | 74.2 |
| ntk_06 | method | 1.0 | 1.0 | 1.0 | 0.8291 | high | False | 72.9 |
| ntk_07 | definition | 1.0 | 1.0 | 1.0 | 0.8576 | high | True | 66.7 |
| ntk_08 | paraphrase | 1.0 | 1.0 | 1.0 | 0.821 | high | True | 72.7 |
| ntk_09 | identifier | 0.0 | 0.0 | 0.0 | 0.7687 | high | False | 77.4 |
| ntk_10 | definition | 1.0 | 1.0 | 1.0 | 0.855 | high | True | 68.0 |
| ntk_11 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8465 | high | True | 67.5 |
| ntk_12 | method | 1.0 | 1.0 | 1.0 | 0.8251 | high | True | 67.7 |
| op_01 | open_problem | 1.0 | 1.0 | 1.0 | 0.8755 | high | True | 61.7 |
| op_02 | open_problem | 1.0 | 1.0 | 1.0 | 0.8383 | high | True | 64.0 |
| op_03 | open_problem | 1.0 | 1.0 | 1.0 | 0.8829 | high | True | 68.9 |
| op_04 | open_problem | 1.0 | 0.3333 | 0.0 | 0.7928 | high | True | 62.2 |
| op_05 | open_problem | 1.0 | 1.0 | 1.0 | 0.8222 | high | True | 64.1 |
| op_06 | paraphrase | 1.0 | 0.5 | 0.0 | 0.8157 | high | True | 69.2 |
| op_07 | open_problem | 1.0 | 1.0 | 1.0 | 0.8743 | high | False | 69.8 |
| op_08 | definition | 1.0 | 1.0 | 1.0 | 0.8765 | high | True | 68.8 |
| op_09 | identifier | 1.0 | 1.0 | 1.0 | 0.8423 | high | True | 74.2 |
| op_10 | open_problem | 1.0 | 1.0 | 1.0 | 0.7975 | high | True | 65.1 |
| acdc_01 | method | 1.0 | 0.5 | 0.0 | 0.835 | high | True | 69.7 |
| acdc_02 | method | 1.0 | 1.0 | 1.0 | 0.8224 | high | True | 63.7 |
| acdc_03 | definition | 1.0 | 0.3333 | 0.0 | 0.8466 | high | True | 67.6 |
| acdc_04 | method | 1.0 | 0.25 | 0.0 | 0.8178 | high | True | 62.6 |
| acdc_05 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8582 | high | True | 64.0 |
| acdc_06 | identifier | 0.0 | 0.0 | 0.0 | 0.7414 | high | True | 68.6 |
| acdc_07 | method | 0.0 | 0.0 | 0.0 | 0.7864 | high | True | 66.5 |
| rev_01 | definition | 1.0 | 1.0 | 1.0 | 0.872 | high | True | 62.1 |
| rev_02 | open_problem | 1.0 | 1.0 | 1.0 | 0.8413 | high | True | 69.5 |
| rev_03 | open_problem | 1.0 | 1.0 | 1.0 | 0.882 | high | True | 62.9 |
| rev_04 | method | 1.0 | 1.0 | 1.0 | 0.8748 | high | False | 63.4 |
| rev_05 | method | 1.0 | 1.0 | 1.0 | 0.8825 | high | True | 64.7 |
| rev_06 | method | 1.0 | 1.0 | 1.0 | 0.8525 | high | False | 63.1 |
| rev_07 | method | 0.0 | 0.0 | 0.0 | 0.8067 | high | False | 64.8 |
| rev_08 | definition | 0.0 | 0.0 | 0.0 | 0.8508 | high | True | 66.5 |
| rev_09 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8468 | high | True | 62.7 |
| rev_10 | open_problem | 1.0 | 1.0 | 1.0 | 0.8242 | high | True | 64.3 |
| cmp_01 | comparison | 1.0 | 0.25 | 0.0 | 0.8303 | high | True | 68.1 |
| cmp_02 | comparison | 1.0 | 1.0 | 1.0 | 0.7896 | high | True | 66.4 |
| cmp_03 | comparison | 1.0 | 0.3333 | 0.0 | 0.78 | high | True | 61.5 |
| cmp_04 | multi_hop | 1.0 | 1.0 | 1.0 | 0.7692 | high | False | 69.8 |
| cmp_05 | comparison | 1.0 | 1.0 | 1.0 | 0.7943 | high | True | 63.2 |
| para_01 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8338 | high | True | 67.1 |
| para_02 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8481 | high | True | 61.9 |
| para_03 | paraphrase | 1.0 | 0.3333 | 0.0 | 0.8003 | high | True | 65.8 |
| para_04 | paraphrase | 1.0 | 1.0 | 1.0 | 0.7934 | high | True | 69.0 |
| para_05 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8076 | high | True | 62.8 |
| para_06 | paraphrase | 0.0 | 0.0 | 0.0 | 0.8002 | high | True | 63.9 |
| para_07 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8036 | high | True | 63.8 |
| para_08 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8875 | high | True | 67.4 |
| id_01 | identifier | 0.0 | 0.0 | 0.0 | 0.7573 | high | True | 73.7 |
| id_02 | identifier | 1.0 | 0.3333 | 0.0 | 0.752 | high | False | 69.7 |
| id_03 | identifier | 1.0 | 1.0 | 1.0 | 0.7933 | high | True | 73.9 |
| id_04 | identifier | 1.0 | 0.5 | 0.0 | 0.785 | high | False | 69.4 |
| id_05 | identifier | 1.0 | 0.3333 | 0.0 | 0.7268 | high | False | 68.8 |
| edge_01 | edge | 1.0 | 1.0 | 1.0 | 0.7468 | high | False | 63.1 |
| edge_02 | edge | 1.0 | 1.0 | 1.0 | 0.7327 | high | False | 68.4 |
| edge_03 | edge | 1.0 | 1.0 | 1.0 | 0.7332 | high | True | 66.3 |
| oos_01 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6803 | partial | False | 64.2 |
| oos_02 | unanswerable | 1.0 | 1.0 | 1.0 | 0.68 | partial | True | 62.6 |
| oos_03 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6662 | partial | False | 70.3 |
| oos_04 | unanswerable | 1.0 | 1.0 | 1.0 | 0.7097 | partial | True | 64.9 |
| oos_05 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6404 | partial | False | 66.7 |
| oos_06 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6316 | partial | False | 62.0 |
| oos_07 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6465 | partial | False | 70.3 |
| oos_08 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6255 | partial | False | 62.9 |
| oos_09 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6681 | partial | False | 65.7 |
| oos_10 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6757 | partial | False | 63.9 |
| oos_11 | unanswerable | 1.0 | 1.0 | 1.0 | 0.7084 | partial | True | 63.4 |
| oos_12 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6463 | partial | True | 66.6 |
| oos_13 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6904 | partial | True | 68.8 |
| oos_14 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6804 | partial | True | 68.6 |
| oos_15 | unanswerable | 1.0 | 1.0 | 1.0 | 0.6409 | partial | True | 68.9 |
| ntk_13 | definition | 1.0 | 1.0 | 1.0 | 0.824 | high | True | 65.9 |
| ntk_14 | method | 1.0 | 1.0 | 1.0 | 0.8122 | high | True | 64.9 |
| ntk_15 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8529 | high | True | 73.8 |
| op_11 | open_problem | 1.0 | 1.0 | 1.0 | 0.7911 | high | True | 67.2 |
| op_12 | open_problem | 1.0 | 1.0 | 1.0 | 0.7997 | high | True | 63.6 |
| op_13 | definition | 0.0 | 0.0 | 0.0 | 0.7534 | high | True | 63.5 |
| acdc_08 | method | 1.0 | 1.0 | 1.0 | 0.7975 | high | True | 68.7 |
| acdc_09 | method | 0.0 | 0.0 | 0.0 | 0.7979 | high | True | 62.5 |
| rev_11 | open_problem | 1.0 | 1.0 | 1.0 | 0.8592 | high | True | 62.2 |
| rev_12 | method | 0.0 | 0.0 | 0.0 | 0.7977 | high | True | 68.7 |
| rev_13 | definition | 0.0 | 0.0 | 0.0 | 0.8727 | high | True | 63.9 |
| cmp_06 | comparison | 0.0 | 0.0 | 0.0 | 0.7978 | high | False | 66.0 |
| para_09 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8363 | high | True | 66.2 |
| para_10 | paraphrase | 1.0 | 1.0 | 1.0 | 0.7787 | high | True | 62.0 |
| id_06 | identifier | 1.0 | 0.5 | 0.0 | 0.8053 | high | True | 68.9 |
| id_07 | identifier | 1.0 | 0.5 | 0.0 | 0.755 | high | True | 68.2 |
| edge_04 | edge | 1.0 | 1.0 | 1.0 | 0.7173 | high | False | 67.5 |
| edge_05 | edge | 1.0 | 1.0 | 1.0 | 0.7565 | high | True | 69.1 |
| pad_1 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8213 | high | True | 69.3 |
| pad_2 | paraphrase | 1.0 | 1.0 | 1.0 | 0.817 | high | True | 68.4 |
| pad_3 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8207 | high | True | 68.2 |
| pad_4 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8134 | high | True | 72.9 |
| pad_5 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8194 | high | True | 77.1 |
| pad_6 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8115 | high | True | 69.8 |
| pad_7 | paraphrase | 1.0 | 1.0 | 1.0 | 0.8193 | high | True | 69.2 |

## Interpretation (AIP concepts)

- **Recall@k / MRR / hit@1** — retrieval quality (Lab 3). Prefer MRR when hit@5 saturates.
- **Refusal recall & precision** — both required (Lab 4); only one hides failure modes.
- **Unanswerable items are noisy** at small n — report raw counts in the full report.
- **Coverage** — score-based gate before generation (grounding).
- **Latency p50/p95** — service SLO style (Lab 7).

## Not safe for

- Unsupervised clinical, legal, or financial decisions.
- Questions outside the ingested full-text set.
- Treating cosine similarity as calibrated probability.
