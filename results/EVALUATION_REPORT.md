# Evaluation Report — ArXiv Paper Suggester

_Generated: 2026-10-20T13:38:24.090662+00:00_  ·  Single harness: `src/final_eval.py`

## 1. What the system does

A researcher builds a paper library from arXiv, ranks papers for an interest, ingests full text, and asks questions answered only from those papers with citations. Embeddings: local EmbeddingGemma 2. Generation: Gemini (optional NVIDIA failsafe).

## 2. How to read the numbers (AIP concepts)

| Concept | Meaning here |
|---------|--------------|
| **Recall@k** | Fraction of labelled questions where *any* expected paper appears in the top-k retrieved papers (Lab 3). |
| **MRR** | Mean reciprocal rank of the first expected paper; rewards putting the right paper higher (Lab 3). Prefer over saturated hit@5. |
| **Hit@1** | Whether the top-1 paper is expected. Harder than Recall@k. |
| **95% bootstrap CI** | Uncertainty on the mean over this gold set (Lab 2). Overlapping CIs ⇒ do not claim a difference. |
| **Wilson CI** | Uncertainty on a proportion (refusal, citation validity). |
| **McNemar / paired bootstrap** | Whether v2 beats v1 on the *same* items (Lab 2). Requires `baseline_v1.json`. |
| **Refusal precision / recall** | Of declines, how many should have been declined; of items that should be declined, how many were (Lab 4). **Both** required. |
| **Citation validity** | Every `[n]` in the answer refers to a supplied source — code-checkable (Lab 4). |
| **Failure hints** | Coarse Lab 5 taxonomy: missing_or_ranking, ranking, false_answer, over_refuse. |
| **Cohen’s κ** | Agreement between human and LLM judge on faithfulness/correctness. **Not computed here** unless you add human labels + a judge (Lab 4). Retrieval metrics do not need κ. |

## 3. Headline results

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| Mean MRR | **0.2833** | (0.1979, 0.3708) |
| Mean Recall@k | **0.4125** | (0.3, 0.525) |
| Mean Hit@1 | **0.1875** | (0.1, 0.2875) |
| Mean top score | 0.8107 | — |
| Index | 3830 papers / 373818 chunks | — |
| Embedder | google/embeddinggemma-2 @ dim=None | — |
| Latency p50 / p95 | 10366.9 / 14949.1 ms | — |
| Cache hit rate | 0.0 | (generation runs) |

**Interpretation:** On 80 questions with expected papers, MRR=0.2833 means that on average the first relevant paper sits near rank ~3.53 when found. Recall=0.4125 means about 41% of those questions surface at least one expected paper in the top-k. The CI width reflects n≈80: small improvements inside the CI should not be treated as proven gains (Lab 2).

## 3.1 Generation, refusal, citations

| Metric | Value | 95% CI |
|--------|-------|--------|
| Refusal recall | 0.5 | (0.2993, 0.7007) |
| Refusal precision | 0.3125 | (0.1795, 0.4857) |
| Refusal accuracy | 0.68 | — |
| Citation validity rate | 0.96 | (0.9016, 0.9843) |
| Answers with any [n] | 88 / 100 | — |

Refusal confusion matrix:

|  | did refuse | did not |
|--|----------:|--------:|
| should refuse | TP 10 | FN 10 |
| should answer | FP 22 | TN 58 |

**Interpretation:** Refusal recall is “of out-of-scope items, how often we decline.” Precision is “of our declines, how often we were right to decline.” A system that refuses everything gets recall 1.0 and is useless (Lab 4). Cache hit rate was 0.0 — if this is near 1.0, refusal numbers may replay older answers; use `--no-cache` for a cold measurement. Citation validity near 1.0 is a **code-level** guarantee that cited indices exist; it does not prove factual correctness (that would need a judge + κ).

## 4. Stratification (Lab 3)

### By kind

| Kind | n | MRR | Recall@k | Hit@1 | MRR CI |
|------|---|-----|----------|-------|--------|
| comparison | 5 | 0.1667 | 0.4 | 0.0 | (0.0, 0.3667) |
| definition | 12 | 0.4583 | 0.5833 | 0.3333 | (0.2083, 0.7083) |
| identifier | 10 | 0.0 | 0.0 | 0.0 | (0.0, 0.0) |
| method | 14 | 0.2381 | 0.3571 | 0.1429 | (0.0595, 0.4286) |
| multi_hop | 1 | 0.0 | 0.0 | 0.0 | (0.0, 0.0) |
| open_problem | 13 | 0.1154 | 0.1538 | 0.0769 | (0.0, 0.3077) |
| paraphrase | 25 | 0.46 | 0.68 | 0.32 | (0.3033, 0.6167) |

### By difficulty

| Difficulty | n | MRR | Recall@k | Hit@1 |
|------------|---|-----|----------|-------|
| easy | 21 | 0.2936 | 0.5714 | 0.0952 |
| hard | 24 | 0.2292 | 0.2917 | 0.1667 |
| medium | 35 | 0.3143 | 0.4 | 0.2571 |

**Interpretation:** Identifier and comparison kinds are typically harder for dense-only retrieval (Lab 3 Q44-style exact IDs favor lexical matching). Open-problem and paraphrase rows usually lead when the supporting papers are ingested. Medium difficulty can look worse than hard if medium items reference papers still weakly ranked.

## 5. Failures and Pareto (Lab 5)

Failure hint counts: `{'ranking': 18, 'over_refuse': 2, 'missing_or_ranking': 47, 'false_answer': 10}`

| arXiv ID (expected but often not in top-k) | # questions |
|---------------------------------------------|------------:|
| 2501.16496 | 27 |
| 2304.14997 | 13 |
| 1806.07572 | 6 |
| 2407.13594 | 5 |
| 2506.18852 | 3 |
| 2504.13151 | 3 |
| 2406.11779 | 2 |
| 2404.14082 | 2 |
| 2502.20914 | 1 |

**Interpretation:** `missing_or_ranking` means the expected paper never appeared in top-k — either it is not ingested, or the embedder ranked other papers higher. If the PDF is already in the index, the fix is ranking (hybrid BM25, rerank, query rewrite), not another download. `ranking` means the paper was in top-k but not rank 1. `false_answer` / `over_refuse` are generation-side (Lab 4 operating point).

## 6. Calibration

### Coverage label vs actual Recall@k

| Coverage | n | Mean Recall@k |
|----------|---|---------------|
| high | 80 | 0.4125 |

### Top-score quintiles vs hit rate

| Quintile | Score range | n | Hit@1 | Recall |
|----------|-------------|---|-------|--------|
| 1 | 0.7736–0.8068 | 16 | 0.0 | 0.0625 |
| 2 | 0.8077–0.8242 | 16 | 0.125 | 0.3125 |
| 3 | 0.8244–0.8312 | 16 | 0.0625 | 0.4375 |
| 4 | 0.8324–0.8477 | 16 | 0.3125 | 0.5 |
| 5 | 0.8481–0.872 | 16 | 0.4375 | 0.75 |

**Interpretation:** If higher score quintiles show higher hit@1, the similarity score is at least ordinally useful as a confidence signal. If not, do not threshold scores as if they were calibrated probabilities.

## 7. Embedding dimension sweep (Matryoshka)

| Dim | MRR | MRR CI | Recall@10 | Hit@1 |
|-----|-----|--------|-----------|-------|
| 128 | 0.0 | (0.0, 0.0) | 0.0 | 0.0 |
| 256 | 0.0 | (0.0, 0.0) | 0.0 | 0.0 |
| 512 | 0.0 | (0.0, 0.0) | 0.0 | 0.0 |
| 768 | 0.0 | (0.0, 0.0) | 0.0 | 0.0 |

**Recommendation:** Consider EMBEDDING_DIM=128 (MRR=0.0 vs full 0.0; Δ≤0.02). Rebuild indexes after change.

This sweep embeds library **titles+abstracts**, not full-text RAG chunks. Changing `EMBEDDING_DIM` requires resetting the RAG collection and re-ingesting. If CIs for 512 and 768 overlap heavily, prefer the cheaper dim only for ops cost, not as a proven quality win (Lab 2 honesty).

## 8. Paired comparison vs baseline (Lab 2)

Baseline: `results/rag_eval/baseline_v1.json` (n=80)

| Test | Δ (v2 − v1) | 95% CI | p-value |
|------|-------------|---------|---------|
| Paired bootstrap Recall@k | -0.4375 | (-0.5625, -0.3125) | 0.0 |
| Paired bootstrap MRR | -0.4667 | (-0.5635, -0.3677) | 0.0 |
| McNemar (recall success) | b=38 c=3 | — | 0.0 |

**Interpretation:** McNemar uses only discordant items (v1 right/v2 wrong vs the reverse). Small p (e.g. < 0.05) supports a real improvement on this gold set. Large p ⇒ choose on cost/latency instead of claiming accuracy gains.

## 9. SLO / regression gate (Lab 7)

```json
{
  "gates": {
    "mean_mrr_min": 0.55,
    "mean_recall_min": 0.55,
    "refusal_precision_min": 0.5,
    "citation_validity_min": 0.95,
    "p95_latency_ms_max": 20000
  },
  "mrr_pass": false,
  "recall_pass": false,
  "latency_pass": true,
  "refusal_precision_pass": false,
  "citation_validity_pass": true,
  "all_pass": false
}
```

Gates are project-specific thresholds, not universal truths. Failing refusal precision while passing retrieval is a typical pattern: retrieval can be demo-ready while refusal is not deployment-ready.

## 10. What is not safe / not claimed

- Clinical, legal, or financial decisions without human review.
- Questions about papers outside the RAG index.
- Treating cosine similarity as a calibrated probability.
- Faithfulness/correctness **LLM-judge scores** or **Cohen’s κ** — not run (would need human labels).
- Gold-context decomposition (oracle chunks vs retrieved) — not run; residual error is only coarsely attributed via fail hints.
- Lab 6 attack block rates — out of scope unless tool-use is productized.

## 11. Executive recommendation

With 3830 papers indexed, MRR=0.2833 (CI (0.1979, 0.3708)) and Recall@k=0.4125 on 80 labelled items. Residual fail hints: {'ranking': 18, 'over_refuse': 2, 'missing_or_ranking': 47, 'false_answer': 10}. Most frequent top-k miss: 2501.16496 (n=27) — treat as ranking if the PDF is already ingested. Refusal P/R=0.3125/0.5; citation validity=0.96. Consider EMBEDDING_DIM=128 (MRR=0.0 vs full 0.0; Δ≤0.02). Rebuild indexes after change. Suitable for a constrained portfolio demo on the ingested MI/NTK library; not for unsupervised refusal-critical deployment until OOS precision is stronger.

## 12. Reproducibility

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
uv run python src/final_eval.py                  # retrieval + dim sweep + report
```

Artifacts: `results/EVALUATION_REPORT.md`, `results/rag_eval/final_eval.json`,
`results/rag_eval/dim_sweep.json`, optional `baseline_v1.json` for paired tests.
