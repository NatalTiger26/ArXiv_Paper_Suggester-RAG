# Evaluation Report: ArXiv Paper Suggester

_Generated: 2026-10-30T10:59:15.262728+00:00_  ·  Single harness: `src/final_eval.py`

## 1. What the system does

A researcher builds a paper library from arXiv, ranks papers for an interest, ingests full text, and asks questions answered only from those papers with citations. Embeddings: local EmbeddingGemma 2. Generation: Gemini (optional NVIDIA failsafe).

## 2. How to read the numbers (AIP concepts)

| Concept | Meaning here |
|---------|--------------|
| **Recall@k** | Fraction of labelled questions where *any* expected paper appears in the top-k retrieved papers (Lab 3). |
| **MRR** | Mean reciprocal rank of the first expected paper, rewards putting the right paper higher (Lab 3). Prefer over saturated hit@5. |
| **Hit@1** | Whether the top-1 paper is expected. Harder than Recall@k. |
| **95% bootstrap CI** | Uncertainty on the mean over this gold set (Lab 2). Overlapping CIs ⇒ do not claim a difference. |
| **Wilson CI** | Uncertainty on a proportion (refusal, citation validity). |
| **McNemar / paired bootstrap** | Whether v2 beats v1 on the *same* items (Lab 2). Requires `baseline_v1.json`. |
| **Refusal precision / recall** | Of declines, how many should have been declined, and of items that should be declined, how many were (Lab 4). **Both** required. |
| **Citation validity** | Every `[n]` in the answer refers to a supplied source, code-checkable (Lab 4). |
| **Failure hints** | Coarse Lab 5 taxonomy: missing_or_ranking, ranking, false_answer, over_refuse. |
| **Cohen’s κ** | Agreement between human and LLM judge on faithfulness/correctness. **Not computed here** unless you add human labels + a judge (Lab 4). Retrieval metrics do not need κ. |

## 3. Headline results

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| Mean MRR | **0.5873** | (0.4933, 0.6792) |
| Mean Recall@k | **0.7625** | (0.6625, 0.85) |
| Mean Hit@1 | **0.4625** | (0.35, 0.575) |
| Mean top score | 0.8161 | - |
| Index | 5657 papers / 553516 chunks | - |
| Embedder | google/embeddinggemma-2 @ dim=None | - |
| Latency p50 / p95 | 227.6 / 437.1 ms | - |
| Cache hit rate | 0.0 | (generation runs) |

**Interpretation:** On 80 questions with expected papers, MRR=0.5873 means that on average the first relevant paper sits near rank ~1.7 when found. Recall=0.7625 means about 76% of those questions surface at least one expected paper in the top-k. The CI width reflects n≈80: small improvements inside the CI should not be treated as proven gains (Lab 2).

_Generation metrics skipped (`--retrieval-only`). Re-run with `--with-generation` (and optionally `--no-cache`)._

## 4. Stratification (Lab 3)

### By kind

| Kind | n | MRR | Recall@k | Hit@1 | MRR CI |
|------|---|-----|----------|-------|--------|
| comparison | 5 | 0.3667 | 0.6 | 0.2 | (0.0667, 0.7) |
| conceptual | 1 | 1.0 | 1.0 | 1.0 | (1.0, 1.0) |
| definition | 12 | 0.5167 | 0.75 | 0.3333 | (0.2917, 0.725) |
| factual | 2 | 1.0 | 1.0 | 1.0 | (1.0, 1.0) |
| identifier | 10 | 1.0 | 1.0 | 1.0 | (1.0, 1.0) |
| method | 14 | 0.4881 | 0.6429 | 0.3571 | (0.2857, 0.7143) |
| multi_hop | 1 | 0.5 | 1.0 | 0.0 | (0.5, 0.5) |
| open_problem | 13 | 0.4679 | 0.6923 | 0.3077 | (0.2564, 0.6667) |
| paraphrase | 18 | 0.4741 | 0.7222 | 0.3333 | (0.2963, 0.6574) |
| technical | 3 | 1.0 | 1.0 | 1.0 | (1.0, 1.0) |
| theory | 1 | 1.0 | 1.0 | 1.0 | (1.0, 1.0) |

### By difficulty

| Difficulty | n | MRR | Recall@k | Hit@1 |
|------------|---|-----|----------|-------|
| easy | 14 | 0.7643 | 0.8571 | 0.7143 |
| hard | 26 | 0.5096 | 0.7308 | 0.3077 |
| medium | 40 | 0.5758 | 0.75 | 0.475 |

**Interpretation:** Identifier and comparison kinds are typically harder for dense-only retrieval (Lab 3 Q44-style exact IDs favor lexical matching). Open-problem and paraphrase rows usually lead when the supporting papers are ingested. Medium difficulty can look worse than hard if medium items reference papers still weakly ranked.

## 5. Failures and Pareto (Lab 5)

Failure hint counts: `{'ranking': 24, 'missing_or_ranking': 19}`

| arXiv ID (expected but often not in top-k) | # questions |
|---------------------------------------------|------------:|
| 2501.16496 | 12 |
| 2304.14997 | 11 |
| 1806.07572 | 4 |
| 2407.13594 | 3 |
| 2404.14082 | 2 |
| 2504.13151 | 1 |

**Interpretation:** `missing_or_ranking` means the expected paper never appeared in top-k: either it is not ingested, or the embedder ranked other papers higher. If the PDF is already in the index, the fix is ranking (hybrid BM25, rerank, query rewrite), not another download. `ranking` means the paper was in top-k but not rank 1. `false_answer` / `over_refuse` are generation-side (Lab 4 operating point).

## 6. Calibration

### Coverage label vs actual Recall@k

| Coverage | n | Mean Recall@k |
|----------|---|---------------|
| high | 79 | 0.7595 |
| partial | 1 | 1.0 |

### Top-score quintiles vs hit rate

| Quintile | Score range | n | Hit@1 | Recall |
|----------|-------------|---|-------|--------|
| 1 | 0.6984–0.8195 | 16 | 0.625 | 0.75 |
| 2 | 0.8206–0.8374 | 16 | 0.25 | 0.75 |
| 3 | 0.8377–0.8486 | 16 | 0.4375 | 0.625 |
| 4 | 0.8486–0.8643 | 16 | 0.5 | 0.8125 |
| 5 | 0.8656–0.896 | 16 | 0.5 | 0.875 |

**Interpretation:** If higher score quintiles show higher hit@1, the similarity score is at least ordinally useful as a confidence signal. If not, do not threshold scores as if they were calibrated probabilities.

## 7. Embedding dimension sweep (Matryoshka)

_Dim sweep not run._

## 8. Paired comparison vs baseline (Lab 2)

Baseline: `results/rag_eval/baseline_v1.json` (n=73)

| Test | Δ (v2 − v1) | 95% CI | p-value |
|------|-------------|---------|---------|
| Paired bootstrap Recall@k | -0.0959 | (-0.2192, 0.0274) | 0.156 |
| Paired bootstrap MRR | -0.1783 | (-0.2943, -0.0605) | 0.006 |
| McNemar (recall success) | b=14 c=7 | - | 0.189247 |

**Interpretation:** McNemar uses only discordant items (v1 right/v2 wrong vs the reverse). Small p (e.g. < 0.05) supports a real improvement on this gold set. Large p ⇒ choose on cost/latency instead of claiming accuracy gains.

## 9. SLO / regression gate (Lab 7)

```json
{
  "gates": {
    "mean_mrr_min": 0.55,
    "mean_recall_min": 0.55,
    "refusal_precision_min": 0.5,
    "citation_validity_min": 0.95,
    "p95_latency_ms_max": 500
  },
  "mrr_pass": true,
  "recall_pass": true,
  "latency_pass": true,
  "all_pass": true
}
```

Gates are project-specific thresholds, not universal truths. Failing refusal precision while passing retrieval is a typical pattern: retrieval can be demo-ready while refusal is not deployment-ready.

## 10. What is not safe / not claimed

- Clinical, legal, or financial decisions without human review.
- Questions about papers outside the RAG index.
- Treating cosine similarity as a calibrated probability.
- Faithfulness/correctness **LLM-judge scores** or **Cohen’s κ** (not run, would need human labels).
- Gold-context decomposition (oracle chunks vs retrieved), not run. Residual error is only coarsely attributed via fail hints.
- Lab 6 attack block rates, out of scope unless tool-use is productized.

## 11. Executive recommendation

With 5657 papers indexed, MRR=0.5873 (CI (0.4933, 0.6792)) and Recall@k=0.7625 on 80 labelled items. Residual fail hints: {'ranking': 24, 'missing_or_ranking': 19}. Most frequent top-k miss: 2501.16496 (n=12). Treat as ranking if the PDF is already ingested. Suitable for a constrained portfolio demo on the ingested MI/NTK library, not for unsupervised refusal-critical deployment until OOS precision is stronger.

## 12. Reproducibility

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
uv run python src/final_eval.py                  # retrieval + dim sweep + report
```

Artifacts: `results/EVALUATION_REPORT.md`, `results/rag_eval/final_eval.json`,
`results/rag_eval/dim_sweep.json`, optional `baseline_v1.json` for paired tests.
