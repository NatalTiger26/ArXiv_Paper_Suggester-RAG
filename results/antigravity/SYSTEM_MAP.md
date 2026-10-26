# System Map: ArXiv_Paper_Suggester / Literature Discovery RAG

**Project:** Literature Discovery with Multi-Signal Ranking and Citation-Grounded RAG  
**Conference / Track:** ConfAI 2026 (Plaksha University) · AI in Practice  
**Date:** October 2026  
**Author:** Senior ML Systems & Research Engineer (Antigravity Mission)

---

## 1. Architectural Overview & Dual-Index Design

The project bridges two fundamental gaps in scientific literature exploration:
1. **Discovery:** Traditional keyword search ignores user research narrative and multi-signal priors (recency, citation volume, author continuity).
2. **Grounding:** Chat assistants hallucinate or fail to point to specific verifiable passages.

The system enforces a **deliberate two-index separation**:

```
                       arXiv API / Input IDs
                                │
                                ▼
                       ┌───────────────────┐
                       │  corpus_builder   │──► data/corpus.json
                       │                   │──► data/chroma_corpus (Recommend Index)
                       └─────────┬─────────┘
                                 │
                         User Research Interest
                                 ▼
                       ┌───────────────────┐
                       │ ranking/recommend │──► Top-k Ranked Papers (inspectable weights)
                       └─────────┬─────────┘
                                 │ Selective Ingest (User choice)
                                 ▼
                       ┌───────────────────┐
                       │ PDF Download      │──► data/papers/*.pdf
                       │ pdf_to_md         │──► data/markdown/*.md
                       │ chunking          │
                       │ embed + Chroma    │──► data/chroma_rag (RAG Index)
                       └─────────┬─────────┘
                                 │
                            User Question
                                 ▼
                       ┌───────────────────┐
                       │  Retrieval Gate   │──► Cosine score check (high / partial / low / none)
                       │  Semantic Cache   │──► data/rag_cache/semantic_answers.json
                       │  Gemini / LLM     │──► Grounded Answer + [1], [2] Citations
                       └───────────────────┘
```

### The Two Vector Stores

| Dimension | Recommend Index (`data/chroma_corpus`) | RAG Index (`data/chroma_rag`) |
| :--- | :--- | :--- |
| **Collection Name** | `arxiv_corpus` | `paper_chunks` |
| **Granularity** | Title + Abstract per paper | Section-aware markdown chunks (~1,000 chars) |
| **Scale** | Library-wide (e.g. 5,716 papers) | Selective subset (5,657 ingested / 551,743 chunks in full run; 10 in curated baseline) |
| **Primary Use** | Fast candidate scoring & cosine similarity | Passages retrieved as context for Gemini Q&A |
| **Embedding Model** | `google/embeddinggemma-2` (768-d) or `all-MiniLM-L6-v2` | Must match Recommend Index embedding model and dimension |

> **Critical constraint:** Chroma collections are strictly bound to the embedding model and dimension. Vectors from different models (or truncated Matryoshka dimensions) cannot be mixed.

---

## 2. Inventory of Entry Points

### 2.1 CLI Modules
- **`src/corpus_builder.py`**:
  - `search "<query>" -n 20 [--category cs.LG] [--build-index]`
  - `from-ids 1806.07572 2501.16496 [--build-index]`
  - `status`: reports corpus size and Chroma index status.
  - `rebuild-index`: re-embeds all abstracts in `data/corpus.json`.
- **`src/recommend.py`**:
  - Simple CLI wrapper around `src/ranking.py` (`score_and_rank`).
- **`src/ranking.py`**:
  - Multi-signal scoring engine combining:
    - $\text{Embedding similarity}$ (default weight 0.50)
    - $\text{Recency score}$ exponential decay $2^{-\Delta t / \text{half\_life}}$ (default weight 0.20, half-life 4.0 years)
    - $\text{Citation count}$ $\log(1 + \text{citations}) / \log(1 + \text{max\_citations})$ (default weight 0.25)
    - $\text{Author continuity}$ frequency score (default weight 0.05)
    - Optional LLM relevance judge (default weight 0.10 when enabled)
  - Inspectable presets: Balanced, Semantic-heavy, Frontier, Citation-focused, Custom.
- **`src/rag/cli.py`** (`python -m src.rag.cli`):
  - `status`: index chunk count, paper count, store directory.
  - `papers`: lists ingested papers and titles.
  - `ingest <ids...>`: `--from-recommend`, `--from-ranking`, `--force`.
  - `ask "<question>"`: `-k 6`, `--paper-ids`, `--verbose`, `--no-cache`, `--export`.
  - `eval`: runs `src/rag/evaluate.py`.
  - `reset-index`: resets `chroma_rag`.
- **`src/final_eval.py`**:
  - Official unified evaluation harness. Runs 100 gold questions (`gold_questions.json`), calculates MRR, Recall@k, Hit@1, Refusal Confusion/Precision/Recall, Wilson CIs, Paired McNemar/Bootstrap tests vs `baseline_v1.json`, runs abstract dimension sweep, and renders `results/EVALUATION_REPORT.md`.

### 2.2 Web Application
- **`src/app.py`** (Streamlit):
  - Tab 1: **Home** (Workflow overview, stats counters, quick 1-click demo).
  - Tab 2: **Library** (Keyword / arXiv ID search, paper table, manual add, rebuild index).
  - Tab 3: **Recommend** (Interest text query, weight sliders & presets, interactive ranking, "Ingest for Q&A" action buttons).
  - Tab 4: **Ask** (Search/filter papers to scope retrieval, passages slider $k$, semantic cache toggle, question form, cited markdown answer block with verification chips and expandable passage snippets).
  - Tab 5: **Help** (Architecture explanation, presets guide, limits & FAQ).

### 2.3 Auxiliary & Offline Research Modules (Out of ConfAI Headline Scope)
- **`src/citation_library/`**:
  - `cli.py`, `collect.py`, `enrich.py`, `report.py`, `store.py`.
  - Collects reference lists from Semantic Scholar / OpenAlex into `data/citation_library/graph.sqlite3`.
  - Used for citation graph analysis and hub identification. Cleanly isolated behind CLI.
- **`src/novelty/`**:
  - `cli.py`, `refs.py`, `score.py`, `metrics.py`, `ids.py`, `cache.py`.
  - Reference resolution, prior-work embedding comparison, semantic drift, and novelty scoring. Cleanly isolated behind CLI.

### 2.4 Makefile Targets
- `setup`: `uv sync` or virtualenv bootstrap.
- `health`: runs lightweight check on data paths and Chroma.
- `corpus-search Q="..." N=10`: queries arXiv and builds recommend index.
- `recommend Q="..." K=8`: multi-signal ranking.
- `rag-ingest IDS="..."`: runs full-text ingest.
- `rag-ask Q="..."`: asks citation-grounded question.
- `rag-eval`: retrieval evaluation.
- `web`: launches Streamlit application.
- `final-eval` / `final-eval-full`: runs `src/final_eval.py`.

---

## 3. Configuration & Environment Variables

| Variable | Default (Local) | Streamlit Cloud Fallback | Purpose |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(from `.env`)* | *(from Secrets)* | Primary LLM generation key for citation-grounded answers. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | `gemini-3.8-flash` | Gemini model ID. Re-checked dynamically. |
| `FALLBACK_LLM_API_KEY` | *(optional)* | *(optional)* | OpenAI-compatible endpoint key (e.g. NVIDIA NIM / Groq). |
| `FALLBACK_LLM_BASE_URL`| `https://integrate.api.nvidia.com/v1` | Same | Base URL for fallback completions. |
| `FALLBACK_LLM_MODEL`   | `meta/llama-3.1-8b-instruct` | Same | Fallback model name. |
| `EMBEDDING_MODEL`      | `google/embeddinggemma-2` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer embedding model. Auto-downgrades on Cloud to avoid 1GB RAM OOM kills. |
| `EMBEDDING_DIM`        | `0` (full, 768) | `0` (full, 384) | Matryoshka dimension truncation (e.g. 512, 256, 128). |
| `ALLOW_LARGE_EMBED`    | `0` | `0` | If `1`, permits Gemma 2 on Cloud (risks container termination). |
| `SEMANTIC_SCHOLAR_API_KEY` / `S2_API_KEY` | *(optional)* | N/A | S2 rate-limit bypass key. |
| `OPENALEX_MAILTO`      | `ajay.sharma.pg26@plaksha.edu.in` | N/A | OpenAlex polite pool identification. |
| `RAG_SEMANTIC_CACHE`   | `1` | `1` | Enable/disable answer caching. |
| `RAG_CACHE_THRESHOLD`  | `0.92` | `0.92` | Cosine similarity threshold for question cache hit. |
| `RAG_CHUNK_SIZE`       | `1000` | `1000` | Target characters per Markdown chunk. |
| `RAG_CHUNK_OVERLAP`    | `180` | `180` | Overlap characters between chunks. |

---

## 4. Evaluation Artifacts & ConfAI Analysis

The evaluation framework consists of two main sets of artifacts:

### 4.1 `results/rag_eval/`
- **`gold_questions.json`**: 100 benchmark items with categories (`definition`, `method`, `paraphrase`, `comparison`, `open_problem`, `identifier`, `out_of_scope`), difficulty levels, `expected_paper_ids`, and `should_refuse` flags (20 questions have `should_refuse: true`).
- **`final_eval.json`**: Machine-readable JSON output of the latest evaluation run.
- **`last_eval.json` / `last_eval.md`**: Legacy report outputs.
- **`baseline_v1.json`**: Curated reference baseline for paired statistical tests.
- **`dim_sweep.json`**: Dimensions sweep outputs.
- **`archive_2026-10-24/`**: Archived evaluation run from previous benchmark sweep.
- **`post_ingest_2026-10-24/`**: Evaluation after full library ingest.

### 4.2 `results/confai_analysis/`
Scripts and data created to analyze scaling and discrepancies:
- **`A_presence_check.py` / `A_presence.md`**: Confirms that all 10 gold expected papers are present in the 550k-chunk RAG index (89 to 281 chunks each).
- **`B_index_size_sweep.py` / `B_index_size_sweep.md`**: Evaluates retrieval across index sizes (10, 100, 1000, full 5,663) using Chroma's `$in` metadata filter. Demonstrates that MRR declines smoothly from ~0.78 (size 10) to ~0.46 (full library) as distractor papers increase.
- **`C_top1_mismatch_sample.py` / `C_top1_mismatch_sample.md`**: Qualitatively audits 20 cases where rank-1 paper in the full library was not the labelled gold paper; discovers that rank-1 papers are often legitimate semantic equivalents/extensions in the same domain.
- **`D_oos_full_answers.py` / `D_oos_full_answers.json`**: Captures raw generation outputs on 20 out-of-scope questions.
- **`F_refusal_discrepancy.md`**: Resolves the discrepancy between reported refusal metrics (0.45/0.45 in curated 10-paper snapshot vs 0.31/0.50 in full-library run vs 0.39/0.60 in alternate row audit).
- **`G_full_gen_eval.py` / `G_full_gen_rows.json`**: Full-answer dump of 100 gold questions on the complete index, recording both official refusal substring hits and broader heuristic hits.

---

## 5. Known Pain Points & Root Causes

1. **Streamlit Import Warning Spam in Pure CLI:**
   - *Cause:* `src/settings.py` top-level executed `import streamlit as st` and `_apply_streamlit_secrets()`. Streamlit emitted CORS and websocket configuration warnings into stdout/stderr whenever *any* CLI tool or script imported `settings.py`.
2. **Double/Triple Query Embeddings on RAG Ask:**
   - *Cause:* In `src/rag/rag_pipeline.py`, `ask()` called `lookup()` (which embedded the query), then on miss called `query_chunks()` (which embedded the query again), and finally on store called `store()` (which embedded the query a third time).
3. **Redundant Gemini Client Instantiation:**
   - *Cause:* `_call_gemini()` created a new `genai.Client(api_key=api_key)` on every invocation rather than reusing a singleton client.
4. **Dim Sweep Oddity (Abstract MRR ~0.05):**
   - *Cause:* `run_dim_sweep()` in `final_eval.py` and `eval_suite.py` attempted zero-shot retrieval of RAG passage-level questions against 5,716 *abstracts* in `corpus.json` instead of RAG full-text chunks. Because specific questions (e.g. asking for specific theorems, formulas, or ablation percentages in Section 4) do not appear in short abstracts, the MRR collapsed to ~0.05 regardless of dimension.
5. **Path Resolution in `confai_analysis/scripts`:**
   - *Cause:* Scripts were moved from `results/confai_analysis/` to `results/confai_analysis/scripts/` without updating `parents[2]` to `parents[3]`, causing `ModuleNotFoundError: No module named 'rag'`.
6. **Status Command "Ingested papers: (none yet)" False Negative:**
   - *Cause:* `src/rag/cli.py` called `index_stats()` without `include_papers=True`. While chunk count was reported accurately (551,743), the paper list returned empty, causing CLI users to mistakenly believe no papers were ingested.
7. **External API 429 Rate Limits and Uncached HTTP Calls:**
   - *Cause:* `src/novelty/refs.py` made un-cached HTTP calls to Semantic Scholar and OpenAlex without a local disk/memory cache. Repeated executions re-requested identical papers and triggered rate limits.
8. **Eval Artifact Overwrite Risk:**
   - *Cause:* `final_eval.py` unconditionally overwrote `final_eval.json` and `last_eval.json` on every run, while leaving `last_eval.md` stale, creating confusion about which numbers belonged to which run.
