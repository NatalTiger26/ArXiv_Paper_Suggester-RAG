# ConfAI 2026 Poster: Complete Data & Metrics Source Trail

This document provides a line-by-line audit trail for every numerical value, hyperparameter, equation, and empirical finding displayed on the **ConfAI 2026 academic poster**.

---

## 1. Corpus Scale & Storage Metrics

| Poster Metric | Displayed Value | Repository Source File | Exact Key / Query |
| :--- | :--- | :--- | :--- |
| **Catalog Papers** | **5,752** | `data/catalog.sqlite3` | `SELECT count(*) FROM papers` |
| **Citation Graph Papers** | **101,716** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM papers` |
| **Citation Graph Edges** | **226,857** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM edges` |
| **Full-Text RAG Chunks** | **553,516** | `data/chroma_rag/` | Collection `paper_chunks` count |
| **Evaluated Batch Novelty Papers** | **105 papers** | `results/batch_novelty_report.json` | `stats.total_successful` |

---

## 2. Multi-Signal Ranking Parameters

| Parameter | Value | Source File | Code Reference |
| :--- | :--- | :--- | :--- |
| **Default Weights** | `0.50, 0.20, 0.25, 0.05` | `src/ranking.py` | `DEFAULT_WEIGHTS = {"embedding": 0.50, "recency": 0.20, "citation": 0.25, "author": 0.05}` |
| **Recency Half-Life** | **4.0 years** | `src/ranking.py` | `DEFAULT_HALF_LIFE = 4.0` |
| **Candidate Retrieval Pool** | **$3 \times k$ (min 20)** | `src/ranking.py` | `DEFAULT_POOL_MULTIPLIER = 3, MIN_POOL_SIZE = 20` |
| **Preset: Balanced** | `0.50, 0.20, 0.25, 0.05` | `src/app.py` | `WEIGHT_PRESETS["Balanced (default)"]` |
| **Preset: Precision** | `0.65, 0.10, 0.20, 0.05` | `src/app.py` | `WEIGHT_PRESETS["Precision (embedding ↑)"]` |
| **Preset: Prefer Recent** | `0.45, 0.35, 0.15, 0.05` | `src/app.py` | `WEIGHT_PRESETS["Prefer recent"]` |
| **Preset: Citation Heavy** | `0.35, 0.10, 0.50, 0.05` | `src/app.py` | `WEIGHT_PRESETS["Citation heavy"]` |

---

## 3. RAG Architecture & Ingestion Hyperparameters

| Parameter | Value | Source File | Code Reference |
| :--- | :--- | :--- | :--- |
| **Local Embedding Model** | `google/embeddinggemma-2` | `src/settings.py` | `_DEFAULT_LOCAL_EMBED = "google/embeddinggemma-2"` |
| **Embedding Dimension** | **768** (Matryoshka cap 0) | `src/settings.py` | `EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0"))` |
| **Cloud Fallback Embedder** | `all-MiniLM-L6-v2` (384-dim) | `src/settings.py` | `_DEFAULT_CLOUD_EMBED = "sentence-transformers/all-MiniLM-L6-v2"` |
| **Target Chunk Size** | **1,000 characters** | `src/settings.py` | `CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))` |
| **Chunk Overlap** | **180 characters** | `src/settings.py` | `CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))` |
| **Minimum Chunk Length** | **80 characters** | `src/settings.py` | `MIN_CHUNK_CHARS = 80` |
| **Context Window Retrieval** | **Top-6 chunks** | `src/settings.py` | `DEFAULT_TOP_K = 6` |
| **Primary Synthesis Model** | `gemini-2.0-flash` / `gemini-3.8-flash` | `src/settings.py` | `get_gemini_model()` |
| **Failsafe Synthesis Model** | `meta/llama-3.1-8b-instruct` | `src/settings.py` | `FALLBACK_LLM_MODEL` via NVIDIA NIM |

---

## 4. ConfAI 2026 Empirical Evaluation & Scaling Sweeps

### A. Index Size Scaling Sweep (`results/confai_analysis/B_index_size_sweep.json`)
Evaluated across 100 benchmark questions (80 answerable + 20 out-of-scope refusals):

| Index Scale | Papers ($N$) | Chunks | Mean MRR | 95% Bootstrap CI | Recall@k | Hit@1 | Source Row |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Curated Subspace** | **10** | ~1,725 | **0.786** | [0.702, 0.863] | **0.888** | **0.725** | `size_label=10, seed=0` |
| **Topic Neighborhood** | **100** | ~9,800 | **0.757** | [0.658, 0.842] | **0.825** | **0.713** | `size_label=100, seed=0` |
| **Domain Corpus** | **1,000** | ~98,000 | **0.638** | [0.546, 0.740] | **0.725** | **0.588** | `size_label=1000, seed=2` |
| **Full Uncurated Library** | **5,663** | 551,743 | **0.457** | [0.360, 0.549] | **0.638** | **0.350** | `size_label=full, seed=0` |

### B. Retrieval Precision & Generation Grounding (`results/confai_analysis/G_full_gen_summary.json`)
Full-scale generation run over 100 questions:

| Evaluation Metric | Value | Source Key | Meaning |
| :--- | :--- | :--- | :--- |
| **Passage Citation Validity** | **98.9%** | `citation.cited_valid = 89`, `citation.cited_invalid = 1` | 89 of 90 cited passage links accurately map to retrieved context |
| **Answerable Refusal Rate** | **23.8%** (19 / 80) | `refusal_official.FP = 19` | Conservative refusal behavior on borderline evidence |
| **Out-of-Scope Recall** | **65.0%** (13 / 20) | `refusal_official.TP = 13` | True rejection rate when requested information is absent |
| **Refusal Precision** | **40.6%** | `refusal_official.precision = 0.40625` | Indicates open calibration challenge for safety gating |
| **Median Generation Latency** | **9.85 s** | `latency_p50_ms = 9854.5` | End-to-end cloud synthesis latency |

### C. Dense Retrieval vs. Hybrid RRF Reranking (`results/rag_eval/last_eval.md`)
| Protocol | MRR | Recall@k | Hit@1 | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only (`google/embeddinggemma-2`)** | **0.574** | 0.688 | 0.488 | Baseline dense retrieval |
| **Hybrid RRF + Section Priors** | **0.612** | **0.738** | **0.538** | **+6.6% MRR**, resolves exact-ID and method queries |

---

## 5. 2D Reference Cone Geometry Metrics (`src/novelty/metrics.py`)

| Metric | Formula | Mathematical Significance |
| :--- | :--- | :--- |
| **Cone Explained Energy** | $R^2_+ = \|\hat{\mathbf{x}}\|^2$ | Fraction of paper embedding variance spanned by non-negative combination of cited works |
| **Closest Parent Cosine** | $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$ | Cosine similarity to the closest single foundational predecessor |
| **Synergy Blend Gain** | $\Delta = R^2_+ - \max(m, 0)^2$ | Non-linear variance gain unlocked by synthesizing multiple distinct ancestral lineages |
| **Orthogonal Residual Gap** | $\text{Res} = \sqrt{\max(0, 1 - R^2_+)}$ | Euclidean departure distance outside the prior-art cone to the unit hypersphere |
| **Orthonormal Canvas Basis** | $\mathbf{e}_1 = \mathbf{y}_{m^*}, \mathbf{e}_2 = \mathbf{u} / \|\mathbf{u}\|$ | Gram-Schmidt coordinate system where closest ancestor sits at $(1, 0)$ |
| **Exemplar: BIG-bench (2206.04615)** | $m=0.774, \Delta=0.138, R^2_+=0.737$ | 80 references extracted via arXiv HTML, with BERT identified as top parent (24.9% share) |
