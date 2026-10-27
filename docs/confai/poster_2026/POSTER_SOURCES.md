# ConfAI 2026 Academic Poster: Empirical Data and Metrics Audit Trail

This document provides a line-by-line audit trail for every numerical value, hyperparameter, equation, and empirical finding displayed on the ConfAI 2026 academic poster.

---

## 1. Corpus Scale & Storage Metrics

| Poster Metric | Displayed Value | Repository Source File | Exact Key / Query |
| :--- | :--- | :--- | :--- |
| **Catalog Papers** | **5,752** | `data/catalog.sqlite3` | `SELECT count(*) FROM papers` |
| **Citation Graph Papers** | **101,716** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM papers` |
| **Citation Graph Edges** | **226,857** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM edges` |
| **Full-Text RAG Chunks** | **553,516** | `data/chroma_rag/` | Collection `paper_chunks` count |
| **Evaluated Batch Novelty Cohort** | **2,159 papers** | `results/batch_novelty_report.json` | `stats.total_successful` |

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
| **Target Chunk Overlap** | **200 characters** | `src/settings.py` | `CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))` |
| **RRF Constant $k$** | **60** | `src/rag/retrieval.py` | `k: int = 60` |
| **RRF Dense Weight $\alpha$** | **0.50** | `src/rag/retrieval.py` | `alpha: float = 0.5` |
| **Top Passages Retrieved ($k$)** | **6 passages** | `results/rag_eval/final_eval.json` | `recall@k` evaluation threshold |

---

## 4. Distractor Scaling Law Curve (Pure Dense Retrieval)

Source: `results/confai_analysis/B_index_size_sweep.json`  
Protocol: Pure dense retrieval (`query_chunks`) across four library sizes using five random distractor seeds at $N \in \{10, 100, 1000\}$ and three full runs at $N=5,663$.

| Index Size $N$ | Mean MRR | 95% Bootstrap CI | Mean Hit@1 | Mean Recall@6 |
| :--- | :--- | :--- | :--- | :--- |
| **$N = 10$** (Curated) | **0.7865** | [0.6979, 0.8625] | **0.7250** | **0.8875** |
| **$N = 100$** | **0.7526** | [0.6625, 0.8354] | **0.7000** | **0.8333** |
| **$N = 1,000$** | **0.6382** | [0.5708, 0.7729] | **0.5750** | **0.7458** |
| **$N = 5,663$** (Full Library) | **0.4573** | [0.3542, 0.5625] | **0.3500** | **0.6375** |

**Net Degradation from Curated to Full Scale:**
- MRR Drop: $0.7865 \rightarrow 0.4573$ (**-41.9%**)
- Hit@1 Drop: $0.7250 \rightarrow 0.3500$ (**-51.7%**)
- Recall@6 Drop: $0.8875 \rightarrow 0.6375$ (**-28.2%**)

---

## 5. Dense vs Hybrid RRF Retrieval (Paired Evaluation)

Sources:
- Pure Dense Baseline: `results/rag_eval/archive_2026-10-24_1/final_eval.json` and `results/confai_analysis/B_index_size_sweep.json`
- Hybrid BM25 + Dense RRF: `results/rag_eval/final_eval.json`

Evaluation Set: Exactly 80 answerable gold questions evaluated against the full index.

| Retrieval Mode | Mean MRR | Mean Hit@1 | Mean Recall@6 | P95 Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Chroma Only** | **0.4687** | **0.3500** | **0.6375** | **184.2 ms** |
| **Hybrid BM25 + Dense RRF** | **0.5739** | **0.4875** | **0.6875** | **502.1 ms** |

**Paired Bootstrap Differences (10,000 resamples):**
- **MRR Difference:** $\mathbf{+0.1052}$ (95% CI: $[+0.0177, +0.1969]$, statistically significant at $\alpha = 0.05$)
- **Hit@1 Difference:** $\mathbf{+0.1375}$ (95% CI: $[+0.0375, +0.2500]$)
- **Recall@6 Difference:** $\mathbf{+0.0500}$ (95% CI: $[-0.0625, +0.1625]$)

**Recovered Fraction of Lost MRR:**
- Mean point estimate: $(0.5739 - 0.4573) / (0.7865 - 0.4573) = \mathbf{35.4\%}$ (or $33.1\%$ relative to dense baseline 0.4687)
- Bootstrap uncertainty range: **6.0% to 61.1%**

---

## 6. Attribution Integrity & Calibration Facts

Source: `results/confai_analysis/final_eval.json` and `results/rag_eval/last_eval.md`

| Metric | Measured Proportion | Wilson 95% Confidence Interval |
| :--- | :--- | :--- |
| **Citation Validity** | **98.89%** ($89 / 90$ claims verifiable) | **[93.97%, 99.80%]** |
| **Refusal Recall** | **65.00%** ($13 / 20$ unanswerable queries declined) | **[43.29%, 81.88%]** |
| **Refusal Precision** | **40.62%** ($13 / 32$ total refusals were correct) | **[25.53%, 57.73%]** |
| **False Refusal Rate** | **23.75%** ($19 / 80$ answerable queries declined) | **[15.82%, 34.00%]** |

---

## 7. 2D Novelty Cone Cohort & Exemplars

Sources: `results/batch_novelty_report.json` and `data/citation_library/graph.sqlite3`

**Cohort Distribution (2,159 valid papers):**
- High Synergy ($\Delta \ge 0.05, R^2_+ \ge 0.50$): **1,936 papers** (89.67%)
- Moderate Synthesis ($\Delta \ge 0.02$): **208 papers** (9.63%)
- Linear Extension ($R^2_+ \ge 0.70, \Delta < 0.02$): **13 papers** (0.60%)
- Paradigm Shift ($R^2_+ < 0.30, m < 0.50$): **2 papers** (0.09%)

**Exemplar Papers:**
- **Primary Exemplar (Full Coverage): Grad-CAM++ (1710.11063)**
  - Resolved embedded references: **18 of 18** (100.0% coverage)
  - Dominant parent: Grad-CAM (1610.02391), share: **49.7%**
  - Metrics: $m = 0.938$, $\Delta = 0.047$, $R^2_+ = 0.927$, $\text{Res} = 0.270$
- **High Synergy Exemplar: BIG-bench (2206.04615)**
  - Resolved embedded references: **80 of 706** raw HTML references
  - Dominant parent: BERT (1810.04805), share: **24.7%**
  - Metrics: $m = 0.774$, $\Delta = 0.138$, $R^2_+ = 0.737$, $\text{Res} = 0.513$

**Isotropic Null Control vs Real Citation Cones:**
- Isotropic Null Control (Random references, $k=35$ in 768d):
  - Mean $\Delta_{\text{null}} = \mathbf{0.0171}$, Median: $0.0161$, 95% CI: $[0.0065, 0.0304]$
- Real Citation Cones (Cohort mean):
  - Mean $\Delta_{\text{real}} = \mathbf{0.0849}$
- **Synergy Enhancement Ratio:** $0.0849 / 0.0171 = \mathbf{5.0\times}$

---

## 8. Interactive Verification URLs

- **GitHub Repository:** `https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG` (HTTP 200 OK)
- **Live Interactive Demo:** `https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/` (HTTP 303 Redirect to authenticated Streamlit session, HTTP 200 active)
