# ConfAI 2026 Academic Poster: Comprehensive Empirical Data Reference and Audit Trail

**Event:** ConfAI 2026 — AI in Practice, Extended Track (Poster Exhibition, Plaksha University)  
**Author:** Ajay Sharma (M.S. in AI, Plaksha University)  
**Supervisor:** Dr. Subrat Panda  
**Format:** ISO 216 A1 Portrait (594 mm × 841 mm)  
**Repository URLs:**  
- GitHub: `https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG`  
- Live Demo: `https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/`

This document serves as the exhaustive, single source of truth for every empirical claim, numerical value, equation, hyperparameter, and figure presented on the ConfAI 2026 academic poster. All values are deterministically derived from tracked experimental logs and SQLite/Chroma databases in this repository.

---

## 1. Corpus Scale, Citation Graph, and Storage Metrics

| Metric | Measured Value | Repository Source File | Query / Verification Command |
| :--- | :--- | :--- | :--- |
| **Catalog Papers** | **5,752 papers** | `data/catalog.sqlite3` | `SELECT count(*) FROM papers` |
| **Catalog FTS5 Index** | **5,752 entries** | `data/catalog.sqlite3` | `SELECT count(*) FROM papers_fts` |
| **Citation Graph Nodes** | **101,716 papers** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM papers` |
| **Citation Graph Edges** | **226,857 directed citations** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM edges` |
| **Citation Fetch Status Records** | **5,716 records** | `data/citation_library/graph.sqlite3` | `SELECT count(*) FROM fetch_status` |
| **Full-Text Passage Chunks** | **553,516 passages** | `data/chroma_rag/` | Collection `paper_chunks` count |
| **Abstract Corpus Items** | **5,716 items** | `data/chroma_corpus/` | Collection `arxiv_corpus` count |
| **Ingested Markdown Files** | **6,341 documents** | `data/markdown/` | Filesystem count of `.md` full-text files |
| **Novelty Cache Size** | **763.8 MB** | `data/novelty/novelty_cache.sqlite3` | SQLite database file size |
| **Novelty Cached Embeddings** | **114,365 768d vectors** | `data/novelty/novelty_cache.sqlite3` | `SELECT count(*) FROM embeddings` |
| **Novelty Reference Edges** | **240,487 edges** | `data/novelty/novelty_cache.sqlite3` | `SELECT count(*) FROM ref_edges` |
| **Novelty Scored Profiles** | **5,711 profiles** | `data/novelty/novelty_cache.sqlite3` | `SELECT count(*) FROM scores` |
| **Evaluated Batch Novelty Cohort** | **5,707 successful papers** | `results/batch_novelty_report.json` | Total evaluated: 5,752, Success rate: 99.22% |

---

## 2. Batch Novelty Subspace Analysis Cohort (N = 5,707 Papers)

### 2.1 Overview and Success Rate
- **Total Papers Evaluated:** 5,752 papers
- **Successfully Scored:** 5,707 papers (99.22%)
- **Failed Papers:** 45 papers (0.78%)
  - Too few reference embeddings ($< 3$ refs): 17 papers
  - No references found in catalog/graph: 28 papers

### 2.2 Complete Descriptive Statistics Table
All papers are evaluated in fixed 768-dimensional space (Google EmbeddingGemma-2) against the non-negative convex cone of their cited references:
$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

| Metric | Definition | Mean | Std Dev | Median (P50) | Min | P01 | P05 | P10 | P25 | P75 | P90 | P95 | P99 | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anchor ($m$)** | $\max_j (\mathbf{x}^T \mathbf{y}_j)$ | **0.8691** | 0.0407 | **0.8726** | 0.6484 | 0.7255 | 0.8007 | 0.8195 | 0.8478 | 0.8966 | 0.9148 | 0.9261 | 0.9449 | 1.0000 |
| **Synergy ($\Delta$)** | $R^2_+ - \max(m, 0)^2$ | **0.0877** | 0.0276 | **0.0880** | 0.0000 | 0.0258 | 0.0430 | 0.0516 | 0.0686 | 0.1068 | 0.1229 | 0.1326 | 0.1502 | 0.2117 |
| **Cone Energy ($R^2_+$)** | $\|\hat{\mathbf{x}}\|^2$ | **0.8446** | 0.0577 | **0.8538** | 0.5074 | 0.6348 | 0.7397 | 0.7751 | 0.8178 | 0.8838 | 0.9059 | 0.9182 | 0.9377 | 1.0000 |
| **Residual Gap ($\text{Res}$)** | $\sqrt{1 - R^2_+}$ | **0.3879** | 0.0701 | **0.3823** | 0.0000 | 0.2497 | 0.2862 | 0.3068 | 0.3410 | 0.4268 | 0.4742 | 0.5102 | 0.6043 | 0.7018 |
| **Refs Used ($N_{\text{refs}}$)** | Reference embeddings | **40.53** | 19.96 | **38.00** | 3.0 | 6.0 | 12.0 | 16.0 | 25.0 | 53.0 | 73.0 | 80.0 | 80.0 | 80.0 |

### 2.3 Pairwise Correlation Matrix
Measured across all $N = 5,707$ valid papers:

| Metric Pair | Pearson $r$ | $p$-value | Interpretation |
| :--- | :---: | :---: | :--- |
| **$(m, \Delta)$** | **-0.5658** | $< 10^{-300}$ | Moderate negative correlation. High single-parent alignment leaves less residual headroom for multi-parent synergy gain. |
| **$(m, R^2_+)$** | **+0.9262** | $< 10^{-300}$ | Strong positive correlation. The closest anchor provides the dominant baseline energy for the cone. |
| **$(m, \text{Res})$** | **-0.9213** | $< 10^{-300}$ | Strong negative correlation. Papers close to an anchor have small orthogonal residuals. |
| **$(N_{\text{refs}}, R^2_+)$** | **+0.1463** | $1.15 \times 10^{-28}$ | Weak positive correlation. Cone reconstruction increases slightly with reference count, but does not dominate. |
| **$(N_{\text{refs}}, \Delta)$** | **+0.1649** | $4.69 \times 10^{-36}$ | Weak positive correlation. Having more diverse parents yields a slight increase in synergy. |
| **$(R^2_+, \text{Res})$** | **-0.9870** | $< 10^{-300}$ | Near-perfect inverse non-linear relationship conforming to $\text{Res} = \sqrt{1 - R^2_+}$. |

### 2.4 Five-Way Discovery Classification Distribution

| Classification Category | Operational Decision Rule | Paper Count | Percentage | Qualitative Architectural Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **High Synergy Cone Blend** | $\Delta \ge 0.05$ and $R^2_+ \ge 0.50$ | **5,176** | **90.70%** | Non-linearly synthesizes multiple foundational prior works. |
| **Moderate Synthesis** | $0.02 \le \Delta < 0.05$ and $R^2_+ \ge 0.40$ | **489** | **8.57%** | Combines prior references with modest multi-parent energy gain. |
| **Linear / Incremental Extension** | $R^2_+ \ge 0.70, \Delta < 0.02, m \ge 0.95$ | **34** | **0.60%** | Heavily anchored to a single ancestor with negligible synergy. |
| **Paradigm Shift / Subspace Outlier** | $R^2_+ < 0.60$ and $m < 0.75$ | **8** | **0.14%** | Large orthogonal departure distance from all cited references. |
| **Diffused Reference Combination** | Dispersed energy without dominant cone | **0** | **0.00%** | No papers fell into this unconstrained fallback bin. |
| **Total Cohort** | Successful NNLS runs | **5,707** | **100.00%** | Complete full-library evaluation. |

### 2.5 Top Synergistic Papers (Highest Multi-Parent Synergy $\Delta$)

| arXiv ID | Title | Year | Citations | $\Delta$ (Synergy) | $m$ (Anchor) | $R^2_+$ | Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **2108.07258** | *Implicit Graph Neural Networks* | 2021 | 14 | **0.2117** | 0.724 | 0.736 | High Synergy Cone Blend |
| **1810.03993** | *Model Cards for Model Reporting* | 2018 | 42 | **0.1752** | 0.734 | 0.713 | High Synergy Cone Blend |
| **2204.03772** | *Multi-objective optimization determines when...* | 2022 | 0 | **0.1730** | 0.694 | 0.655 | High Synergy Cone Blend |
| **2008.10138** | *PermuteAttack: Counterfactual Explanation...* | 2020 | 0 | **0.1719** | 0.683 | 0.639 | High Synergy Cone Blend |
| **2304.03442** | *Generative Agents: Interactive Simulacra...* | 2023 | 312 | **0.1708** | 0.844 | 0.883 | High Synergy Cone Blend |
| **2007.00251** | *Unifying Model Explainability and Robustness...* | 2020 | 0 | **0.1686** | 0.838 | 0.870 | High Synergy Cone Blend |
| **2310.10631** | *Llemma: An Open Language Model For Mathematics* | 2023 | 88 | **0.1669** | 0.829 | 0.854 | High Synergy Cone Blend |
| **2211.07982** | *Evaluating the Faithfulness of Saliency-based...* | 2022 | 0 | **0.1652** | 0.844 | 0.878 | High Synergy Cone Blend |
| **2603.27331** | *SACRED: A Faithful Annotated Multimedia Multi...* | 2026 | 0 | **0.1624** | 0.829 | 0.849 | High Synergy Cone Blend |
| **2302.14383** | *Linear Spaces of Meanings: Compositional Stru...* | 2023 | 0 | **0.1624** | 0.833 | 0.856 | High Synergy Cone Blend |

### 2.6 Top Paradigm Shifts / Subspace Outliers (Highest Residual $\text{Res}$)

| arXiv ID | Title | Year | Citations | Novelty Residual | $R^2_+$ | $m$ (Anchor) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1609.02907** | *Semi-Supervised Classification with GCNs* | 2016 | 14,210 | **0.7018** | 0.5074 | 0.6484 |
| **2501.10292** | *Enhancing AI Transparency: XRL-Based Resource Management* | 2025 | 185 | **0.6994** | 0.5108 | 0.6812 |
| **1801.06562** | *Uncertainty quantification of tabulated super...* | 2018 | 5 | **0.6930** | 0.5197 | 0.6941 |
| **2210.15678** | *Diffusion Models in Vision: A Survey* | 2022 | 820 | **0.6830** | 0.5335 | 0.6660 |
| **2505.16376** | *DeCafNet: Delegate and Conquer for Efficient...* | 2025 | 0 | **0.6796** | 0.5381 | 0.6881 |
| **2501.02355** | *CorrFill: Enhancing Faithfulness in Reference...* | 2025 | 0 | **0.6775** | 0.5410 | 0.6622 |
| **1911.11351** | *Distraction-Aware Feature Learning for Human Attribute Recognition* | 2019 | 4,120 | **0.6742** | 0.5454 | 0.6942 |
| **1001.3540** | *Local transition gradients determine protein energy landscapes* | 2010 | 9,840 | **0.6722** | 0.5481 | 0.7061 |

### 2.7 Top Foundationally Rooted Works (Highest Single Parent $m$)

| arXiv ID | Title | Year | Citations | $m$ (Anchor) | $\Delta$ (Synergy) | Dominant Parent Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1404.1530** | *Provable Deterministic Leverage Score Sa...* | 2014 | 88 | **1.0000** | 0.0000 | `openalex:W2109818719` ($\lambda=1.00$) |
| **2209.06481** | *Targeting interventions for displacement...* | 2022 | 0 | **1.0000** | 0.0000 | `openalex:W4315488832` ($\lambda=1.00$) |
| **2201.12828** | *Comprehensive Saliency Fusion for Object...* | 2022 | 0 | **1.0000** | 0.0000 | `openalex:W4205623571` ($\lambda=1.00$) |
| **1607.00133** | *Weight Normalization: A Simple Reparamet...* | 2016 | 2,140 | **1.0000** | 0.0000 | `openalex:W2473418344` ($\lambda=1.00$) |
| **2402.17242** | *Scalable Community Search with Accuracy...* | 2024 | 0 | **0.9991** | 0.0000 | `openalex:W4400909681` ($\lambda=0.95$) |
| **2504.00435** | *User authentication on earable devices...* | 2025 | 0 | **0.9885** | 0.0018 | `openalex:W4283219085` ($\lambda=0.85$) |
| **2109.05201** | *Fast Approximate Nearest Neighbor Search...* | 2021 | 12 | **0.9843** | 0.0012 | `openalex:W3097305019` ($\lambda=0.95$) |
| **2101.03164** | *An Empirical Study of Training Dynamics...* | 2021 | 8 | **0.9808** | 0.0121 | `s2:3dc12dadcf6a2...` ($\lambda=0.66$) |
| **1910.12828** | *Blind Robust 3-D Mesh Watermarking based...* | 2019 | 0 | **0.9798** | 0.0041 | `openalex:W2911342611` ($\lambda=0.78$) |
| **2408.12398** | *A Comparative Analysis of Faithfulness...* | 2024 | 0 | **0.9792** | 0.0011 | `2406.15264` ($\lambda=0.93$) |

### 2.8 Exemplar Paper Deep-Dives
1. **Primary Exemplar (Full Coverage): Grad-CAM++ (`1710.11063`)**
   - **Title:** *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks*
   - **Resolved Citations:** Exactly **18 of 18** cited references resolved in graph (100.0% coverage).
   - **Metrics:** $m = 0.9380$, $\Delta = 0.0471$, $R^2_+ = 0.9270$, $\text{Res} = 0.2703$.
   - **Classification:** Moderate Synthesis ($\Delta < 0.05$).
   - **Attribution Breakdown:**
     - Grad-CAM (`1610.02391`): $\lambda = 0.497$ (**49.7%** share)
     - CAM (`1512.04150`): $\lambda = 0.214$ (**21.4%** share)
     - Guided Backpropagation (`1412.6806`): $\lambda = 0.142$ (**14.2%** share)
     - DeconvNet (`1311.2901`): $\lambda = 0.089$ (**8.9%** share)
     - Residual references: $\lambda = 0.058$ (**5.8%** share)
2. **High Synergy Exemplar: BIG-bench (`2206.04615`)**
   - **Title:** *Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models*
   - **Resolved Citations:** **80 of 706** raw references resolved (11.3% coverage; capped at 80 reference embeddings).
   - **Metrics:** $m = 0.7737$, $\Delta = 0.1384$, $R^2_+ = 0.7370$, $\text{Res} = 0.5128$.
   - **Classification:** High Synergy Cone Blend ($\Delta = 0.1384 \gg 0.05$).
   - **Attribution Breakdown:**
     - BERT (`1810.04805`): $\lambda = 0.247$ (**24.7%** share)
     - GPT-3 (`2005.14165`): $\lambda = 0.218$ (**21.8%** share)
     - T5 (`1910.10683`): $\lambda = 0.184$ (**18.4%** share)
     - RoBERTa (`1907.11692`): $\lambda = 0.141$ (**14.1%** share)
     - Other references: $\lambda = 0.210$ (**21.0%** share)
3. **Foundational Milestone: Attention Is All You Need (`1706.03762`)**
   - **Metrics:** $m = 0.8507$, $\Delta = 0.0592$, $R^2_+ = 0.7829$, $\text{Res} = 0.4659$, $N_{\text{refs}} = 33$.
   - **Classification:** High Synergy Cone Blend.

### 2.9 Empirical Null Control vs Real Citation Cones
- **Isotropic 768-Dimensional Null Control:**
  - $N = 300$ simulated synthetic papers where target vector and $k = 35$ reference vectors are drawn uniformly from the 768-dimensional sphere.
  - Mean $m_{\text{null}} = 0.0765$
  - Mean $R^2_{+,\text{null}} = 0.0232$
  - Mean Synergy $\Delta_{\text{null}} = \mathbf{0.0171}$, Median: $0.0164$, 95% Bootstrap CI: $[0.0065, 0.0304]$.
- **Real Empirical Citation Cones (N = 5,707 Papers):**
  - Mean Anchor $m_{\text{real}} = \mathbf{0.8691}$
  - Mean Cone Energy $R^2_{+,\text{real}} = \mathbf{0.8446}$
  - Mean Synergy $\Delta_{\text{real}} = \mathbf{0.0877}$, Median: $0.0880$, 95% Bootstrap CI: $[0.0870, 0.0884]$.
- **Synergy Enhancement Ratio:**
  $$\frac{\Delta_{\text{real}}}{\Delta_{\text{null}}} = \frac{0.0877}{0.0171} = \mathbf{5.13\times}$$
- **Embedding Anisotropy Context:**
  In raw uncentered Gemma-2 embeddings, pairwise background similarity between random catalog papers averages $0.7607 \pm 0.0626$ due to standard language model cone anisotropy. Centering against the cohort mean isolates reference-specific geometric alignment.

---

## 3. RAG Retrieval Distractor Scaling Law Curve (Index Size Sweep)

### 3.1 Experimental Protocol
- **Source Artifact:** `results/confai_analysis/B_index_size_sweep.json`
- **Method:** Pure dense retrieval (`query_chunks` via ChromaDB, 768-dim Google EmbeddingGemma-2) across four index scales $N \in \{10, 100, 1000, 5663\}$.
- **Gold Question Set:** Exactly 80 answerable research questions evaluated across all conditions.
- **Seeds:** Three random distractor seeds (0, 1, 2) per scale, yielding 12 fully tracked evaluation runs.
- **Note on $N=10$:** Zero distractors present (contains exclusively the gold expected papers, hence all three seeds yield identical metric values).

### 3.2 Full 12-Run Scaling Sweep Results

| Scale Label | Target $N$ | Actual Subset Papers | Seed | Evaluated Questions | Mean MRR | MRR 95% Bootstrap CI | Mean Hit@1 | Mean Recall@6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** (Curated) | 10 | 10 | 0 | 80 | **0.7865** | [0.6979, 0.8625] | **0.7250** | **0.8875** |
| **10** (Curated) | 10 | 10 | 1 | 80 | **0.7865** | [0.7000, 0.8625] | **0.7250** | **0.8875** |
| **10** (Curated) | 10 | 10 | 2 | 80 | **0.7865** | [0.7021, 0.8625] | **0.7250** | **0.8875** |
| **100** | 100 | 100 | 0 | 80 | **0.7573** | [0.6583, 0.8417] | **0.7125** | **0.8250** |
| **100** | 100 | 100 | 1 | 80 | **0.7583** | [0.6656, 0.8417] | **0.7125** | **0.8250** |
| **100** | 100 | 100 | 2 | 80 | **0.7421** | [0.6560, 0.8275] | **0.6750** | **0.8500** |
| **1000** | 1000 | 1000 | 0 | 80 | **0.6775** | [0.5741, 0.7646] | **0.6125** | **0.7625** |
| **1000** | 1000 | 1000 | 1 | 80 | **0.5996** | [0.5002, 0.6950] | **0.5250** | **0.7500** |
| **1000** | 1000 | 1000 | 2 | 80 | **0.6375** | [0.5458, 0.7396] | **0.5875** | **0.7250** |
| **Full** | 5663 | 5663 | 0 | 80 | **0.4573** | [0.3604, 0.5490] | **0.3500** | **0.6375** |
| **Full** | 5663 | 5663 | 1 | 80 | **0.4573** | [0.3594, 0.5542] | **0.3500** | **0.6375** |
| **Full** | 5663 | 5663 | 2 | 80 | **0.4573** | [0.3666, 0.5625] | **0.3500** | **0.6375** |

### 3.3 Aggregate Scaling Decay Summary

| Index Size $N$ | Aggregate Mean MRR | 95% Bootstrap CI | Aggregate Mean Hit@1 | Aggregate Mean Recall@6 |
| :--- | :---: | :---: | :---: | :---: |
| **$N = 10$** (Curated Baseline) | **0.7865** | [0.6979, 0.8625] | **0.7250** | **0.8875** |
| **$N = 100$** | **0.7526** | [0.6625, 0.8354] | **0.7000** | **0.8333** |
| **$N = 1,000$** | **0.6382** | [0.5708, 0.7729] | **0.5750** | **0.7458** |
| **$N = 5,663$** (Full Corpus Scale) | **0.4573** | [0.3542, 0.5625] | **0.3500** | **0.6375** |

**Net Degradation from Curated to Full Scale:**
- **MRR Drop:** $0.7865 \rightarrow 0.4573$ (**-41.86%** relative reduction)
- **Hit@1 Drop:** $0.7250 \rightarrow 0.3500$ (**-51.72%** relative reduction)
- **Recall@6 Drop:** $0.8875 \rightarrow 0.6375$ (**-28.17%** relative reduction)

---

## 4. Dense vs. Hybrid RRF Retrieval (Paired Evaluation)

### 4.1 Evaluation Setup
- **Questions:** Exactly 80 answerable research questions evaluated under identical index conditions ($N = 5,657$ indexed papers, $552,055$ chunks).
- **Dense Baseline:** Pure Chroma dense vector retrieval ($k=6$).
- **Hybrid RRF Pipeline:** Exact identifier extraction, BM25 full-text indexing, dense Chroma retrieval, fused via Reciprocal Rank Fusion ($k=60, \alpha=0.50$).

### 4.2 Aggregate Comparative Performance

| Retrieval Pipeline | Mean MRR | MRR 95% CI | Mean Hit@1 | Hit@1 95% CI | Mean Recall@6 | Recall@6 95% CI | Latency P50 | Latency P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline** | **0.4687** | [0.3760, 0.5677] | **0.3500** | [0.2500, 0.4625] | **0.6375** | [0.5375, 0.7500] | 62.4 ms | 184.2 ms |
| **Hybrid BM25 + Dense RRF** | **0.5740** | [0.4729, 0.6719] | **0.4875** | [0.3750, 0.6000] | **0.6875** | [0.5750, 0.7875] | 64.1 ms | 502.1 ms |

### 4.3 Paired Bootstrap Differences (10,000 Resamples)
- **MRR Difference:** $\mathbf{+0.1052}$ (95% CI: $[+0.0146, +0.1969]$, statistically significant at $\alpha = 0.05$)
- **Hit@1 Difference:** $\mathbf{+0.1375}$ (95% CI: $[+0.0375, +0.2500]$, statistically significant at $\alpha = 0.05$)
- **Recall@6 Difference:** $\mathbf{+0.0500}$ (95% CI: $[-0.0625, +0.1625]$)

### 4.4 Recovered Fraction of Lost MRR
- Distractor decay caused pure dense MRR to decline from $0.7865$ (curated) to $0.4573$ (sweep full) or $0.4687$ (baseline run).
- Hybrid retrieval lifts full-scale MRR to $0.5740$.
- **Mean Recovery Point Estimate:**
  $$\frac{0.5740 - 0.4573}{0.7865 - 0.4573} = \frac{0.1167}{0.3292} = \mathbf{35.45\%}$$
- **Bootstrap Uncertainty Range:** **6.0% to 61.1%**

### 4.5 Performance Breakdown by Question Category

| Question Category ($N$) | Dense Baseline MRR (5,657 Papers) | Hybrid RRF MRR (5,657 Papers) | Net Shift | Key Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Identifier ($n=10$)** | **0.1000** | **1.0000** | **+0.9000** | BM25 exact matching completely resolves bi-encoder blindness to arXiv IDs. |
| **Multi-hop ($n=1$)** | **0.3333** | **1.0000** | **+0.6667** | Exact lexical graph links resolve cross-paper multi-hop evidence. |
| **Comparison ($n=5$)** | **0.1500** | **0.3500** | **+0.2000** | Hybrid combination improves recall on multi-model comparative queries. |
| **Definition ($n=12$)** | **0.6250** | **0.6667** | **+0.0417** | Consistent precision on formal mathematical definitions. |
| **Method ($n=14$)** | **0.4524** | **0.4643** | **+0.0119** | Section intent priors preserve methodological passages. |
| **Open Problem ($n=13$)** | **0.6795** | **0.6538** | **-0.0257** | Dense semantics favored on open conceptual questions. |
| **Paraphrase ($n=25$)** | **0.5100** | **0.4067** | **-0.1033** | Paraphrased queries benefit from adaptive dense weighting. |
| **Weighted Total ($n=80$)** | **0.4688** | **0.5740** | **+0.1052** | Consistent full-index 5,657-paper evaluation across both columns. |

---

## 5. Attribution Integrity and Refusal Calibration Matrix

### 5.1 Refusal Evaluation Setup
- **Dataset:** Exactly 100 questions evaluated against full library scale ($N = 5,657$ papers).
  - 80 Answerable Questions (grounded in index papers)
  - 20 Out-of-Scope / Unanswerable Questions (adversarial queries outside the library corpus)
- **Model:** Google Gemini 1.5 Flash (temperature 0.2, prompt instructions requiring refusal when evidence is absent).

### 5.2 2x2 Refusal Confusion Matrix (Official Whitelist Grader)
Evaluated with the official strict 12-phrase whitelist grader:

| | Truly Unanswerable (Positive, $N=20$) | Truly Answerable (Negative, $N=80$) | Total |
| :--- | :---: | :---: | :---: |
| **Model Refused** | **$\text{TP} = 13$** | **$\text{FP} = 19$** (False Refusal) | **32** |
| **Model Answered** | **$\text{FN} = 7$** | **$\text{TN} = 61$** | **68** |
| **Total** | **20** | **80** | **100** |

**Derived Calibration Metrics:**
- **Refusal Recall:** $\text{TP} / 20 = 13 / 20 = \mathbf{65.00\%}$ (Wilson 95% CI: $[43.29\%, 81.88\%]$)
- **False Refusal Rate:** $\text{FP} / 80 = 19 / 80 = \mathbf{23.75\%}$ (Wilson 95% CI: $[15.82\%, 34.00\%]$)
- **Refusal Precision:** $\text{TP} / 32 = 13 / 32 = \mathbf{40.62\%}$ (Wilson 95% CI: $[25.53\%, 57.73\%]$)
- **Correct Non-Refusal Rate (Specificity):** $\text{TN} / 80 = 61 / 80 = \mathbf{76.25\%}$ (Wilson 95% CI: $[66.00\%, 84.18\%]$)

### 5.3 Grader Definition Sensitivity Analysis
- **Official Whitelist:** Strict match against 12 exact canonical refusal stems ($\text{TP}=13, \text{FN}=7$).
- **Broader Semantic Grader:** Recognizes natural conversational hedges (e.g., *"not discussed in the provided text"*, *"the corpus does not contain information on"*).
  - $\text{TP} = 18$ ($\mathbf{90.00\%}$ recall)
  - $\text{FN} = 2$ ($10.00\%$ unflagged)
  - $\text{FP} = 21$ ($26.25\%$ false refusal rate)
  - $\text{TN} = 59$ ($73.75\%$ specificity)
  - Refusal precision: $18 / (18 + 21) = 46.15\%$.

### 5.4 Citation Validity and Verification
- **Total In-Text Citations Evaluated:** 90 bracketed citation claims (`[n]`).
- **Verifiable Valid Passages:** 89 citations (**98.89%**, Wilson 95% CI: $[93.97\%, 99.80\%]$).
- **Invalid Passage Citations:** 1 citation (**1.11%**).
- **Uncited Refusals:** 8 queries correctly declined without citations.
- **Generation Latency:**
  - P50 Latency: **9,854.5 ms** (9.85 s)
  - P95 Latency: **14,843.8 ms** (14.84 s)

---

## 6. Multi-Signal Ranking Architecture and Heuristics

### 6.1 Composite Ranking Formulation
$$S(p, q) = w_{\text{emb}} S_{\text{emb}}(p, q) + w_{\text{rec}} S_{\text{rec}}(p) + w_{\text{cit}} S_{\text{cit}}(p) + w_{\text{auth}} S_{\text{auth}}(p)$$

### 6.2 Component Signal Definitions
1. **Semantic Similarity ($S_{\text{emb}}$):**
   $$S_{\text{emb}}(p, q) = \cos(\mathbf{e}_q, \mathbf{e}_p) = \frac{\mathbf{e}_q \cdot \mathbf{e}_p}{\|\mathbf{e}_q\| \|\mathbf{e}_p\|}$$
   Computed over 768-dimensional title and abstract embeddings.
2. **Exponential Recency ($S_{\text{rec}}$):**
   $$S_{\text{rec}}(p) = 0.5^{(\text{age} / t_{1/2})}, \quad \text{age} = \max(0, \text{year}_{\text{now}} - \text{year}_{\text{pub}})$$
   Hyperparameter: $t_{1/2} = 4.0\text{ years}$. Missing year defaults to neutral $0.30$.
3. **Log-Normalized Citation Score ($S_{\text{cit}}$):**
   $$S_{\text{cit}}(p) = \min\left(1.0, \frac{\ln(1 + c)}{\ln(1 + 500)}\right)$$
   Normalized against a 500-citation benchmark ceiling. Missing count defaults to neutral $0.30$.
4. **Annualized Citation Velocity ($S_{\text{vel}}$):**
   $$v = \frac{c + 5.0 \cdot c_{\text{lib}}}{\max(1, \text{age})}, \quad S_{\text{vel}}(p) = \min\left(1.0, \frac{\ln(1 + v)}{\ln(1 + 50)}\right)$$
   Normalized against a 50 citations/year benchmark. Highlights rapid-velocity breakout preprints.
5. **Author Continuity ($S_{\text{auth}}$):**
   $$S_{\text{auth}}(p) = \min\left(1.0, \frac{\ln(1 + f)}{\ln(1 + 10)}\right)$$
   Log-scaled frequency $f$ of paper authors across the user's active library session.

### 6.3 Standard Inspectable Weight Presets

| Preset Name | $w_{\text{emb}}$ | $w_{\text{rec}}$ | $w_{\text{cit}}$ | $w_{\text{auth}}$ | Primary Search Intent |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Balanced (Default)** | **0.50** | **0.20** | **0.25** | **0.05** | General literature discovery balancing relevance, freshness, and impact. |
| **Precision** | **0.65** | **0.10** | **0.20** | **0.05** | Narrow topical searches where semantic alignment takes top priority. |
| **Prefer Recent** | **0.45** | **0.35** | **0.15** | **0.05** | Tracking emerging preprints in rapidly evolving research areas. |
| **Citation Heavy** | **0.35** | **0.10** | **0.50** | **0.05** | Finding landmark papers and foundational historical works. |

### 6.4 Worked Top-5 Query Breakdown: "vision transformer"

| Final Rank | arXiv ID | Short Title | Semantic (50%) | Recency (20%) | Citation (25%) | Author (5%) | Total Score |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1** | **2604.133** | *Adaptive Sparse Attention in ViTs* | 0.421 | 0.200 | 0.076 | 0.046 | **0.743** |
| **#2** | **2604.144** | *Hierarchical Vision Transformers* | 0.422 | 0.200 | 0.076 | 0.042 | **0.740** |
| **#3** | **2605.282** | *Linear Attention for Image Classification* | 0.420 | 0.200 | 0.076 | 0.031 | **0.727** |
| **#4** | **2604.180** | *Cross-Covariance Image Transformers* | 0.425 | 0.200 | 0.076 | 0.021 | **0.722** |
| **#5** | **2601.141** | *Multi-Scale Vision Encoders* | 0.420 | 0.200 | 0.076 | 0.022 | **0.718** |

---

## 7. System Hyperparameters and Architecture Specifications

| Subsystem | Parameter / Component | Value | Rationale |
| :--- | :--- | :--- | :--- |
| **Embedding Engine** | Local Primary Model | `google/embeddinggemma-2` | 768-dim native representation, local Apple Silicon MPS and CPU execution. |
| **Embedding Engine** | Cloud Fallback Model | `all-MiniLM-L6-v2` | 384-dim fallback for lightweight or remote environments. |
| **Chunking Engine** | Target Chunk Size | **1,000 characters** | Balances sentence context and retrieval specificity. |
| **Chunking Engine** | Chunk Overlap | **200 characters** | Preserves context across chunk boundaries. |
| **Hybrid Retrieval** | Fusion Formula | **Reciprocal Rank Fusion (RRF)** | $RRF(d) = \sum \frac{w_i}{k + r_i(d)}$ with $k = 60$. |
| **Hybrid Retrieval** | Dense / BM25 Weight | $\alpha = 0.50, 1 - \alpha = 0.50$ | Equal balance of lexical precision and dense semantics. |
| **Hybrid Retrieval** | Passages Retrieved | $k = 6\text{ passages}$ | Sized for Gemini 1.5 Flash input budget. |
| **Generation Engine** | Primary LLM | `google/gemini-1.5-flash` | Fast inference, grounded prompt, structured quote attribution. |
| **Generation Engine** | Generation Temperature | $0.2$ | Low temperature to prevent hallucinated citations and enforce refusal. |
| **Ranking Engine** | Retrieval Pool Multiplier | $3 \times k$ (minimum 20) | Generates candidate pool for multi-signal re-ranking. |
