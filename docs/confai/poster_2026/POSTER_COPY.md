# ConfAI 2026 Poster: Complete Text Copy

**Event:** ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition  
**Title:** Literature Discovery with Multi-Signal Ranking, Citation-Grounded Hybrid RAG, and 2D Novelty Geometry  
**Author:** Ajay Sharma (M.S. in AI, Plaksha University)  
**Supervisor:** Dr. Subrat Panda  
**Affiliation:** Plaksha University, Punjab, India  
**Repository:** https://github.com/NatalTiger26/ArXiv_Paper_Suggester  

---

## 1. Problem Statement & Motivation
Keyword search returns unranked catalogs, while general large language models answer without verifiable links to passages in papers the researcher actually chose to study. Furthermore, existing tools treat scientific innovation as a black box rather than projecting it geometrically against foundational ancestors. We introduce a complete scientific discovery workstation bridging multi-signal abstract ranking, selective full-text hybrid RAG with verifiable passage-level citations, and 2D reference cone novelty geometry.

---

## 2. Key Takeaways
- **Multi-Signal Discovery Over 101k Papers:** Transparent 4-signal ranking (semantic embedding, recency decay with 4.0 yr half-life, citation impact, and author continuity) over 5,752 FTS5 catalog papers and 101k citation graph hubs.
- **Selective Ingestion & Hybrid RRF RAG:** Empirical evaluation proves pure dense retrieval degrades from 0.786 to 0.457 MRR at scale. Selective full-text ingestion combined with dense + BM25 Reciprocal Rank Fusion and section priors preserves high precision with 98.9% citation validity.
- **2D Reference Cone Geometry:** Non-Negative Least Squares (NNLS) cone projection decomposes papers into closest parent ($m$), multi-parent synergy gain ($\Delta$), explained energy ($R^2_+$), and orthogonal novelty residual gap ($\sqrt{1 - R^2_+}$).

---

## 3. End-to-End System Architecture

### Pipeline Narrative
1. **Literature Discovery:** Remote papers from arXiv, Semantic Scholar, and OpenAlex are indexed in SQLite FTS5 (5,752 papers) and an edge citation graph (101,716 papers, 226,857 edges).
2. **Multi-Signal Ranking:** Candidates are filtered using inspectable weight presets across embedding similarity, recency decay, citation saturation, and author domain authority.
3. **Selective Ingestion:** Researchers selectively ingest chosen papers. PDFs/HTML are converted to clean Markdown via PyMuPDF4LLM, segmented into header-aware chunks (1,000 chars), and indexed in Chroma with local `google/embeddinggemma-2`.
4. **Hybrid RAG Synthesis:** Queries undergo dense + BM25 Reciprocal Rank Fusion with section-intent priors, delivering grounded context to Gemini with verifiable passage-level `[1], [2]` citations.
5. **2D Novelty Geometry:** References are resolved via a multi-tier cascade (S2, OpenAlex, direct arXiv HTML), projected onto the reference cone via NNLS, and rendered on an interactive 2D canvas.

---

## 4. Pillar 1: Multi-Signal Ranking & Citation Hubs

### Scoring Formula
$$\text{Score}(\mathbf{q}, \mathbf{p}) = w_{\text{emb}} \cdot S_{\text{emb}}(\mathbf{q}, \mathbf{p}) + w_{\text{rec}} \cdot S_{\text{rec}}(\mathbf{p}) + w_{\text{cit}} \cdot S_{\text{cit}}(\mathbf{p}) + w_{\text{auth}} \cdot S_{\text{auth}}(\mathbf{p})$$

- **Embedding Similarity ($S_{\text{emb}}$):** Cosine dot product in dense embedding space.
- **Recency Decay ($S_{\text{rec}}$):** Exponential half-life $S_{\text{rec}} = 2^{-\Delta t / t_{1/2}}$ with default $t_{1/2} = 4.0$ years.
- **Citation Saturation ($S_{\text{cit}}$):** Log-saturated impact $S_{\text{cit}} = \log(1 + C) / \log(1 + C_{\text{max}})$.
- **Author Domain Continuity ($S_{\text{auth}}$):** Normalized author frequency proxy within the topic collection.

### Inspectable Preset Configurations
- **Balanced (Default):** $w_{\text{emb}} = 0.50, w_{\text{rec}} = 0.20, w_{\text{cit}} = 0.25, w_{\text{auth}} = 0.05$
- **Precision (Topic Focus):** $w_{\text{emb}} = 0.65, w_{\text{rec}} = 0.10, w_{\text{cit}} = 0.20, w_{\text{auth}} = 0.05$
- **Prefer Recent (Emerging Trends):** $w_{\text{emb}} = 0.45, w_{\text{rec}} = 0.35, w_{\text{cit}} = 0.15, w_{\text{auth}} = 0.05$
- **Citation Heavy (Landmark Works):** $w_{\text{emb}} = 0.35, w_{\text{rec}} = 0.10, w_{\text{cit}} = 0.50, w_{\text{auth}} = 0.05$

---

## 5. Pillar 2: Selective Ingestion & Hybrid RAG

### The Selective Ingest Advantage
Rather than indexing millions of uncurated documents that dilute dense retrieval precision, the researcher curates high-relevance full texts into a focused project library.

### Processing & Chunking Specs
- **Parser:** PyMuPDF4LLM with layout preservation and OCR fallback for scanned figures.
- **Chunking:** Header-aware markdown splitting, 1,000 character target size, 180 character boundary overlap, 80 character minimum length.
- **Hybrid RRF & Section Priors:**
  $$\text{RRF Score}(d) = \frac{w_{\text{dense}}}{60 + r_{\text{dense}}(d)} + \frac{w_{\text{lex}}}{60 + r_{\text{lex}}(d)} + \text{Prior}(\text{section}, \mathbf{q})$$
  Dynamically promotes Methodology chunks for procedural queries and Results chunks for benchmark questions.
- **Grounded Streaming Generation:** Prompt forces explicit passage mapping, yielding 98.9% citation validity (89/90 valid references).

---

## 6. Pillar 3: 2D Novelty Geometry & Subspace Cones

### Mathematical Optimization
Given paper vector $\mathbf{x} \in \mathbb{R}^d$ and cited reference matrix $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_k] \in \mathbb{R}^{d \times k}$:
$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

### Innovation Decomposition Metrics
- **Closest Parent Anchor ($m$):** $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$ aligned along basis vector $\mathbf{e}_1 = \mathbf{y}_{m^*}$ at coordinate $(1, 0)$.
- **Multi-Parent Synergy Gain ($\Delta$):** $\Delta = R^2_+ - \max(m, 0)^2$ where $R^2_+ = \|\hat{\mathbf{x}}\|^2$.
- **Orthogonal Novelty Residual Gap:** $\text{Res} = \sqrt{\max(0, 1 - R^2_+)}$ measuring departure from prior art.
- **2D Orthonormal Basis:** $\mathbf{u} = \hat{\mathbf{x}} - (\hat{\mathbf{x}}^T \mathbf{e}_1)\mathbf{e}_1$, with synergy axis $\mathbf{e}_2 = \mathbf{u} / \|\mathbf{u}\|$. Projection $\hat{\mathbf{x}}$ sits at $(m, \sqrt{\Delta})$.

### Classification Typology
- **High-Synergy Blend ($\Delta \ge 0.05$):** Combines disparate intellectual lineages.
- **Linear Extension ($m \ge 0.60, \Delta < 0.05$):** Extends single closest parent.
- **Paradigm Shift ($1 - R^2_+ \ge 0.45$):** Orthogonal departure from cited subspace.

---

## 7. Pillar 4: ConfAI 2026 Empirical Evaluation

### Index Scaling Law: Retrieval Quality vs. Corpus Size
Evaluated on fixed 100-question benchmark with 95% bootstrap confidence intervals:

| Ingested Corpus Scale | Papers ($N$) | Chunks | Mean MRR | 95% Bootstrap CI | Recall@k | Hit@1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Curated Subspace** | **10** | ~1,725 | **0.786** | [0.702, 0.863] | **0.888** | **0.725** |
| **Topic Neighborhood** | **100** | ~9,800 | **0.757** | [0.658, 0.842] | **0.825** | **0.713** |
| **Domain Corpus** | **1,000** | ~98,000 | **0.638** | [0.546, 0.740] | **0.725** | **0.588** |
| **Full Uncurated Library** | **5,663** | 551,743 | **0.457** | [0.360, 0.549] | **0.638** | **0.350** |

*Key Finding:* Pure dense retrieval degrades by 41.9% in MRR and 51.7% in Hit@1 when scaling from curated subsets to full uncurated libraries, demonstrating why selective ingestion and hybrid RRF are critical.

### Generation & Attribution Integrity
- **Passage Citation Validity:** 98.9% (89 of 90 cited links verifiable against retrieved chunks).
- **Out-of-Scope Rejection Recall:** 65.0% (13 of 20 unanswerable queries correctly refused).
- **Refusal Precision:** 40.6% (highlights open calibration boundary where the model tends to decline conservatively on weak evidence).

---

## 8. Open Limits & Engineering Boundaries
- **Refusal Calibration:** Conservative safety prompting produces false refusals on marginal contexts (23.8% answerable queries declined).
- **Lexical Identifier Queries:** Dense embeddings underperform on exact alphanumeric theorem/equation IDs without hybrid BM25 support.
- **Hardware Footprint:** 2B-parameter `google/embeddinggemma-2` requires proactive Metal cache draining on Apple Silicon (batch size capped at 6).

---

## 9. References & Acknowledgements
1. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
2. Jacot et al., "Neural Tangent Kernel," NeurIPS 2018.
3. Cormack et al., "Reciprocal Rank Fusion," SIGIR 2009.
4. Srivastava et al., "Beyond the Imitation Game (BIG-bench)," arXiv 2022.

*Supervised by Dr. Subrat Panda, Plaksha University. Presented at ConfAI 2026 AI in Practice Track.*
