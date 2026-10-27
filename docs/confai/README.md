# ConfAI 2026 Poster Deliverable Package

**Event:** ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition  
**Author:** Ajay Sharma (M.S. in AI, Plaksha University)  
**Supervisor:** Dr. Subrat Panda  
**Institution:** Center for Artificial Intelligence, Plaksha University  
**Official Title:** *Literature Discovery with Multi-Signal Ranking, Citation-Grounded Hybrid RAG, and 2D Novelty Geometry*  
**Paper Format:** Vertical A1 Portrait (594 mm × 841 mm)  

---

## Deliverable Inventory

All active ConfAI 2026 deliverables are located in the `docs/confai/poster_2026/` directory:

| File | Type | Description |
|---|---|---|
| [`poster.pdf`](poster_2026/poster.pdf) | Vector PDF (1.39 MB) | Print-ready A1 portrait document (594 mm × 841 mm, exact single-page MediaBox: `1684.08 pt × 2383.92 pt`). |
| [`poster.html`](poster_2026/poster.html) | Standalone HTML | Self-contained HTML/CSS source layout with inline vector SVGs, KaTeX formulas, and CSS print media rules. |
| [`generate_poster.py`](poster_2026/generate_poster.py) | Python Script | Automated builder script to render `poster.html` and invoke headless Chrome to generate `poster.pdf`. |
| [`POSTER_COPY.md`](poster_2026/POSTER_COPY.md) | Markdown Document | Complete verbatim text copy, including title, institutional affiliations, takeaways, equations, table data, and references. |
| [`POSTER_SOURCES.md`](poster_2026/POSTER_SOURCES.md) | Markdown Document | Line-by-line audit trail linking every metric, table cell, and equation to verified codebase files and SQLite databases. |
| [`qr_github.svg`](poster_2026/qr_github.svg) | Vector SVG | High-contrast vector QR code pointing to the GitHub repository. |
| [`qr_demo.svg`](poster_2026/qr_demo.svg) | Vector SVG | High-contrast vector QR code pointing to quick start and CLI usage. |

---

## Core System Architecture & Narrative

The poster synthesizes the three core capabilities of the ArXiv Scientific Discovery workstation:

### Pillar 1: Transparent Multi-Signal Literature Discovery
- **Catalog Scale:** 5,752 curated arXiv papers indexed in SQLite FTS5.
- **Citation Graph Scale:** 101,716 papers connected across 226,857 citation edges.
- **Heuristic Discovery Formula:** Transparent linear combination of 4 distinct signals:
  - Dense cosine similarity via local 768-dimensional `EmbeddingGemma 2`.
  - Half-life exponential recency decay with a 4.0 year characteristic parameter.
  - Saturated logarithmic citation impact: `log1p(citations) / log1p(1000)`.
  - Author continuity overlap: `|paper_authors ∩ query_authors| / |paper_authors|`.
- **User Inspectability:** Interactive weight presets (Semantic First, Recent Breakthroughs, Foundational Giants, Author Lineage).

### Pillar 2: Selective Ingestion & Citation-Grounded Hybrid RAG
- **Selective Full-Text Ingestion:** Bypasses brute-force vector storage overhead by indexing full text only for researcher-selected papers.
- **Corpus Scaling Empirical Law:** Pure dense vector search degrades sharply as corpus size scales to 5,663 uncurated papers:
  - Curated baseline (N=10): MRR 0.786 [95% CI 0.70 to 0.86], Recall@k 0.888, Hit@1 0.725.
  - Medium corpus (N=1,000): MRR 0.638, Recall@k 0.725, Hit@1 0.588.
  - Full uncurated library (N=5,663): MRR 0.457 [95% CI 0.36 to 0.55], Recall@k 0.638, Hit@1 0.350 (-51.7% drop in top-1 precision).
- **Hybrid RRF Boost:** Blending BM25 lexical search with dense vector rankings via Reciprocal Rank Fusion and section priors lifts MRR from 0.574 to 0.612 (+6.6%), directly recovering exact alphanumeric identifier queries.
- **Attribution Integrity:** Achieves 98.9% citation validity (89 of 90 cited statements verified against source text) with 65.0% refusal recall on unanswerable questions.

### Pillar 3: 2D Reference Cone Geometry & Innovation Attribution
- **Convex Cone Projection:** Decomposes target paper vector `x` into cited bibliography basis `Y = [y_1, ..., y_k]` via Non-Negative Least Squares (NNLS):
  - `lambda* = argmin_{lambda >= 0} || x - Y lambda ||^2`
  - Reconstructed prior-art vector: `x_hat = Y lambda*`
  - Subspace Energy: `R^2_+ = || x_hat ||^2 in [0, 1]`
- **Orthonormal Canvas Basis:**
  - `e_1`: Closest parent ancestor `y_{m*}` plotted at `(1, 0)`.
  - `e_2`: Orthogonal multi-parent synergy vector `u / ||u||`.
  - Synergy blend gain: `Delta = R^2_+ - max(m, 0)^2`.
  - Novelty residual gap: `Res = sqrt(max(0, 1 - R^2_+))`.
- **BIG-bench Exemplar (`2206.04615`):** 706 parsed references, anchor BERT (`1810.04805`, 24.9% share, cos=0.766), anchor correlation `m = 0.774`, synergy blend `Delta = 0.138`, subspace energy `R^2_+ = 0.737`, residual gap `Res = 0.513`.

---

## Instructions: Rebuilding & Printing

### Rebuilding Deliverables
To recompile the HTML layout and vector PDF:

```bash
python3 docs/confai/poster_2026/generate_poster.py
```

Or invoke headless Google Chrome directly:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless \
  --disable-gpu \
  --print-to-pdf="docs/confai/poster_2026/poster.pdf" \
  --print-to-pdf-no-header \
  "docs/confai/poster_2026/poster.html"
```

### Physical Printing Specifications
- **Dimensions:** ISO 216 A1 Portrait (594 mm wide × 841 mm high).
- **Margins:** 0 mm borderless print using the CSS page rule `@page { size: 594mm 841mm }` with zero external margins.
- **Internal Margins:** Safe content margin is embedded inside the container (12 mm top/bottom, 14 mm left/right).
- **Resolution:** All visual diagrams (hero architecture pipeline, 2D novelty cone canvas, QR codes) are scalable vector graphics (SVG) with sharp typography.
