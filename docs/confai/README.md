# ConfAI 2026 Academic Poster Deliverable Package

**Event:** ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition  
**Author:** Ajay Sharma (M.S. in AI, Plaksha University)  
**Supervisor:** Dr. Subrat Panda  
**Institution:** Center for Artificial Intelligence, Plaksha University  
**Official Title:** *Literature Discovery with Multi-Signal Ranking, Citation-Grounded Hybrid RAG, and 2D Novelty Geometry*  
**Paper Format:** Vertical ISO 216 A1 Portrait (594 mm × 841 mm)  

---

## Deliverable Inventory

All ConfAI 2026 deliverables reside in the `docs/confai/poster_2026/` directory:

| File | Type | Description |
|---|---|---|
| [`poster.pdf`](poster_2026/poster.pdf) | Vector PDF | Print-ready A1 portrait document (594 mm × 841 mm, exact single-page MediaBox: 1684.08 pt × 2383.92 pt). |
| [`poster.html`](poster_2026/poster.html) | Standalone HTML | Self-contained HTML layout with pure CSS styling, inline vector SVGs, and print media rules. |
| [`generate_poster.py`](poster_2026/generate_poster.py) | Python Script | Automated builder script compiling HTML to vector PDF via headless Google Chrome with automated audit suite. |
| [`generate_charts.py`](poster_2026/generate_charts.py) | Python Script | Dedicated Matplotlib publication chart generator creating 9 core and 2 companion vector SVGs. |
| [`figures/`](poster_2026/figures/) | Directory | Complete vector chart package (9 core charts embedded on poster, 2 companion charts archived). |
| [`preview.png`](poster_2026/preview.png) | High-Res PNG | Rendered full-poster preview for inspection and automated visual audit. |
| [`POSTER_COPY.md`](poster_2026/POSTER_COPY.md) | Markdown Document | Complete verbatim text copy of the final poster. |
| [`POSTER_SOURCES.md`](poster_2026/POSTER_SOURCES.md) | Markdown Document | Line-by-line audit trail linking every metric, table, and equation to verified repository files and SQLite databases. |
| [`qr_github.svg`](poster_2026/qr_github.svg) | Vector SVG | High-contrast vector QR code pointing to the public GitHub repository. |
| [`qr_demo.svg`](poster_2026/qr_demo.svg) | Vector SVG | High-contrast vector QR code pointing to the live interactive Streamlit Cloud demo. |

---

## Compilation & Verification Instructions

To regenerate all charts, recompile the poster, and run the complete headless audit suite:

```bash
uv run python docs/confai/poster_2026/generate_charts.py
uv run python docs/confai/poster_2026/generate_poster.py --audit
```

The audit suite performs seven automated checks:
1. Single-page ISO 216 A1 MediaBox dimensions (594.1 mm × 841.0 mm).
2. Minimum typography floor enforcement (Title 70 pt, Headings 42 pt, Body 22-24 pt, Axis 24-25 pt, References 18 pt).
3. Zero prose semicolons and zero em-dashes across all rendered text.
4. Word count budget (target: 450 to 650 prose words).
5. Bounding box overflow and text overlap verification across the entire canvas.
6. Independent OpenCV QR code decoding verifying live GitHub and Streamlit URLs.
7. Inset whitespace scan across all three columns verifying that the longest blank vertical gap is 12 mm or less.
