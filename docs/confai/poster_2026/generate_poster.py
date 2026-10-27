#!/usr/bin/env python3
"""
Generate self-contained print-ready ConfAI 2026 A1 Poster (HTML & Vector PDF).
Dimensions: 594mm x 841mm (A1 Portrait).
"""

from pathlib import Path
import subprocess

DIR = Path(__file__).resolve().parent

with open(DIR / "qr_github.svg") as f:
    gh_raw = f.read()
gh_svg = '<svg' + gh_raw.split('<svg', 1)[1]

with open(DIR / "qr_demo.svg") as f:
    demo_raw = f.read()
demo_svg = '<svg' + demo_raw.split('<svg', 1)[1]

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ConfAI 2026 Poster: Literature Discovery with Multi-Signal Ranking, Hybrid RAG, and 2D Novelty Geometry</title>
  <style>
    @page {{
      size: 594mm 841mm;
      margin: 0;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    body {{
      width: 594mm;
      height: 841mm;
      margin: 0 auto;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #0f172a;
      line-height: 1.35;
      font-size: 8.5pt;
    }}
    .poster-container {{
      width: 594mm;
      height: 841mm;
      padding: 12mm 14mm;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: #ffffff;
    }}

    /* Header styling */
    .header {{
      border-bottom: 2.5px solid #1e3a8a;
      padding-bottom: 8mm;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10mm;
    }}
    .header-main {{
      flex: 1;
    }}
    .event-badge {{
      display: inline-block;
      background: #1e3a8a;
      color: #ffffff;
      font-weight: 700;
      font-size: 8.5pt;
      letter-spacing: 0.5px;
      padding: 2.5px 10px;
      border-radius: 4px;
      text-transform: uppercase;
      margin-bottom: 3mm;
    }}
    .title {{
      font-size: 20pt;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.15;
      margin-bottom: 3.5mm;
      letter-spacing: -0.3px;
    }}
    .subtitle {{
      font-size: 11pt;
      font-weight: 600;
      color: #2563eb;
      margin-bottom: 4mm;
    }}
    .authors-line {{
      font-size: 9.5pt;
      color: #1e293b;
    }}
    .authors-line strong {{
      color: #0f172a;
      font-weight: 700;
    }}
    .inst-badge {{
      font-size: 9pt;
      color: #475569;
      margin-top: 1.5mm;
    }}
    .header-aside {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 2mm;
      min-width: 55mm;
    }}
    .institution-pill {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      padding: 3mm 5mm;
      border-radius: 6px;
      text-align: right;
    }}
    .institution-pill .inst-name {{
      font-weight: 800;
      font-size: 10pt;
      color: #0f172a;
    }}
    .institution-pill .inst-dept {{
      font-size: 8pt;
      color: #64748b;
    }}

    /* Key Takeaways Bar */
    .takeaways-strip {{
      display: grid;
      grid-template-columns: 1fr 1.15fr 1fr;
      gap: 4mm;
      margin: 4.5mm 0;
    }}
    .takeaway-card {{
      background: #f8fafc;
      border: 1.2px solid #cbd5e1;
      border-top: 3.5px solid #2563eb;
      border-radius: 6px;
      padding: 3mm 4mm;
    }}
    .takeaway-card.accent {{
      border-top-color: #059669;
      background: #f0fdf4;
      border-color: #bbf7d0;
    }}
    .takeaway-card.novelty {{
      border-top-color: #7c3aed;
      background: #faf5ff;
      border-color: #e9d5ff;
    }}
    .takeaway-title {{
      font-weight: 700;
      font-size: 9pt;
      color: #0f172a;
      margin-bottom: 1.5mm;
      display: flex;
      align-items: center;
      gap: 2mm;
    }}
    .takeaway-desc {{
      font-size: 7.8pt;
      color: #334155;
      line-height: 1.3;
    }}

    /* Hero SVG Diagram Banner */
    .hero-banner {{
      background: #f8fafc;
      border: 1.2px solid #e2e8f0;
      border-radius: 8px;
      padding: 3mm 4mm;
      margin-bottom: 4.5mm;
    }}
    .section-tag {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2mm;
    }}
    .section-tag-title {{
      font-size: 8.5pt;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #1e3a8a;
      display: flex;
      align-items: center;
      gap: 2mm;
    }}
    .section-tag-title::before {{
      content: "";
      width: 3.5mm;
      height: 3.5mm;
      background: #2563eb;
      border-radius: 2px;
    }}
    .hero-legend {{
      display: flex;
      gap: 4mm;
      font-size: 7.5pt;
      font-weight: 600;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 1.5mm;
    }}
    .legend-box {{
      width: 3mm;
      height: 3mm;
      border-radius: 2px;
    }}
    .legend-human {{ background: #fef3c7; border: 1px solid #d97706; }}
    .legend-auto {{ background: #dbeafe; border: 1px solid #2563eb; }}
    .legend-storage {{ background: #e2e8f0; border: 1px solid #64748b; }}

    /* Main Columns Layout */
    .main-columns {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6mm;
      flex: 1;
    }}
    .col {{
      display: flex;
      flex-direction: column;
      gap: 4.5mm;
    }}

    /* Panel Card Styles */
    .panel-card {{
      background: #ffffff;
      border: 1.2px solid #cbd5e1;
      border-radius: 7px;
      padding: 3.5mm 4.5mm;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    .panel-header {{
      border-bottom: 1.5px solid #e2e8f0;
      padding-bottom: 1.8mm;
      margin-bottom: 2.5mm;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}
    .panel-title {{
      font-size: 10pt;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: -0.2px;
      display: flex;
      align-items: center;
      gap: 2mm;
    }}
    .panel-kicker {{
      font-size: 7.5pt;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}

    /* Content Typography & Math */
    p.lead-text {{
      font-size: 8.2pt;
      color: #334155;
      margin-bottom: 2.2mm;
      line-height: 1.35;
    }}
    .formula-box {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3.5px solid #2563eb;
      border-radius: 4px;
      padding: 2mm 3.5mm;
      margin: 2mm 0;
      font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
      font-size: 9pt;
      color: #0f172a;
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    .formula-explanation {{
      font-size: 7.6pt;
      color: #475569;
      line-height: 1.3;
      margin-bottom: 2mm;
    }}

    /* Visual Weight Bars */
    .preset-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1.5mm;
      font-size: 7.5pt;
    }}
    .preset-table th {{
      background: #f1f5f9;
      text-align: left;
      padding: 1.5mm 2mm;
      font-weight: 700;
      color: #1e293b;
      border-bottom: 1px solid #cbd5e1;
    }}
    .preset-table td {{
      padding: 1.5mm 2mm;
      border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }}
    .bar-wrap {{
      display: flex;
      height: 4.5mm;
      border-radius: 3px;
      overflow: hidden;
      width: 100%;
      background: #e2e8f0;
    }}
    .bar-seg {{
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 6pt;
      font-weight: 700;
      color: #ffffff;
    }}
    .seg-emb {{ background: #2563eb; }}
    .seg-rec {{ background: #059669; }}
    .seg-cit {{ background: #d97706; }}
    .seg-auth {{ background: #7c3aed; }}

    /* Data Tables */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 7.5pt;
      margin: 1.5mm 0;
    }}
    .data-table th {{
      background: #f1f5f9;
      padding: 1.8mm 2mm;
      text-align: left;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1.5px solid #cbd5e1;
    }}
    .data-table td {{
      padding: 1.6mm 2mm;
      border-bottom: 1px solid #e2e8f0;
      color: #334155;
    }}
    .data-table tr.highlight {{
      background: #f0fdf4;
      font-weight: 700;
    }}
    .data-table tr.highlight td {{
      color: #065f46;
    }}
    .badge-win {{
      display: inline-block;
      background: #dcfce7;
      color: #166534;
      padding: 1px 4px;
      border-radius: 3px;
      font-weight: 700;
      font-size: 6.5pt;
    }}
    .badge-drop {{
      display: inline-block;
      background: #fee2e2;
      color: #991b1b;
      padding: 1px 4px;
      border-radius: 3px;
      font-weight: 700;
      font-size: 6.5pt;
    }}

    /* 2D Novelty Cone Layout */
    .novelty-grid {{
      display: grid;
      grid-template-columns: 1fr 1.05fr;
      gap: 3.5mm;
      align-items: center;
    }}
    .metric-pills {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5mm;
      margin-top: 1.5mm;
    }}
    .metric-pill {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 1.5mm 2mm;
      border-radius: 4px;
    }}
    .metric-pill-label {{
      font-size: 6.5pt;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
    }}
    .metric-pill-val {{
      font-size: 9.5pt;
      font-weight: 800;
      color: #0f172a;
      font-family: monospace;
    }}

    /* Footer Strip */
    .footer-strip {{
      border-top: 2px solid #cbd5e1;
      padding-top: 4.5mm;
      margin-top: 4mm;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6mm;
    }}
    .footer-left {{
      flex: 1;
    }}
    .limits-box {{
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 5px;
      padding: 2mm 3.5mm;
      font-size: 7.2pt;
      color: #92400e;
      line-height: 1.35;
      margin-bottom: 2mm;
    }}
    .limits-box strong {{
      color: #78350f;
    }}
    .refs-line {{
      font-size: 6.8pt;
      color: #64748b;
      line-height: 1.3;
    }}
    .footer-qrs {{
      display: flex;
      gap: 5mm;
      align-items: center;
    }}
    .qr-block {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1.2mm;
    }}
    .qr-svg {{
      width: 17mm;
      height: 17mm;
      padding: 0.8mm;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
    }}
    .qr-label {{
      font-size: 6.5pt;
      font-weight: 700;
      color: #1e293b;
      text-align: center;
    }}
  </style>
</head>
<body>

<div class="poster-container">

  <!-- 1. HEADER -->
  <header class="header">
    <div class="header-main">
      <div class="event-badge">ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition</div>
      <h1 class="title">Literature Discovery with Multi-Signal Ranking,<br>Citation-Grounded Hybrid RAG, and 2D Novelty Geometry</h1>
      <div class="subtitle">An End-to-End Scientific Synthesis Workstation Bridging Discovery, Comprehension, and Prior-Art Innovation</div>
      <div class="authors-line">
        <strong>Ajay Sharma</strong><sup>1</sup> (M.S. in AI) &nbsp;·&nbsp;
        Supervisor: <strong>Dr. Subrat Panda</strong><sup>1</sup>
      </div>
      <div class="inst-badge">
        <sup>1</sup>Center for Artificial Intelligence, <strong>Plaksha University</strong>, Mohali, Punjab, India
      </div>
    </div>
    <div class="header-aside">
      <div class="institution-pill">
        <div class="inst-name">Plaksha University</div>
        <div class="inst-dept">M.S. in Artificial Intelligence</div>
      </div>
      <div style="font-size: 7.5pt; color: #64748b; font-weight: 600; margin-top: 1mm;">
        Track: AI in Practice &nbsp;•&nbsp; 2026
      </div>
    </div>
  </header>

  <!-- 2. KEY TAKEAWAYS STRIP -->
  <div class="takeaways-strip">
    <div class="takeaway-card">
      <div class="takeaway-title">
        <span style="color:#2563eb;">①</span> Transparent Multi-Signal Ranking
      </div>
      <div class="takeaway-desc">
        Ranks <strong>5,752 catalog papers</strong> and navigates <strong>101,716 works (226,857 edges)</strong> using an inspectable 4-signal heuristic: semantic embedding, exponential recency decay (4.0 yr half-life), log citation saturation, and author domain continuity.
      </div>
    </div>
    <div class="takeaway-card accent">
      <div class="takeaway-title">
        <span style="color:#059669;">②</span> Selective Ingest &amp; Hybrid RRF RAG
      </div>
      <div class="takeaway-desc">
        Pure dense retrieval degrades by <strong>41.9% in MRR</strong> as the corpus scales to 5,663 uncurated papers. Selective full-text ingestion with dense + BM25 Reciprocal Rank Fusion and section priors preserves high precision with <strong>98.9% citation validity</strong>.
      </div>
    </div>
    <div class="takeaway-card novelty">
      <div class="takeaway-title">
        <span style="color:#7c3aed;">③</span> 2D Reference Cone Geometry
      </div>
      <div class="takeaway-desc">
        Non-Negative Least Squares (NNLS) cone projection into cited references decomposes paper innovation into Closest Parent (m), Multi-Parent Synergy Blend (&Delta;), Subspace Energy (R²₊), and Orthogonal Novelty Residual Gap (&radic;1-R²₊).
      </div>
    </div>
  </div>

  <!-- 3. HERO PIPELINE FIGURE (VECTOR SVG) -->
  <div class="hero-banner">
    <div class="section-tag">
      <div class="section-tag-title">End-to-End System Architecture &amp; Human-in-the-Loop Workflow</div>
      <div class="hero-legend">
        <div class="legend-item"><div class="legend-box legend-human"></div> Human Decision Point</div>
        <div class="legend-item"><div class="legend-box legend-auto"></div> Automated Engine</div>
        <div class="legend-item"><div class="legend-box legend-storage"></div> Structured Store</div>
      </div>
    </div>
    
    <!-- Vector Pipeline SVG -->
    <svg viewBox="0 0 1100 135" style="width: 100%; height: auto; display: block;" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748b" />
        </marker>
        <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
        </marker>
      </defs>

      <!-- Stage 1: Discovery & Catalog -->
      <g transform="translate(10, 10)">
        <rect x="0" y="0" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
        <rect x="8" y="8" width="149" height="22" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1" />
        <text x="82" y="23" font-size="9" font-weight="700" text-anchor="middle" fill="#92400e">Researcher Topic Query</text>
        <text x="82" y="42" font-size="8" text-anchor="middle" fill="#475569">arXiv API &amp; OpenAlex</text>
        <rect x="12" y="52" width="141" height="52" rx="4" fill="#ffffff" stroke="#94a3b8" stroke-width="1" />
        <text x="82" y="67" font-size="8.5" font-weight="700" text-anchor="middle" fill="#1e293b">SQLite FTS5 Catalog</text>
        <text x="82" y="80" font-size="7.5" text-anchor="middle" fill="#2563eb">5,752 Papers Indexed</text>
        <text x="82" y="93" font-size="7" text-anchor="middle" fill="#64748b">Sub-millisecond BM25</text>
      </g>

      <!-- Arrow 1 -> 2 -->
      <line x1="180" y1="67" x2="202" y2="67" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

      <!-- Stage 2: Multi-Signal Ranking -->
      <g transform="translate(208, 10)">
        <rect x="0" y="0" width="180" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
        <rect x="8" y="8" width="164" height="22" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1" />
        <text x="90" y="23" font-size="9" font-weight="700" text-anchor="middle" fill="#92400e">Tunable Signal Presets</text>
        <rect x="12" y="38" width="156" height="66" rx="4" fill="#ffffff" stroke="#2563eb" stroke-width="1" />
        <text x="90" y="53" font-size="8.5" font-weight="700" text-anchor="middle" fill="#1e40af">Multi-Signal Ranker</text>
        <text x="90" y="66" font-size="7.5" text-anchor="middle" fill="#334155">w_emb (0.50) + w_rec (0.20)</text>
        <text x="90" y="77" font-size="7.5" text-anchor="middle" fill="#334155">+ w_cit (0.25) + w_auth (0.05)</text>
        <text x="90" y="94" font-size="7.5" font-weight="700" text-anchor="middle" fill="#059669">101k Citation Graph Hubs</text>
      </g>

      <!-- Arrow 2 -> 3 (Split: Selective Ingest OR Novelty Geometry) -->
      <line x1="393" y1="50" x2="418" y2="35" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-blue)" />
      <line x1="393" y1="85" x2="775" y2="100" stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arr)" />

      <!-- Stage 3: Selective Ingestion -->
      <g transform="translate(424, 10)">
        <rect x="0" y="0" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
        <rect x="8" y="8" width="149" height="22" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1" />
        <text x="82" y="23" font-size="9" font-weight="700" text-anchor="middle" fill="#92400e">Select Papers to Ingest</text>
        <rect x="12" y="38" width="141" height="66" rx="4" fill="#ffffff" stroke="#94a3b8" stroke-width="1" />
        <text x="82" y="53" font-size="8" font-weight="700" text-anchor="middle" fill="#1e293b">PDF / HTML Extraction</text>
        <text x="82" y="66" font-size="7.5" text-anchor="middle" fill="#475569">PyMuPDF4LLM + OCR</text>
        <text x="82" y="79" font-size="7.5" text-anchor="middle" fill="#475569">Header Chunks (1000ch)</text>
        <text x="82" y="93" font-size="7.5" font-weight="700" text-anchor="middle" fill="#2563eb">Chroma RAG Vector Store</text>
      </g>

      <!-- Arrow 3 -> 4 -->
      <line x1="594" y1="67" x2="618" y2="67" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

      <!-- Stage 4: Hybrid RRF RAG & Generation -->
      <g transform="translate(624, 10)">
        <rect x="0" y="0" width="180" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
        <rect x="8" y="8" width="164" height="22" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1" />
        <text x="90" y="23" font-size="9" font-weight="700" text-anchor="middle" fill="#92400e">Researcher Question</text>
        <rect x="12" y="38" width="156" height="66" rx="4" fill="#ffffff" stroke="#059669" stroke-width="1" />
        <text x="90" y="53" font-size="8.5" font-weight="700" text-anchor="middle" fill="#065f46">Hybrid RRF + Section Priors</text>
        <text x="90" y="66" font-size="7.5" text-anchor="middle" fill="#334155">Dense + BM25 Lexical Fusion</text>
        <text x="90" y="79" font-size="7.5" text-anchor="middle" fill="#2563eb">Gemini Grounded Synthesis</text>
        <text x="90" y="94" font-size="7.5" font-weight="700" text-anchor="middle" fill="#059669">98.9% Passage Citations [1][2]</text>
      </g>

      <!-- Arrow 4 -> 5 -->
      <line x1="809" y1="67" x2="833" y2="67" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

      <!-- Stage 5: 2D Novelty Cone Geometry -->
      <g transform="translate(839, 10)">
        <rect x="0" y="0" width="250" height="115" rx="6" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1.2" />
        <rect x="8" y="8" width="234" height="22" rx="3" fill="#ede9fe" stroke="#7c3aed" stroke-width="1" />
        <text x="125" y="23" font-size="8.5" font-weight="700" text-anchor="middle" fill="#5b21b6">2D Novelty Geometry Subspace</text>
        <rect x="12" y="38" width="226" height="66" rx="4" fill="#ffffff" stroke="#7c3aed" stroke-width="1" />
        <text x="125" y="53" font-size="8.5" font-weight="700" text-anchor="middle" fill="#6d28d9">NNLS Reference Cone Projection</text>
        <text x="125" y="66" font-size="7.5" text-anchor="middle" fill="#475569">S2 + OpenAlex + arXiv HTML (706 refs)</text>
        <text x="125" y="79" font-size="7.5" font-weight="700" text-anchor="middle" fill="#0f172a">m=0.774 &nbsp;|&nbsp; &Delta;=0.138 &nbsp;|&nbsp; R²₊=0.737</text>
        <text x="125" y="94" font-size="7.5" font-weight="700" text-anchor="middle" fill="#7c3aed">Interactive HTML5 Canvas :8000</text>
      </g>
    </svg>
  </div>

  <!-- 4. MAIN TWO-COLUMN BODY -->
  <div class="main-columns">

    <!-- LEFT COLUMN: Ranking & Hybrid RAG -->
    <div class="col">

      <!-- PANEL 1: Multi-Signal Ranking -->
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">1. Multi-Signal Literature Discovery</div>
          <div class="panel-kicker">Ranking &amp; 101k Graph Hubs</div>
        </div>
        <p class="lead-text">
          Keyword search in unranked catalogs floods researchers with obsolete or marginally relevant preprints. Our platform computes an inspectable multi-signal utility score balancing dense semantics, time decay, citation authority, and author continuity:
        </p>
        
        <div class="formula-box">
          Score(q, p) = w_emb · S_emb(q, p) + w_rec · 2^(-&Delta;t / t_half) + w_cit · (log(1+C) / log(1+C_max)) + w_auth · S_auth(p)
        </div>
        <div class="formula-explanation">
          Where t_half = 4.0 yr exponential decay, C = global citation count, and S_auth measures domain author frequency.
        </div>

        <!-- Presets Table & Visual Bars -->
        <table class="preset-table">
          <thead>
            <tr>
              <th style="width: 28%;">Weight Preset</th>
              <th style="width: 15%;">Formula</th>
              <th style="width: 57%;">Signal Distribution (w_emb | w_rec | w_cit | w_auth)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Balanced</strong> (Default)</td>
              <td>0.50, 0.20, 0.25, 0.05</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-seg seg-emb" style="width: 50%;">50% Emb</div>
                  <div class="bar-seg seg-rec" style="width: 20%;">20% Rec</div>
                  <div class="bar-seg seg-cit" style="width: 25%;">25% Cit</div>
                  <div class="bar-seg seg-auth" style="width: 5%;">5%</div>
                </div>
              </td>
            </tr>
            <tr>
              <td><strong>Precision</strong> (Focus)</td>
              <td>0.65, 0.10, 0.20, 0.05</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-seg seg-emb" style="width: 65%;">65% Emb</div>
                  <div class="bar-seg seg-rec" style="width: 10%;">10%</div>
                  <div class="bar-seg seg-cit" style="width: 20%;">20% Cit</div>
                  <div class="bar-seg seg-auth" style="width: 5%;">5%</div>
                </div>
              </td>
            </tr>
            <tr>
              <td><strong>Prefer Recent</strong></td>
              <td>0.45, 0.35, 0.15, 0.05</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-seg seg-emb" style="width: 45%;">45% Emb</div>
                  <div class="bar-seg seg-rec" style="width: 35%;">35% Recency</div>
                  <div class="bar-seg seg-cit" style="width: 15%;">15%</div>
                  <div class="bar-seg seg-auth" style="width: 5%;">5%</div>
                </div>
              </td>
            </tr>
            <tr>
              <td><strong>Citation Heavy</strong></td>
              <td>0.35, 0.10, 0.50, 0.05</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar-seg seg-emb" style="width: 35%;">35% Emb</div>
                  <div class="bar-seg seg-rec" style="width: 10%;">10%</div>
                  <div class="bar-seg seg-cit" style="width: 50%;">50% Citations</div>
                  <div class="bar-seg seg-auth" style="width: 5%;">5%</div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>

        <div style="margin-top: 2.2mm; font-size: 7.4pt; color: #475569; line-height: 1.3;">
          <strong>101k Citation Graph Centrality:</strong> Integrates <code>data/citation_library/graph.sqlite3</code> (101,716 papers, 226,857 edges) computing in-library landmark degree hubs (e.g. <em>Attention Is All You Need</em>, <em>ResNet</em>, <em>Adam</em>) to bias discovery toward foundational literature.
        </div>
      </div>

      <!-- PANEL 2: Selective Ingestion & Hybrid RAG -->
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">2. Selective Ingestion &amp; Hybrid RRF RAG</div>
          <div class="panel-kicker">High-Precision Full-Text Grounding</div>
        </div>
        <p class="lead-text">
          Mass uncurated indexing degrades dense retrieval precision. Our workflow prompts the researcher to rank abstracts first, selectively downloading and ingesting only high-relevance full texts into Chroma:
        </p>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2.5mm; margin: 1.5mm 0;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2mm 2.5mm; border-radius: 4px;">
            <div style="font-weight: 700; font-size: 7.5pt; color: #1e3a8a; margin-bottom: 1mm;">Section-Aware Parsing</div>
            <div style="font-size: 7.2pt; color: #334155; line-height: 1.3;">
              PDFs/HTML converted via <strong>PyMuPDF4LLM</strong> with OCR fallback. Split into 1,000 character target chunks (180 overlap) retaining markdown headings (<code>## Methodology</code>, <code>### Results</code>).
            </div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2mm 2.5mm; border-radius: 4px;">
            <div style="font-weight: 700; font-size: 7.5pt; color: #059669; margin-bottom: 1mm;">Hybrid RRF &amp; Section Priors</div>
            <div style="font-size: 7.2pt; color: #334155; line-height: 1.3;">
              Combines dense embeddings (<code>google/embeddinggemma-2</code>) and BM25 lexical matches via Reciprocal Rank Fusion. Section-intent priors boost relevant sections based on query phrasing.
            </div>
          </div>
        </div>

        <div class="formula-box" style="font-size: 8.5pt;">
          RRF_Score(d) = (w_dense / (60 + rank_dense(d))) + (w_lex / (60 + rank_lex(d))) + Prior(section, q)
        </div>

        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 4px; padding: 2.2mm 3mm; margin-top: 2mm;">
          <div style="font-size: 7.8pt; font-weight: 700; color: #166534; margin-bottom: 0.8mm;">
            Passage-Level Verifiable Synthesis (Gemini 2.0 / 3.8 Flash)
          </div>
          <div style="font-size: 7.3pt; color: #14532d; line-height: 1.3;">
            The generator prompt strictly forces answers to anchor statements to exact passage numbers: <code>[1], [2]</code>. ConfAI evaluation measures <strong>98.9% citation validity</strong> (89 of 90 cited links verifiable against retrieved text chunks).
          </div>
        </div>
      </div>

    </div>

    <!-- RIGHT COLUMN: 2D Novelty Geometry & ConfAI Benchmarks -->
    <div class="col">

      <!-- PANEL 3: 2D Novelty Cone Geometry -->
      <div class="panel-card" style="border-color: #d8b4fe;">
        <div class="panel-header" style="border-bottom-color: #f3e8ff;">
          <div class="panel-title" style="color: #5b21b6;">3. 2D Reference Cone Geometry</div>
          <div class="panel-kicker" style="color: #7c3aed;">Subspace Innovation Decomposition</div>
        </div>
        <p class="lead-text">
          Does a new paper truly break ground, synthesize disparate lineages, or simply make an incremental step? We project normalized paper vector <strong>x</strong> into the convex cone of its cited references <strong>Y = [y_1, ..., y_k]</strong> via Non-Negative Least Squares:
        </p>

        <div class="formula-box" style="border-left-color: #7c3aed;">
          &lambda;* = argmin_{{&lambda; &ge; 0}} || x - Y &lambda; ||&sup2;, &nbsp;&nbsp;&nbsp; x&#770; = Y &lambda;*, &nbsp;&nbsp;&nbsp; R&sup2;_+ = || x&#770; ||&sup2; &in; [0, 1]
        </div>

        <!-- 2D Canvas Diagram + Math Breakdown -->
        <div class="novelty-grid">
          <div>
            <div style="font-size: 7.4pt; color: #334155; line-height: 1.3;">
              <strong>Orthonormal Canvas Basis:</strong><br>
              • <strong>X-axis (e_1):</strong> Aligned to closest parent y_{{m*}}, plotted at (1, 0).<br>
              • <strong>Y-axis (e_2):</strong> Orthogonal synergy vector u / ||u||.<br>
              • <strong>Synergy Blend gain:</strong> &Delta; = R&sup2;_+ - max(m, 0)&sup2;.<br>
              • <strong>Novelty Residual:</strong> Res = &radic;max(0, 1 - R&sup2;_+).
            </div>
            
            <div class="metric-pills">
              <div class="metric-pill">
                <div class="metric-pill-label">Anchor (m)</div>
                <div class="metric-pill-val">0.774</div>
              </div>
              <div class="metric-pill">
                <div class="metric-pill-label">Synergy (&Delta;)</div>
                <div class="metric-pill-val">0.138</div>
              </div>
              <div class="metric-pill">
                <div class="metric-pill-label">Subspace (R&sup2;₊)</div>
                <div class="metric-pill-val">0.737</div>
              </div>
              <div class="metric-pill">
                <div class="metric-pill-label">Residual Gap</div>
                <div class="metric-pill-val">0.513</div>
              </div>
            </div>
          </div>

          <!-- Vector SVG of 2D Novelty Cone -->
          <svg viewBox="0 0 210 130" style="width: 100%; height: auto; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 5px;">
            <!-- Unit Circle Arc -->
            <path d="M 30,110 A 80,80 0 0 0 110,30" fill="none" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="3,2" />
            <text x="75" y="45" font-size="6" fill="#94a3b8">Unit Hypersphere (||x||=1)</text>

            <!-- Axes -->
            <line x1="30" y1="110" x2="135" y2="110" stroke="#475569" stroke-width="1.5" />
            <line x1="30" y1="110" x2="30" y2="15" stroke="#475569" stroke-width="1.5" />
            <text x="138" y="113" font-size="6.5" font-weight="700" fill="#1e293b">e_1 (Dominant Ancestor)</text>
            <text x="32" y="12" font-size="6.5" font-weight="700" fill="#1e293b">e_2 (Synergy Axis)</text>

            <!-- Origin -->
            <circle cx="30" cy="110" r="2.5" fill="#475569" />

            <!-- Closest Parent (1, 0) at x=110, y=110 -->
            <circle cx="110" cy="110" r="4.5" fill="#7c3aed" />
            <text x="110" y="122" font-size="6.5" font-weight="700" text-anchor="middle" fill="#7c3aed">y_{{m*}} (1, 0)</text>

            <!-- Other Active Parent Dots -->
            <circle cx="78" cy="72" r="3" fill="#9333ea" opacity="0.8" />
            <circle cx="62" cy="85" r="3" fill="#9333ea" opacity="0.8" />
            <circle cx="95" cy="98" r="2.2" fill="#94a3b8" opacity="0.6" />
            <circle cx="85" cy="102" r="2.2" fill="#94a3b8" opacity="0.6" />

            <!-- Reconstruction vector x_hat: x = 30 + 0.774*80 = 92, y = 110 - sqrt(0.138)*80 = 110 - 29.7 = 80 -->
            <line x1="30" y1="110" x2="92" y2="80" stroke="#0891b2" stroke-width="2" />
            <circle cx="92" cy="80" r="4.5" fill="#0891b2" />
            <circle cx="92" cy="80" r="7" fill="none" stroke="#0891b2" stroke-width="1" opacity="0.5" />
            <text x="96" y="76" font-size="7" font-weight="800" fill="#0e7490">x&#770; (Blend)</text>

            <!-- Orthogonal Residual Line to unit circle -->
            <line x1="92" y1="80" x2="106" y2="48" stroke="#d97706" stroke-width="1.5" stroke-dasharray="2,2" />
            <circle cx="106" cy="48" r="3" fill="#d97706" />
            <text x="111" y="47" font-size="6" font-weight="700" fill="#b45309">Residual (&radic;1-R&sup2;₊)</text>
          </svg>
        </div>

        <div style="margin-top: 2mm; font-size: 7.2pt; color: #475569; line-height: 1.3;">
          <strong>Real-World Discovery Validation:</strong> Evaluated on <em>BIG-bench</em> (arXiv:2206.04615). Our multi-tier citation resolver parsed <strong>706 references</strong> directly from arXiv HTML; NNLS attribution correctly identified <em>BERT</em> (arXiv:1810.04805) as the top foundational ancestor with <strong>24.9% attribution share</strong>.
        </div>
      </div>

      <!-- PANEL 4: ConfAI 2026 Evaluation & Scaling Sweeps -->
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">4. ConfAI 2026 Empirical Evaluation</div>
          <div class="panel-kicker">Scaling Laws &amp; Retrieval Quality</div>
        </div>
        <p class="lead-text">
          Evaluating on a fixed 100-question gold benchmark reveals the <strong>Index Scaling Law</strong>: pure dense retrieval degrades drastically as uncurated documents are added, establishing the necessity of our selective-ingest paradigm:
        </p>

        <!-- Benchmark Table -->
        <table class="data-table">
          <thead>
            <tr>
              <th>Ingested Corpus Scale</th>
              <th>Papers (N)</th>
              <th>Chunks</th>
              <th>Mean MRR</th>
              <th>95% Bootstrap CI</th>
              <th>Recall@k</th>
              <th>Hit@1</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight">
              <td><strong>Curated Subspace</strong></td>
              <td>10</td>
              <td>~1,725</td>
              <td><strong>0.786</strong></td>
              <td>[0.702, 0.863]</td>
              <td><strong>0.888</strong></td>
              <td><strong>0.725</strong> <span class="badge-win">Peak</span></td>
            </tr>
            <tr>
              <td>Topic Neighborhood</td>
              <td>100</td>
              <td>~9,800</td>
              <td>0.757</td>
              <td>[0.658, 0.842]</td>
              <td>0.825</td>
              <td>0.713</td>
            </tr>
            <tr>
              <td>Domain Corpus</td>
              <td>1,000</td>
              <td>~98,000</td>
              <td>0.638</td>
              <td>[0.546, 0.740]</td>
              <td>0.725</td>
              <td>0.588</td>
            </tr>
            <tr>
              <td>Full Uncurated Library</td>
              <td>5,663</td>
              <td>551,743</td>
              <td>0.457</td>
              <td>[0.360, 0.549]</td>
              <td>0.638</td>
              <td>0.350 <span class="badge-drop">-51.7%</span></td>
            </tr>
          </tbody>
        </table>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2.5mm; margin-top: 2.2mm;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2mm 2.5mm; border-radius: 4px;">
            <div style="font-weight: 700; font-size: 7.4pt; color: #1e3a8a;">Hybrid RRF Boost</div>
            <div style="font-size: 7.1pt; color: #334155;">
              Adding BM25 reciprocal rank fusion and section priors lifts dense MRR from <strong>0.574 &rarr; 0.612</strong> (+6.6%), directly recovering exact alphanumeric identifier queries.
            </div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2mm 2.5mm; border-radius: 4px;">
            <div style="font-weight: 700; font-size: 7.4pt; color: #059669;">Attribution Integrity</div>
            <div style="font-size: 7.1pt; color: #334155;">
              <strong>98.9% citation validity</strong> (89/90 cited statements verified). Rejection recall reaches <strong>65.0%</strong> on out-of-scope unanswerable questions.
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- 5. FOOTER & LIMITS -->
  <footer class="footer-strip">
    <div class="footer-left">
      <div class="limits-box">
        <strong>Honest Operational Limits (AI in Practice):</strong> (1) Refusal calibration remains an open boundary: conservative safety gating leads to 23.8% false refusals on marginal contexts. (2) Alphanumeric IDs and equations underperform under pure dense retrieval without BM25 hybrid assistance. (3) 2B-parameter local embeddings require proactive Metal cache draining on Apple Silicon.
      </div>
      <div class="refs-line">
        <strong>Key References:</strong> [1] Vaswani et al., <em>Attention Is All You Need</em>, NeurIPS 2017. [2] Jacot et al., <em>Neural Tangent Kernel</em>, NeurIPS 2018. [3] Cormack et al., <em>Reciprocal Rank Fusion</em>, SIGIR 2009. [4] Srivastava et al., <em>BIG-bench</em>, arXiv:2206.04615, 2022.
      </div>
    </div>
    
    <div class="footer-qrs">
      <div class="qr-block">
        <div class="qr-svg">{gh_svg}</div>
        <div class="qr-label">Inspect Code (GitHub)</div>
      </div>
      <div class="qr-block">
        <div class="qr-svg">{demo_svg}</div>
        <div class="qr-label">Quick Start &amp; CLI</div>
      </div>
    </div>
  </footer>

</div>

</body>
</html>
"""

out_html = DIR / "poster.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated A1 Poster HTML: {out_html} ({len(html_content)} bytes)")

# Compile to PDF using headless Chrome if available
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out_pdf = DIR / "poster.pdf"

if Path(chrome_path).exists():
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={out_pdf}",
        "--print-to-pdf-no-header",
        str(out_html),
    ]
    print(f"Compiling vector PDF via headless Chrome: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_pdf.exists():
        print(f"Successfully compiled vector PDF: {out_pdf} ({out_pdf.stat().st_size} bytes)")
    else:
        print(f"PDF compilation returned non-zero code {res.returncode}: {res.stderr}")
else:
    print(f"Chrome binary not found at {chrome_path}. Manual export instruction provided.")
