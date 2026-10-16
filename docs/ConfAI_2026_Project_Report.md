# Literature Discovery with Multi-Signal Ranking and Citation-Grounded RAG

**A practical system for personalised arXiv recommendation and grounded question answering**

---

| | |
|---|---|
| **Author** | Ajay Sharma |
| **Programme** | M.S. in Artificial Intelligence, Plaksha University |
| **Supervisor** | Dr. Subrat Panda |
| **Course context** | AI in Practice — extended track project |
| **Venue** | ConfAI 2026 (poster), Plaksha University, Mohali · 30 Oct – 1 Nov 2026 |
| **Suggested tracks** | Recent Advances in AI; Responsible and Ethical AI |
| **Live demo** | https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/ |
| **Repository** | https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG |

**Purpose of this report.** A concise technical summary for faculty review—including discussion of possible course credit—and a foundation for a ConfAI 2026 poster presentation under the guidance of **Dr. Subrat Panda**.

---

## 1. Abstract

Finding the right papers is only half of literature work; the other half is *using* those papers to answer precise questions without unsupported claims. This project delivers an end-to-end system that (i) **ranks** arXiv papers for a free-text research interest using multiple signals—semantic similarity, recency, citation impact, and author continuity—and (ii) supports **retrieval-augmented generation (RAG)** over full-text content with explicit paper-level citations.

Embeddings are computed **locally** (EmbeddingGemma 2 on capable machines; a lighter MiniLM path on constrained cloud hosts). Answer generation uses Gemini, with a failsafe path when the primary API is unavailable. Users interact through a **command-line interface** (batch jobs, diagnosis) and a **Streamlit web application** (interactive discovery and Q&A). Evaluation follows evaluation-driven development from the AI in Practice sequence: a gold question set on the order of 100 items, retrieval metrics (MRR, Recall@k, Hit@1) with bootstrap confidence intervals, Matryoshka dimension sweeps, latency, and refusal behaviour.

On a curated full-text index, representative runs reached **mean MRR ≈ 0.75** (95% CI approximately 0.66–0.83) and **Recall@k ≈ 0.85** on labelled items. Quality is strong when relevant papers are ingested; remaining gaps—especially refusal calibration and scale of full-text coverage—are measured and discussed rather than hidden.

---

## 2. Motivation

### 2.1 The practical gap

Academic search engines return long lists. General-purpose chat models answer fluently but often without faithful grounding. Researchers and students need a middle path:

1. **Discover** papers aligned with a research narrative, not only exact keyword overlap.  
2. **Select and ingest** a manageable set of full texts into a private index.  
3. **Ask** focused questions and receive answers **tied to passages**, with clear limits when evidence is thin.

### 2.2 Connection to AI in Practice

Under the supervision of **Dr. Subrat Panda**, the project was framed as an extension of an existing arXiv recommender into a full pipeline through RAG and measurement—matching the course arc through Labs focused on retrieval, generation, systems concerns (caching, latency, operational discipline), and evaluation-driven development (metrics, regression thinking, confidence intervals).

| Course theme | Manifestation in this system |
|--------------|------------------------------|
| Data pipelines | arXiv API → corpus JSON → dual Chroma indexes |
| Embeddings & search | Local EmbeddingGemma 2; optional dimension truncation |
| Ranking beyond cosine | Weighted fusion of semantic, temporal, citation, and author signals |
| RAG with citations | Chunk retrieval → constrained generation with `[n]` references |
| Systems practice | CLI, caching, resumable ingest, environment-aware deployment |
| Evaluation | Gold set, MRR/Recall, CIs, dim sweep, refusal analysis |

### 2.3 Fit with ConfAI 2026

[ConfAI 2026](https://www.conf-ai.in/) (Plaksha University, 30 October – 1 November 2026) emphasises dialogue between research and practice across tracks such as Recent Advances in AI and Responsible AI. This work is a **deployed student system**: architecture, metrics, and live demo—not a notebook-only prototype—suitable for a **poster** and for faculty discussion of applied competence.

---

## 3. System design

### 3.1 Architecture at a glance

```text
  Research interest (natural language)
              │
              ▼
  ┌─────────────────────────────────────┐
  │  Recommend index (library)          │
  │  Title + abstract embeddings         │
  │  + recency / citation / author      │
  └─────────────────┬───────────────────┘
                    │ top-k arXiv ids
                    ▼
  ┌─────────────────────────────────────┐
  │  On-demand full-text ingest         │
  │  PDF → Markdown (+ OCR if needed)   │
  │  → section-aware chunks → embed     │
  └─────────────────┬───────────────────┘
                    ▼
  ┌─────────────────────────────────────┐
  │  RAG index                          │
  │  Retrieve top passages → generate   │
  │  Answer + citations + coverage note │
  └─────────────────────────────────────┘
```

**Central design choice.** The system does not attempt to full-text index all of arXiv. A **library** holds metadata and abstract-level vectors for ranking; **full text is ingested only for papers the user selects**. That matches how literature work actually proceeds and keeps laptop-scale hardware viable.

### 3.2 Multi-signal recommendation

For a free-text interest profile, each library paper receives a composite score from:

- **Semantic similarity** between the interest and title/abstract embeddings  
- **Recency** (emphasis when the user prioritises frontier work)  
- **Citation / impact** signals when available  
- **Author** continuity (optional preference for known lineages)

Weight **presets** (e.g. Balanced, Semantic-heavy, Frontier) ship with numeric values shown to two decimals. If the user moves sliders away from a preset, the UI switches to **Custom** so ranking remains inspectable rather than opaque.

### 3.3 Full-text RAG path

1. **Download** the PDF from arXiv (disk cache; optional force-refresh).  
2. **Convert** to Markdown (pymupdf4llm), with Tesseract OCR when the text layer is insufficient.  
3. **Chunk** with section awareness and overlap suitable for passage retrieval.  
4. **Embed** with EmbeddingGemma 2 (native 768 dimensions; Matryoshka truncation to 512 / 256 / 128).  
5. **Index** in a dedicated Chroma collection with paper id, title, and section metadata.  
6. **Answer:** embed the question → retrieve top-k chunks → generate with Gemini under instructions to cite sources and acknowledge weak coverage; optional semantic cache for repeated questions.

### 3.4 Interfaces and tooling

| Layer | Role |
|-------|------|
| **CLI** (`uv run python -m src.rag.cli …`) | Ingest, ask, status, evaluation; reproducible automation |
| **Streamlit application** | Library growth, recommend, ingest, Ask UI, activity log, help |
| **Corpus builder** | Expand the library from keywords or explicit arXiv ids (merge, not wipe) |
| **Environment** | **uv** for dependency lockfiles and `uv run` execution |

---

## 4. Design rationale

**Local embeddings, API generation.** Vectorisation is frequent and bulk; it is kept local for cost and control. Generation is less frequent and benefits from a strong instruction-following model (Gemini), with a failsafe when quotas or outages intervene.

**Hybrid deployment.** Capable local machines use EmbeddingGemma 2. Streamlit Cloud uses a smaller embedder so free-tier memory limits do not break the public demo. Behaviour is selected by settings, not by forking the product.

**Trust by construction.** Answers carry paper-linked citations and source context. Coverage cues discourage over-confidence when retrieval is weak. Ranking weights are visible. Evaluation reports failure modes alongside headline scores.

---

## 5. Evaluation

### 5.1 Protocol

Evaluation is treated as part of the product, not an afterthought:

- **Gold questions:** on the order of **100**, spanning definitions, methods, open problems, comparisons, identifiers, multi-hop items, and deliberately unanswerable cases.  
- **Retrieval metrics:** MRR, Recall@k, Hit@1, mean top similarity; breakdowns by question kind where informative.  
- **Uncertainty:** bootstrap **confidence intervals** on MRR.  
- **Systems metrics:** latency percentiles; cache behaviour when generation is enabled.  
- **Behavioural metrics:** refusal precision and recall.  
- **Embedding dimension sweep:** 128 / 256 / 512 / 768 under Matryoshka truncation.

### 5.2 Representative results

On a full-text index built from a focused set of ingested papers (order of ten papers in early controlled runs; library and ingest capacity later scaled substantially):

| Metric | Approximate result |
|--------|--------------------|
| Mean MRR | **0.75** (95% CI ≈ **0.66–0.83**) |
| Recall@k | **0.85** |
| Hit@1 | ≈ **0.69** |
| Dimension sweep | **512-d** often matched or slightly exceeded full 768-d MRR (Δ ≲ 0.02) |

**Reading the numbers.** When the right papers are in the RAG index, retrieval is strong enough for grounded Q&A demos and coursework. Truncating to 512 dimensions is a practical default for storage and speed with little measured loss.

### 5.3 Failure modes (explicit)

1. **Missing full text** — A paper can rank highly in the library yet yield weak RAG answers if it was never ingested; ranking and reading indexes must stay coupled in the user’s workflow.  
2. **Refusal over-triggering** — Early generation runs showed refusal precision below a conservative ~0.7 bar: the model sometimes declined answerable questions. Prompting and coverage thresholds are the primary remedies.  
3. **Corpus noise** — Very broad keyword harvests dilute recommendation quality; category-aware and shorter arXiv queries improve precision.  
4. **Operational scale** — Full-text ingest over thousands of PDFs requires resumable, memory-aware jobs so overnight runs can stop and continue without corruption.

Stating these limits is intentional: ConfAI and faculty review both benefit from measured honesty.

---

## 6. Engineering under constraints

- **Resumable ingest** with progress files, skip of completed ids, and **hard per-paper timeouts** so stuck OCR cannot freeze a long job.  
- **Memory policy** that reserves buffer for the operating system and embedding peaks, and adapts batch size to available RAM.  
- **Dual interface** so the same system is diagnosable from a terminal and presentable in a browser.  
- **Reproducible environments** via uv lockfiles and documented entry points.

---

## 7. Contributions

As an applied MS project, the main contributions are integrative and operational:

1. A **closed loop** from interest → ranked library → selective full-text ingest → cited answers.  
2. **Multi-signal ranking** with inspectable weights, beyond pure embedding similarity.  
3. **Citation-grounded RAG** with coverage-aware generation.  
4. An **evaluation suite** aligned with AI in Practice (metrics, intervals, dimension study, refusal).  
5. **Deployed** CLI and web paths, including a public Streamlit demo under real resource limits.

---

## 8. Future work

- Tighten **refusal calibration** so uncertainty remains visible without excessive non-answers.  
- Add **hybrid retrieval** (lexical + dense) and optional cross-encoder re-ranking.  
- Exploit EmbeddingGemma 2’s **shared image–text space** for figure retrieval (and optional multimodal generation).  
- Improve **code-region chunking** for algorithmic papers.  
- Collect light **user feedback** on recommendations to personalise weights.  
- Freeze gold sets against **versioned index snapshots** for regression gates.

---

## 9. Conclusion

This project implements a practical literature assistant: multi-signal arXiv recommendation, on-demand full-text ingest, and citation-grounded RAG, with both programmatic and interactive interfaces and an evaluation stack suitable for an AI engineering course. Under the supervision of **Dr. Subrat Panda**, it was developed as an extended AI in Practice track project. Retrieval quality is strong when relevant papers are indexed; open issues are quantified and actionable.

For **ConfAI 2026**, the work supports a clear **poster** narrative—problem, architecture, metrics, demo, limitations. For **faculty review and possible course credit**, it demonstrates end-to-end ability across data, models, systems, and evaluation.

---

## References (selected)

1. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.  
2. Jacot et al., *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*, arXiv:1806.07572.  
3. Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, arXiv:2304.14997.  
4. Sharkey et al., *Open Problems in Mechanistic Interpretability*, arXiv:2501.16496.  
5. Google DeepMind, *EmbeddingGemma 2* (open multimodal embedding model, 2026).  
6. ConfAI 2026, Plaksha University — https://www.conf-ai.in/

---

## Appendix A — Suggested poster layout

| Panel | Content |
|-------|---------|
| 1 | Problem: discovery vs grounded use of literature |
| 2 | Architecture diagram (library → ingest → RAG) |
| 3 | Multi-signal ranking and inspectable weights |
| 4 | RAG pipeline (PDF → Markdown → chunk → embed → cite) |
| 5 | Metrics table (MRR, Recall, CI, dim sweep) |
| 6 | Live demo QR + one-line CLI examples |
| 7 | Limitations and next steps |

## Appendix B — Faculty summary (short)

> Ajay Sharma (M.S. AI, Plaksha University), supervised by **Dr. Subrat Panda**, built an end-to-end literature system that combines multi-signal arXiv recommendation with citation-grounded RAG over full-text PDFs. The system uses local EmbeddingGemma 2 embeddings, Gemini generation, dual CLI/web interfaces, and an evaluation suite with gold questions, MRR/Recall, confidence intervals, and embedding-dimension sweeps. Representative retrieval results reach approximately 0.75 MRR and 0.85 Recall@k when relevant papers are ingested. The work extends the AI in Practice sequence and is proposed for a ConfAI 2026 poster presentation.

---

*Version: October 2026 · Prepared for faculty discussion (Dr. Subrat Panda) and ConfAI 2026 poster preparation*
