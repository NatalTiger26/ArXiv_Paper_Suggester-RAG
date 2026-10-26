# Literature Discovery with Multi-Signal Ranking and Citation-Grounded RAG

**Find papers for your research interest. Rank them with inspectable signals. Ingest full text only for what you select. Ask questions and get answers with citations.**

[![Live demo](https://img.shields.io/badge/Streamlit-Live%20demo-FF4B4B?logo=streamlit&logoColor=white)](https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue?logo=python&logoColor=white)](#prerequisites)
[![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](#install-uv)

**ConfAI 2026** poster project · Plaksha University · M.S. in AI · Supervisor: **Dr. Subrat Panda** · AI in Practice extended track

| | |
|---|---|
| **Live demo** | [Streamlit app](https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/) |
| **Repository** | [github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG](https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG) |

---

## Why this exists

Literature work fails in two common ways:

1. **Discovery** — search returns long lists that are not ranked for *your* research narrative.
2. **Grounding** — chat models answer fluently but often without faithful links to passages you can verify.

This project closes the loop:

1. **Discover** papers from arXiv (keywords or explicit IDs).
2. **Rank** them for a free-text interest (semantic similarity + recency + citation impact + author continuity).
3. **Ingest** full text only for papers you choose (PDF → Markdown → chunks → local embeddings).
4. **Ask** questions over those chunks and receive answers with **paper-level citations** and a **coverage** cue when evidence is thin.

Built for real laptops and demos: **CLI** for batch work and diagnosis, **Streamlit** for interactive use, reproducible **`uv`** environments.

---

## What you get

| Capability | Entry point |
|------------|-------------|
| Build a paper library (metadata) | `uv run python src/corpus_builder.py search` / `from-ids` |
| Multi-signal recommendation | `uv run python src/recommend.py` or UI **Recommend** |
| Full-text ingest | `uv run python -m src.rag.cli ingest` or UI |
| Q&A with citations | `uv run python -m src.rag.cli ask` or UI **Ask** |
| Web app | `uv run python -m streamlit run src/app.py` |
| Local health check | `bash scripts/check_local.sh` or `make health` |
| RAG evaluation | `uv run python src/final_eval.py` → `results/EVALUATION_REPORT.md` |

**Important:** **Library ≠ full-text RAG.** Ranking uses titles and abstracts. **Ask** only sees papers you have **ingested**.

---

## Architecture

```text
                         arXiv API
                             │
                             ▼
                    ┌─────────────────┐
                    │ corpus_builder  │──► data/corpus.json
                    │                 │──► data/chroma_corpus   (recommend index)
                    └────────┬────────┘
                             │ interest text
                             ▼
                    ┌─────────────────┐
                    │ ranking /       │──► ranked arXiv IDs
                    │ recommend       │
                    └────────┬────────┘
                             │ selected IDs
                             ▼
                    ┌─────────────────┐
                    │ download PDF    │
                    │ pdf → Markdown  │  (pymupdf4llm + OCR fallback)
                    │ section chunk   │
                    │ embed + Chroma  │──► data/chroma_rag
                    └────────┬────────┘
                             │ question
                             ▼
                    ┌─────────────────┐
                    │ retrieve chunks │
                    │ Gemini + failsafe│──► answer + coverage + citations
                    └─────────────────┘
```

### Two indexes, two jobs

| Index | Path | Content | Used by |
|-------|------|---------|---------|
| **Recommend** | `data/chroma_corpus` | Title + abstract embeddings | Rank / recommend |
| **RAG** | `data/chroma_rag` | Full-text chunks | Ask / evaluation |

Changing **embedding model or dimension** requires rebuilding the affected index(es). Vectors from different models or dims are not interchangeable.

### Design choice

Only **abstracts** are indexed library-wide (cheap ranking at scale). **Full text** is ingested on demand so a laptop remains viable and RAG stays focused on papers the user actually selected.

---

## Stack

| Layer | Local (laptop) | Streamlit Community Cloud |
|--------|----------------|---------------------------|
| **Embeddings** | `google/embeddinggemma-2` (optional Matryoshka dim) | Auto **MiniLM** (`all-MiniLM-L6-v2`) for free-tier RAM |
| **Vector store** | Chroma under `data/` | Same layout; ephemeral disk |
| **Generation** | Gemini (`GEMINI_MODEL`) | Same; set Secrets |
| **Failsafe LLM** | OpenAI-compatible endpoint (e.g. NVIDIA) | Optional |
| **Packages** | **[uv](https://docs.astral.sh/uv/)** (`uv sync` / `uv run`) | `uv.lock` / `requirements.txt` |

> Prefer **`uv run …`** over bare `pip` / global `streamlit`.

---

## Repository layout

```text
ArXiv_Paper_Suggester/
├── pyproject.toml            # uv project + dependencies
├── requirements.txt          # mirror for Streamlit Cloud
├── packages.txt              # apt: tesseract-ocr (Cloud OCR)
├── uv.lock                   # locked deps when present
├── .env.example              # copy → .env (never commit secrets)
├── .streamlit/               # theme / Cloud config
├── README.md
├── WORKFLOW.md               # short CLI path
├── DEPLOY.md                 # Cloud checklist
├── Makefile
├── scripts/
│   ├── check_local.sh
│   ├── reingest_overnight.sh
│   └── reingest_stable.py
├── data/                     # runtime (usually gitignored)
│   ├── corpus.json
│   ├── chroma_corpus/
│   ├── chroma_rag/
│   ├── papers/
│   ├── markdown/
│   ├── embeddings/
│   ├── rag_cache/
│   └── rag_ingest_progress/
├── rankings/
├── results/                  # EVALUATION_REPORT.md, rag_eval/
├── docs/                     # ConfAI notes
├── notebooks/
└── src/
    ├── settings.py
    ├── corpus_builder.py
    ├── corpus_index.py
    ├── ranking.py
    ├── recommend.py
    ├── evaluate_recommender.py
    ├── final_eval.py
    ├── app.py
    └── rag/
        ├── cli.py
        ├── download.py
        ├── pdf_to_md.py
        ├── chunk.py
        ├── embed_store.py
        ├── ingest.py
        ├── rag_pipeline.py
        ├── cache.py
        └── evaluate.py
```

---

## Prerequisites

- Python **3.10–3.12**
- [uv](https://docs.astral.sh/uv/)
- Optional: Tesseract for OCR on scanned PDFs
- Optional: `GEMINI_API_KEY` for generation (retrieval works without it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Install

```bash
git clone https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG.git
cd ArXiv_Paper_Suggester-RAG

uv sync

cp .env.example .env
# Edit .env — GEMINI_API_KEY for Ask generation
```

```bash
uv run python -c "import streamlit, chromadb; print('ok')"
bash scripts/check_local.sh
```

---

## Configuration

| Variable | Role |
|----------|------|
| `GEMINI_API_KEY` | Primary generation |
| `GEMINI_MODEL` | e.g. `gemini-3.8-flash` (must match your API access) |
| `EMBEDDING_MODEL` | Default `google/embeddinggemma-2` |
| `EMBEDDING_DIM` | `0` = full; or `512` / `256` / `128` (rebuild indexes after change) |
| `FALLBACK_LLM_*` | OpenAI-compatible failsafe |
| `HF_TOKEN` | Optional Hugging Face rate limits |

**Streamlit Cloud:** set the same keys under **Secrets**.

---

## Quick start

```bash
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 15 --build-index
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python -m src.rag.cli ingest 2304.14997 2501.16496 1806.07572
uv run python -m src.rag.cli ask "What is circuit discovery in mechanistic interpretability?"
uv run python -m streamlit run src/app.py
```

```bash
uv run python src/corpus_builder.py status
uv run python -m src.rag.cli status
```

---

## End-to-end workflow

Full detail: **[WORKFLOW.md](WORKFLOW.md)**.

### Library

```bash
uv run python src/corpus_builder.py search "vision-language model interpretability" -n 30 --build-index
uv run python src/corpus_builder.py from-ids 1610.02391 2304.14997 --build-index
uv run python src/corpus_builder.py rebuild-index
```

### Rank

```bash
uv run python src/recommend.py "attention faithfulness multimodal" -k 10
uv run python src/recommend.py "Grad-CAM" -k 8 --prefer-recent
```

Weights (semantic, recency, citation, author) are inspectable; the UI exposes presets (Balanced / Semantic-heavy / Frontier / Custom).

### Ingest

```bash
uv run python -m src.rag.cli ingest 2304.14997 2501.16496
uv run python -m src.rag.cli status
```

Large / overnight (resumable):

```bash
bash scripts/reingest_overnight.sh
# Ctrl+C finishes the current batch; resume with the same command.
# Use FRESH=1 only when intentionally resetting the RAG index.
```

Existing PDFs and Markdown are reused when present.

### Ask

```bash
uv run python -m src.rag.cli ask "How does Grad-CAM produce visual explanations?"
```

Coverage notes are **heuristic**, not calibrated probabilities.

---

## Web UI

```bash
uv run python -m streamlit run src/app.py
```

Flow: search/add → rank → ingest → ask with citations.

Public Cloud demo uses a **lighter embedder** and a **smaller library** than a full local run (free-tier RAM). See **[DEPLOY.md](DEPLOY.md)**.

---

## Embeddings

- Local default: **EmbeddingGemma 2** (768-d).
- Optional `EMBEDDING_DIM` (e.g. 512) for speed/memory; eval dim-sweeps often found 512 close to full MRR on the gold set.
- Cloud may force MiniLM (384-d). Do not mix models/dims in one Chroma collection.

---

## Generation

- **Primary:** Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL`).
- **Failsafe:** OpenAI-compatible chat API.

Ranking and retrieval work without Gemini; **answer generation** needs a key (or failsafe).

---

## Evaluation

```bash
uv run python src/final_eval.py --retrieval-only
# uv run python src/final_eval.py --with-generation
```

Outputs under `results/` (e.g. `EVALUATION_REPORT.md`).

Representative protocol figures from curated full-text eval reports (not a claim that every paper in a multi-thousand library is labelled):

| Metric | Approx. | Notes |
|--------|---------|--------|
| Mean MRR | ~0.75 | 95% CI ~0.66–0.83 in reported runs |
| Recall@k | ~0.85 | Labelled gold items |
| Hit@1 | ~0.69 | |
| Citation validity | ~0.95 | When generation is scored |

### Safe Limits & System Boundaries

1. **Ingested Scope Bound:** The Ask RAG pipeline answers questions strictly from the **ingested full-text papers** in `data/chroma_rag`. If a paper is only in the library (`data/corpus.json`) but was never ingested, the model will not have its full text and should refuse.
2. **Refusal Gating is Heuristic:** Refusal scoring uses heuristic substring matching against standardized phrasing (`not covered`, `do not contain`, etc.). It is an engineering safety gate, not a calibrated Bayesian confidence probability.
3. **Dense-Only Retrieval on Identifiers:** Dense semantic embeddings (`embeddinggemma-2`, `all-MiniLM-L6-v2`) prioritize conceptual semantics over literal token matching. Queries containing explicit arXiv IDs (e.g., `1806.07572`) or exact equation symbols benefit from hybrid BM25 lexical retrieval.
4. **Model Vector Incompatibility:** Indexes built with EmbeddingGemma 2 (768-d) cannot be queried with MiniLM (384-d). Changing models or Matryoshka dimensions requires running `make rag-ingest` with the new embedding configuration.
5. **No Hallucinated Citations:** Citation validity enforces that every inline marker `[n]` points to an actual retrieved chunk provided in the prompt context.

---

## Health check

```bash
bash scripts/check_local.sh
bash scripts/check_local.sh --with-model
```

---

## Troubleshooting

| Symptom | Try |
|---------|-----|
| `Failed to spawn: streamlit` | `uv sync`; `uv run python -m streamlit run src/app.py` |
| Empty rank results | Build library + `rebuild-index` |
| Thin Ask answers | Ingest relevant IDs; `rag.cli status` |
| Chroma dim errors | Rebuild after model/dim change |
| Gemini model 404 | Set `GEMINI_MODEL` to a supported id |
| Cloud OOM | Expected with large models; Cloud uses MiniLM |
| Re-ingest confusion | Prefer `reingest_overnight.sh`; avoid unnecessary `FRESH=1` |

---

## ConfAI 2026

Poster project for **ConfAI 2026** (Plaksha University):

**Literature Discovery with Multi-Signal Ranking and Citation-Grounded RAG**

Conference-facing notes: **[docs/CONF_AI.md](docs/CONF_AI.md)** · materials under `docs/confai/` when present.

Scope: **recommend → selective full-text ingest → citation-grounded RAG**, evaluation, and engineering—not ancillary side experiments.

---

## Design principles

1. Closed loop from interest to cited answer  
2. Inspectable ranking weights  
3. Selective full text (cost and focus)  
4. Grounding over uncited fluency  
5. Honest evaluation and limits  
6. Laptop-real CLI + web + Cloud demo path  

---

## Acknowledgements

Academic project, Plaksha University · Supervisor **Dr. Subrat Panda**.  
Third-party APIs and models remain under their own terms.

---

## Further reading

- [WORKFLOW.md](WORKFLOW.md)  
- [DEPLOY.md](DEPLOY.md)  
- [docs/CONF_AI.md](docs/CONF_AI.md)  
- `results/EVALUATION_REPORT.md` (after running eval)  
