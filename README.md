# ArXiv Paper Suggester + RAG

**Find papers. Rank them for your research interest. Read the full text. Ask questions with citations.**

A cold-start literature workspace: multi-signal arXiv recommendation, full-text PDF → Markdown → chunk → embed pipeline, and a grounded RAG assistant (Gemini primary, OpenAI-compatible failsafe). Ships with a **CLI** for diagnosis and batch work and a **Streamlit** UI for demos and exploration.

| Layer | Local (laptop) | Streamlit Community Cloud |
|--------|----------------|---------------------------|
| **Embeddings** | `google/embeddinggemma-2` (Matryoshka dim optional) | Auto **MiniLM** (`all-MiniLM-L6-v2`) — free tier RAM |
| **Vector store** | Chroma (`data/chroma_corpus`, `data/chroma_rag`) | Same layout; ephemeral filesystem |
| **Generation** | Gemini (`GEMINI_MODEL`, default `gemini-3.8-flash`) | Same; Secrets required |
| **Failsafe LLM** | NVIDIA / any OpenAI-compatible endpoint | Optional |
| **Package manager** | **[uv](https://docs.astral.sh/uv/)** | Cloud uses `uv.lock` / `requirements.txt` |

> **Package manager:** this project is built around **`uv sync` / `uv run`**. Prefer those over `pip install`.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [What you can do](#what-you-can-do)
3. [Architecture](#architecture)
4. [Repository layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Install (uv)](#install-uv)
7. [Configuration](#configuration)
8. [Quick start](#quick-start)
9. [End-to-end workflow](#end-to-end-workflow)
10. [CLI reference](#cli-reference)
11. [Web UI (Streamlit)](#web-ui-streamlit)
12. [Embeddings & dimensions](#embeddings--dimensions)
13. [Generation (Gemini + failsafe)](#generation-gemini--failsafe)
14. [Evaluation](#evaluation)
15. [Local health check](#local-health-check)
16. [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
17. [Troubleshooting](#troubleshooting)
18. [Design choices](#design-choices)
19. [Further reading](#further-reading)

---

## Why this exists

Literature review for a new research direction is slow: search, skim abstracts, open PDFs, lose track of what you already read, and struggle to ask *precise* questions across papers.

This project turns that into a single loop:

1. **Discover** papers from arXiv (keywords or explicit IDs).
2. **Rank** them for *your* interest (semantic + recency + citation + author signals).
3. **Ingest** only the papers you care about (PDF → text → chunks → vectors).
4. **Ask** grounded questions and get answers with **references** (arXiv id, section, score, snippet).

It was extended as an **AI in Practice** style lab track: retrieval quality, caching, coverage gating, evaluation metrics (MRR, Recall@k, CIs, refusal), CLI + UI, and deployment constraints.

---

## What you can do

| Capability | How |
|------------|-----|
| Cold-start library | `corpus_builder search` / `from-ids` |
| Multi-signal recommend | `recommend.py` or **Recommend** tab |
| Full-text ingest | `rag.cli ingest` or UI ingest / Quick demo |
| Q&A with citations | `rag.cli ask` or **Ask** tab |
| Unified UI | `streamlit run src/app.py` |
| Health check | `bash scripts/check_local.sh` |
| Rigorous eval | `final_eval.py` → `results/EVALUATION_REPORT.md` |
| Cloud demo | Streamlit Community Cloud (MiniLM + Gemini) |

**Important:** **Library ≠ full-text.** Ranking uses titles/abstracts. **Ask** only sees papers you **ingested**.

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
                             │
              interest text  │
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
                             │
                    question │
                             ▼
                    ┌─────────────────┐
                    │ retrieve chunks │
                    │ Gemini / failsafe│──► answer + coverage + citations
                    └─────────────────┘
```

### Two indexes, two jobs

| Index | Path | Content | Used by |
|-------|------|---------|---------|
| **Recommend** | `data/chroma_corpus` | Title + abstract embeddings | Rank / recommend |
| **RAG** | `data/chroma_rag` | Full-text chunks | Ask / eval |

Changing **embedding model or dimension** requires rebuilding **both** indexes (vectors are not interchangeable).

---

## Repository layout

```text
ArXiv_Paper_Suggester/
├── pyproject.toml           # uv project + dependencies
├── requirements.txt         # mirror for Streamlit Cloud
├── packages.txt             # apt: tesseract-ocr (Cloud OCR)
├── uv.lock                  # locked deps (Cloud may uv-sync this)
├── .env.example             # copy → .env (never commit secrets)
├── .streamlit/              # config + secrets example
├── README.md                # this file
├── WORKFLOW.md              # short CLI path
├── DEPLOY.md                # Cloud checklist
├── Makefile                 # optional helpers
├── scripts/
│   └── check_local.sh       # local verification
├── data/                    # runtime (often empty on clone)
│   ├── corpus.json
│   ├── chroma_corpus/
│   ├── chroma_rag/
│   ├── papers/              # PDFs
│   ├── markdown/
│   ├── embeddings/
│   └── rag_cache/
├── rankings/                # optional personal importance CSVs
├── results/                 # EVALUATION_REPORT.md, rag_eval/
├── docs/ notebooks/ legacy/
└── src/
    ├── settings.py          # paths, models, Cloud detection, Secrets
    ├── corpus_builder.py    # search / from-ids / status / rebuild-index
    ├── corpus_index.py      # recommend embeddings + Chroma
    ├── ranking.py           # multi-signal score_and_rank
    ├── recommend.py         # recommend CLI
    ├── evaluate_recommender.py
    ├── final_eval.py        # main RAG evaluation
    ├── app.py               # Streamlit UI
    └── rag/
        ├── cli.py           # ingest | ask | status | papers | reset-index | eval
        ├── download.py
        ├── pdf_to_md.py
        ├── chunk.py
        ├── embed_store.py
        ├── cache.py
        ├── rag_pipeline.py  # retrieve → generate → cite
        └── evaluate.py
```

---

## Prerequisites

- **Python 3.10–3.12** (3.11 or 3.12 recommended; avoid 3.13+ for torch wheels)
- **[uv](https://docs.astral.sh/uv/)** installed
- Network access: arXiv, Hugging Face (models), Gemini API
- Optional: Hugging Face token (rate limits), NVIDIA / OpenAI-compatible failsafe key

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Install (uv)

```bash
cd ArXiv_Paper_Suggester

# Create .venv and install from pyproject / lock
uv sync

# Secrets (local)
cp .env.example .env
# Edit .env:
#   GEMINI_API_KEY=...
#   GEMINI_MODEL=gemini-3.8-flash
# Optional: HF_TOKEN, NVIDIA_API_KEY / FALLBACK_LLM_*

# Sanity
uv run python -c "import streamlit, chromadb, sentence_transformers, torch; print('ok')"
```

Add a dependency later:

```bash
uv add some-package
```

Always run tools with **`uv run`** (no need to manually activate the venv):

```bash
uv run python src/recommend.py "..."
uv run python -m streamlit run src/app.py
```

---

## Configuration

### Local: `.env`

| Variable | Role | Typical value |
|----------|------|----------------|
| `GEMINI_API_KEY` | Primary LLM | required for generation |
| `GEMINI_MODEL` | Model id | `gemini-3.8-flash` (use what **your** API accepts) |
| `FALLBACK_LLM_API_KEY` / `NVIDIA_API_KEY` | Failsafe LLM | optional |
| `FALLBACK_LLM_BASE_URL` | OpenAI-compatible base | see `.env.example` |
| `FALLBACK_LLM_MODEL` | Failsafe model | e.g. `meta/llama-3.1-8b-instruct` |
| `EMBEDDING_MODEL` | Local embedder | `google/embeddinggemma-2` |
| `EMBEDDING_DIM` | Matryoshka truncate | `0` = full; or `512` / `256` / `128` |
| `HF_TOKEN` | Hugging Face | optional |

After changing **`EMBEDDING_MODEL` or `EMBEDDING_DIM`**:

```bash
uv run python src/corpus_builder.py rebuild-index
uv run python -m src.rag.cli reset-index
uv run python -m src.rag.cli ingest --force <arxiv_ids…>
```

### Streamlit Cloud: Secrets (not `.env`)

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-3.8-flash"
```

Do **not** set `EMBEDDING_MODEL` to EmbeddingGemma on free Cloud (OOM). Leave unset so the app forces MiniLM.

**Note:** Streamlit Secrets **override** `.env` values so Cloud does not keep a stale model name.

---

## Quick start

```bash
cd ArXiv_Paper_Suggester
uv sync
cp .env.example .env   # set GEMINI_API_KEY and GEMINI_MODEL=gemini-3.8-flash

uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python -m src.rag.cli ingest 2501.16496 2304.14997 1806.07572
uv run python -m src.rag.cli ask "What are open problems in mechanistic interpretability?"
uv run python -m streamlit run src/app.py
```

---

## End-to-end workflow

### 1. Build a library (metadata)

```bash
# Keywords
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index

# Or explicit IDs
uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 2304.14997 --build-index

uv run python src/corpus_builder.py status
```

### 2. Rank for an interest

```bash
uv run python src/recommend.py "neural tangent kernel" -k 10
uv run python src/recommend.py "circuits" -k 8 --prefer-recent
uv run python src/recommend.py "NTK" -k 8 --weights 0.65,0.10,0.20,0.05 --json
```

### 3. Ingest full text (slow first time)

```bash
uv run python -m src.rag.cli ingest 1806.07572 2501.16496
uv run python -m src.rag.cli status
uv run python -m src.rag.cli papers
```

### 4. Ask

```bash
uv run python -m src.rag.cli ask "What is the neural tangent kernel and why does it matter?"
```

### 5. UI

```bash
uv run python -m streamlit run src/app.py
```

Prefer **`python -m streamlit`** over bare `uv run streamlit …` so the entrypoint always resolves after `uv sync`.

More detail: **[WORKFLOW.md](WORKFLOW.md)**.

---

## CLI reference

### Corpus

```bash
uv run python src/corpus_builder.py search "keywords" -n 20 --build-index
uv run python src/corpus_builder.py search "..." --category cs.LG --sort relevance
uv run python src/corpus_builder.py from-ids ID1 ID2 --build-index
uv run python src/corpus_builder.py status
uv run python src/corpus_builder.py rebuild-index
```

### Recommend

```bash
uv run python src/recommend.py "query" -k 10
uv run python src/recommend.py "query" -k 8 --prefer-recent --since 2020
uv run python src/recommend.py "query" -k 8 --weights 0.65,0.10,0.20,0.05
uv run python src/recommend.py --demo
```

### RAG

```bash
uv run python -m src.rag.cli ingest ID1 ID2
uv run python -m src.rag.cli ingest --force ID1
uv run python -m src.rag.cli ingest --from-recommend "circuits" -k 4
uv run python -m src.rag.cli status
uv run python -m src.rag.cli papers
uv run python -m src.rag.cli ask "Your question"
uv run python -m src.rag.cli reset-index
uv run python -m src.rag.cli eval --retrieval-only
```

### Evaluation

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
uv run python src/evaluate_recommender.py run
```

Outputs typically land under `results/EVALUATION_REPORT.md` and `results/rag_eval/`.

---

## Web UI (Streamlit)

```bash
uv run python -m streamlit run src/app.py
```

| Tab | Purpose |
|-----|---------|
| **Home** | Orientation + **Quick demo** (seed → rank → ingest → ask) |
| **Library** | Search arXiv / paste IDs / rebuild recommend index |
| **Recommend** | Rank library → ingest top-N for Q&A |
| **Ask** | Question answering with coverage, sources, Markdown download |
| **Help** | Concepts, CLI cheatsheet, eval notes |

**Sidebar:** library count vs full-text count, embedder name/dim, system health (API keys), CLI hints, activity log for long steps.

First embedder load can take **1–2 minutes**; Ask with Gemini often **8–20 seconds**.

---

## Embeddings & dimensions

- **Local default:** EmbeddingGemma 2 (`google/embeddinggemma-2`), needs `torch` + `torchvision`.
- **Matryoshka:** set `EMBEDDING_DIM=512` (or 256 / 128), then rebuild indexes. Dim sweeps live in evaluation.
- **Cloud:** free tier forces **MiniLM (384-d)** unless `ALLOW_LARGE_EMBED=1` (not recommended). Gemma indexes are **not** queryable with MiniLM — rebuild / re-ingest on Cloud.

---

## Generation (Gemini + failsafe)

1. **Primary:** Gemini via `google-genai` and `GEMINI_API_KEY`.
2. Model id is resolved **at call time** from Secrets / env (`get_gemini_model()`), with retries if Google returns a “use models/…” suggestion.
3. **Failsafe:** OpenAI-compatible client (`FALLBACK_LLM_*` or `NVIDIA_API_KEY`).

Coverage labels (`high` / `partial` / `low` / `none`) come from **retrieval scores**, not calibrated probabilities. Prefer more ingested papers for broader questions.

Use the model id **your key supports**. Google retires names over time; if you see a 404, set `GEMINI_MODEL` to the name in the error message (e.g. `gemini-3.8-flash`).

---

## Evaluation

Aligned with lab-style metrics:

- Retrieval: **MRR**, **Recall@k**, **hit@1**, stratified by question kind
- Uncertainty: **bootstrap 95% CIs**
- Generation (optional): refusal precision/recall, citation validity, latency p50/p95
- Dim sweep: 128 / 256 / 512 / 768
- Narrative report: `results/EVALUATION_REPORT.md`

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
```

Re-run after corpus, embedder, or model changes.

---

## Local health check

```bash
bash scripts/check_local.sh            # imports, modules, CLI, Streamlit, arXiv
bash scripts/check_local.sh --quick    # skip network / model load
bash scripts/check_local.sh --with-model  # load embedder + encode test string
```

Or `make check` / `make check-quick` / `make check-model` if those targets are present.

---

## Deploy to Streamlit Cloud

Community Cloud is a **demo** host, not a full substitute for a laptop with EmbeddingGemma.

| Do | Don’t |
|----|--------|
| Python **3.11 or 3.12** | Python 3.13+ |
| Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.8-flash` | Put Gemma in Cloud Secrets |
| Main file: `src/app.py` | Expect persistent `data/` forever |
| Ingest **1–3** papers for demos | Bulk OCR + huge demos on free tier |

Full checklist: **[DEPLOY.md](DEPLOY.md)**.

Live pattern that works: MiniLM → rank → ingest 3 papers → Ask with Gemini 3.8 → answer + citations.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Failed to spawn: streamlit` | Deps / entrypoint | `uv sync` then `uv run python -m streamlit run src/app.py` |
| Empty rank | No library / no recommend index | `corpus_builder … --build-index` or `rebuild-index` |
| Ask “not covered” | No full-text ingest | `rag.cli ingest <ids>` |
| Chroma dimension error | Model/dim changed | `reset-index` + `ingest --force` |
| `Collection already exists` | Rebuild race | Update `corpus_index` (retry/get_or_create); redeploy |
| Gemini 404 | Wrong / retired model id | Set `GEMINI_MODEL` to the id Google names; reboot Cloud |
| Cloud still uses old model | Stale env vs Secrets | Secrets must list correct `GEMINI_MODEL`; reboot app |
| Cloud OOM / connection reset | Heavy embedder | Ensure MiniLM on Cloud; do not force Gemma |
| First command “hangs” 1–2 min | Model download | Normal; watch terminal / Activity log |

---

## Design choices

- **uv-only** workflow for reproducible environments.
- **Local embeddings** for cost control; API only for generation.
- **Hybrid corpus size:** on-demand ingest of recommended or explicit IDs (not bulk-OCR of all arXiv).
- **Citations required** on the RAG path for trust and lab-style deliverables.
- **CLI + UI:** CLI for debugging and batch jobs; Streamlit for demos.
- **Cloud guards:** refuse EmbeddingGemma on free tier; Secrets override stale env; auto-reset incompatible Chroma collections when possible.
- **Evaluation as a first-class artifact**, not an afterthought.

---

## Further reading

| Document | Content |
|----------|---------|
| [WORKFLOW.md](WORKFLOW.md) | Exact CLI path and stuck-state table |
| [DEPLOY.md](DEPLOY.md) | Streamlit Cloud deployment |
| `results/EVALUATION_REPORT.md` | Metrics narrative after you run eval |
| `.env.example` | All supported environment variables |
| `.streamlit/secrets.toml.example` | Cloud secrets template |
| `scripts/check_local.sh` | Local verification |

---

## License / academic use

Intended for coursework, portfolio, and research prototyping. Respect [arXiv terms](https://arxiv.org/help/api), model licenses (EmbeddingGemma, Gemini, failsafe providers), and third-party package licenses.

---

### One-liner recap

```bash
uv sync && cp .env.example .env   # set GEMINI_API_KEY + GEMINI_MODEL=gemini-3.8-flash
uv run python src/corpus_builder.py search "your topic" -n 12 --build-index
uv run python src/recommend.py "your topic" -k 8
uv run python -m src.rag.cli ingest <top_ids…>
uv run python -m src.rag.cli ask "Your question"
uv run python -m streamlit run src/app.py
```
