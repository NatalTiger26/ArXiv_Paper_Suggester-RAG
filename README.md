# ArXiv Paper Suggester + RAG

Cold-start **paper library** → multi-signal **recommendation** → full-text **RAG** with citations, evaluation harness, and a unified Streamlit UI.

Built as an extended AI-in-Practice / lab project: retrieve and rank arXiv papers for a research interest, then ask grounded questions over ingested PDFs with explicit references.

| Layer | Default (local) | Notes |
|--------|------------------|--------|
| **Embeddings** | `google/embeddinggemma-2` | Local only by default; Matryoshka dim via `EMBEDDING_DIM` |
| **Vector store** | Chroma (`data/chroma_corpus`, `data/chroma_rag`) | Separate collections for rank vs full-text |
| **Generation** | Gemini (`GEMINI_API_KEY`) | OpenAI-compatible failsafe (e.g. NVIDIA) |
| **Package manager** | **[uv](https://docs.astral.sh/uv/)** | Prefer `uv sync` / `uv run` — not pip |

---

## Table of contents

1. [What this system does](#what-this-system-does)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [Install (uv)](#install-uv)
6. [Configuration](#configuration)
7. [End-to-end workflow](#end-to-end-workflow)
8. [CLI reference](#cli-reference)
9. [Web UI (Streamlit)](#web-ui-streamlit)
10. [Embeddings and dimension](#embeddings-and-dimension)
11. [Generation and failsafe](#generation-and-failsafe)
12. [Evaluation](#evaluation)
13. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
14. [Troubleshooting](#troubleshooting)
15. [Design choices](#design-choices)
16. [Further reading](#further-reading)

---

## What this system does

1. **Library (metadata)**  
   Search arXiv by keywords or paste IDs. Stores titles/abstracts in `data/corpus.json` and builds a **recommend** index (title/abstract embeddings).

2. **Recommend**  
   Multi-signal ranking for a free-text research interest:
   - semantic similarity (embeddings)
   - recency
   - citation signal
   - author frequency  
   Optional weight presets and post-filters (`--since`, `--prefer-recent`, etc.).

3. **Ingest (full text)**  
   For selected arXiv IDs: download PDF → Markdown (pymupdf4llm, OCR fallback where needed) → section-aware chunks → embed → Chroma RAG collection.

4. **Ask (RAG)**  
   Retrieve top passages, call Gemini (or failsafe LLM), return an answer with **coverage** signal and **citations** (arXiv id, section, score, snippet).

5. **Evaluate**  
   Golden-set style harness: MRR, Recall@k, hit@1, bootstrap CIs, dim sweep, refusal metrics (when generation is enabled), report under `results/`.

---

## Architecture

```text
                    ┌─────────────────────┐
   arXiv API ──────►│ corpus_builder      │──► data/corpus.json
                    │ (+ optional index)  │──► data/chroma_corpus
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
   interest text ──►│ ranking / recommend │──► ranked paper list
                    └──────────┬──────────┘
                               │ selected IDs
                    ┌──────────▼──────────┐
                    │ rag: download       │
                    │   → pdf_to_md       │
                    │   → chunk           │
                    │   → embed_store     │──► data/chroma_rag
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
   question ───────►│ rag_pipeline.ask    │──► answer + references
                    │ (Gemini / failsafe) │
                    └─────────────────────┘
```

**Two indexes, two jobs**

| Index | Path | Used by | Content |
|-------|------|---------|---------|
| Recommend | `data/chroma_corpus` | `recommend.py`, ranking | Title / abstract |
| RAG | `data/chroma_rag` | `rag.cli ask`, Streamlit Ask | Full-text chunks |

Ask **only** sees papers that were **ingested**, not merely listed in the library.

---

## Repository layout

```text
ArXiv_Paper_Suggester/
├── pyproject.toml          # uv project + dependencies
├── requirements.txt        # mirror for Streamlit Cloud
├── packages.txt            # apt packages (Cloud: tesseract)
├── .env.example            # copy to .env (never commit secrets)
├── .streamlit/             # Cloud/UI config + secrets example
├── WORKFLOW.md             # short copy-paste CLI path
├── DEPLOY.md               # Streamlit Cloud checklist
├── README.md               # this file
├── Makefile                # optional helpers
├── data/                   # runtime data (may be empty on clone)
│   ├── corpus.json
│   ├── chroma_corpus/
│   ├── chroma_rag/
│   ├── papers/             # PDFs
│   ├── markdown/           # converted text
│   └── rag_cache/
├── rankings/               # personal importance CSVs (optional)
├── results/                # eval reports and metrics
│   ├── EVALUATION_REPORT.md
│   └── rag_eval/
├── docs/, notebooks/, legacy/, scripts/
└── src/
    ├── settings.py         # paths, models, env
    ├── corpus_builder.py   # search / from-ids / status / rebuild-index
    ├── corpus_index.py     # embed + Chroma for recommend
    ├── ranking.py          # multi-signal score_and_rank
    ├── recommend.py        # recommend CLI
    ├── evaluate_recommender.py
    ├── final_eval.py       # main RAG evaluation entry
    ├── app.py              # unified Streamlit UI
    └── rag/
        ├── cli.py          # ingest, ask, status, reset-index, eval
        ├── download.py
        ├── pdf_to_md.py
        ├── chunk.py
        ├── embed_store.py
        ├── cache.py
        ├── rag_pipeline.py
        └── evaluate.py
```

---

## Prerequisites

- **Python 3.10–3.12** (3.11 or 3.12 recommended; avoid 3.13+ for torch wheels)
- **[uv](https://docs.astral.sh/uv/)** installed
- Network access (arXiv API, Hugging Face for models, Gemini API)
- Optional: Gemini API key, Hugging Face token for higher rate limits
- Optional failsafe: NVIDIA / OpenAI-compatible API key

Install uv if needed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Install (uv)

All dependency and run commands use **uv**. Do not use `pip install` for this project.

```bash
cd ArXiv_Paper_Suggester

# Create .venv and install dependencies from pyproject.toml / lock
uv sync

# Environment file
cp .env.example .env
# Edit .env: set GEMINI_API_KEY=... and GEMINI_MODEL=gemini-2.0-flash
# Optional: HF_TOKEN, FALLBACK_LLM_API_KEY / NVIDIA_API_KEY

# Sanity checks
uv run python -c "import streamlit, chromadb, sentence_transformers; print('ok')"
uv run python -c "import streamlit; print('streamlit', streamlit.__version__)"
```

Add a dependency later:

```bash
uv add some-package
```

Run any script without activating the venv manually:

```bash
uv run python src/recommend.py "..."
uv run python -m streamlit run src/app.py
```

`torch` and `torchvision` are declared explicitly (needed for EmbeddingGemma 2 and Streamlit Cloud installs).

---

## Configuration

### `.env` (local)

| Variable | Purpose | Typical value |
|----------|---------|----------------|
| `GEMINI_API_KEY` | Primary LLM | required for generation |
| `GEMINI_MODEL` | Model id | `gemini-2.0-flash` (use a model your key supports) |
| `FALLBACK_LLM_API_KEY` / `NVIDIA_API_KEY` | Failsafe LLM | optional |
| `FALLBACK_LLM_BASE_URL` | OpenAI-compatible base | NVIDIA default in `.env.example` |
| `FALLBACK_LLM_MODEL` | Failsafe model | e.g. `meta/llama-3.1-8b-instruct` |
| `EMBEDDING_MODEL` | Sentence-transformers id | `google/embeddinggemma-2` (local default) |
| `EMBEDDING_DIM` | Matryoshka truncate | `0` = full; or `512` / `256` / `128` |
| `HF_TOKEN` | Hugging Face | optional rate limits / gated models |

After changing **`EMBEDDING_DIM` or `EMBEDDING_MODEL`**, rebuild indexes (old vectors are incompatible):

```bash
uv run python src/corpus_builder.py rebuild-index
uv run python -m src.rag.cli reset-index
uv run python -m src.rag.cli ingest --force <arxiv_ids…>
```

### Streamlit Cloud secrets

Do **not** rely on `.env` on Cloud. Use **App settings → Secrets** (see [DEPLOY.md](DEPLOY.md)).

---

## End-to-end workflow

From an empty `data/` folder:

```bash
uv sync

# 1) Library + recommend index
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index
# or fixed IDs:
# uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 2304.14997 --build-index

uv run python src/corpus_builder.py status

# 2) Rank by interest
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8

# 3) Full-text ingest (slow first time: download + embed)
uv run python -m src.rag.cli ingest 2501.16496 2304.14997 1806.07572
uv run python -m src.rag.cli status

# 4) Ask
uv run python -m src.rag.cli ask "What are open problems in mechanistic interpretability?"

# 5) Web UI
uv run python -m streamlit run src/app.py
```

More detail: [WORKFLOW.md](WORKFLOW.md).

---

## CLI reference

### Corpus / library

```bash
uv run python src/corpus_builder.py search "keywords" -n 20 --build-index
uv run python src/corpus_builder.py search "..." --category cs.LG --sort relevance
uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 --build-index
uv run python src/corpus_builder.py status
uv run python src/corpus_builder.py rebuild-index
```

### Recommend

```bash
uv run python src/recommend.py "neural tangent kernel" -k 10
uv run python src/recommend.py "circuits" -k 8 --prefer-recent
uv run python src/recommend.py "NTK" -k 8 --weights 0.65,0.10,0.20,0.05
uv run python src/recommend.py "..." -k 8 --since 2020 --json
uv run python src/recommend.py --demo
```

### RAG

```bash
uv run python -m src.rag.cli ingest 1806.07572 2501.16496
uv run python -m src.rag.cli ingest --force 1806.07572          # re-download / re-embed
uv run python -m src.rag.cli status
uv run python -m src.rag.cli papers
uv run python -m src.rag.cli ask "What is the neural tangent kernel and why does it matter?"
uv run python -m src.rag.cli reset-index                       # after EMBEDDING_DIM change
```

### Evaluation

```bash
# Retrieval-focused (no LLM cost)
uv run python src/final_eval.py --retrieval-only

# With generation (cold cache for honest refusal metrics)
uv run python src/final_eval.py --with-generation --no-cache
```

Outputs typically include:

- `results/EVALUATION_REPORT.md`
- `results/rag_eval/final_eval.json`
- `results/rag_eval/dim_sweep.json`

Recommender-side evaluation (vs personal rankings):

```bash
uv run python src/evaluate_recommender.py run
```

---

## Web UI (Streamlit)

```bash
uv run python -m streamlit run src/app.py
```

Prefer **`python -m streamlit`** over bare `uv run streamlit …` so the package entrypoint resolves reliably after `uv sync`.

**Tabs**

| Tab | Purpose |
|-----|---------|
| **Home** | Orientation + optional one-click demo path |
| **Library** | Search arXiv / paste IDs / rebuild recommend index / ranking-based ingest |
| **Recommend** | Rank library → **Ingest top-N for Q&A** |
| **Ask** | Question answering with coverage, sources, Markdown download |
| **Help** | Concepts, CLI cheatsheet, eval report preview |

**UX notes**

- Sidebar: Library count vs Full-text count, system health (key present / missing), CLI hints.
- **Activity log** (when present): timestamped steps if something feels stuck (first embedder load can take 1–2 minutes).
- Ask latency is often **10–20 seconds** when calling the LLM.

---

## Embeddings and dimension

- **Local default:** `google/embeddinggemma-2` (requires `torch` / `torchvision`).
- **Matryoshka:** set `EMBEDDING_DIM=512` (or 256 / 128) to truncate and re-normalize. Quality vs cost trade-off is measured in the dim sweep inside `final_eval`.
- **Streamlit Cloud:** free tier RAM is tight; the code can prefer a **small** embedder (e.g. MiniLM) on Cloud hosts. Indexes built with Gemma **must not** be queried with MiniLM (dimension mismatch). Rebuild or ingest fresh on Cloud. See [DEPLOY.md](DEPLOY.md).

---

## Generation and failsafe

1. Primary: Gemini via `google-genai` and `GEMINI_API_KEY`.
2. If Gemini is missing, rate-limited, or errors: OpenAI-compatible client using `FALLBACK_LLM_*` or `NVIDIA_API_KEY` (see `.env.example`).

Coverage gating reduces confident answers when retrieval is weak; still treat refusal metrics as imperfect until eval thresholds are met (see evaluation report).

---

## Evaluation

The project includes a rigorous RAG eval path aligned with lab-style metrics:

- **Retrieval:** MRR, Recall@k, hit@1, stratified by question kind where labeled
- **Uncertainty:** bootstrap 95% CIs
- **Generation (optional):** refusal precision/recall, citation validity, latency p50/p95
- **Dim sweep:** 128 / 256 / 512 / 768
- **Report:** narrative + recommendation in `results/EVALUATION_REPORT.md`

Re-run after corpus or model changes:

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
```

---

## Deploying to Streamlit Community Cloud

Community Cloud is **not** a full substitute for a local machine with EmbeddingGemma:

- ~1 GB RAM often cannot load large local embedders
- Filesystem is ephemeral; treat Cloud as demo / read-mostly
- Secrets must be configured in the Streamlit UI

**Minimum Cloud checklist**

1. Main file: `src/app.py`
2. Python **3.11 or 3.12**
3. Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL`
4. `requirements.txt` includes **torch**
5. Rebuild RAG index with the **same** embedder Cloud uses

Full steps: **[DEPLOY.md](DEPLOY.md)**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Failed to spawn: streamlit` | Deps not synced / wrong entry | `uv sync` then `uv run python -m streamlit run src/app.py` |
| Rank fails / empty | No library or no recommend index | `corpus_builder.py search … --build-index` or `rebuild-index` |
| Ask “not covered” | No full-text ingest | `rag.cli ingest <ids>` |
| Chroma dimension error | `EMBEDDING_DIM` / model changed | `reset-index` + `ingest --force` |
| Gemini 404 / model not found | Wrong `GEMINI_MODEL` | Set a model your key supports (e.g. `gemini-2.0-flash`) |
| Cloud “Oh no. Error running app.” | Install / Python / OOM | Logs in Manage app; Python 3.11–3.12; torch in requirements; small embedder |
| First command “hangs” 1–2 min | Model download / load | Normal; watch terminal / Activity log |

---

## Design choices

- **uv-only workflow** for reproducible envs (`uv sync`, `uv run`, `uv add`).
- **Local embeddings** to control cost and latency; API only for generation.
- **Hybrid corpus size:** on-demand ingest of recommended or explicit IDs (not bulk-OCR of all of arXiv).
- **Citations required** in the RAG path for lab-style trust.
- **CLI + UI:** CLI for diagnosis and batch work; Streamlit for demos and exploration.
- **Evaluation as a first-class artifact** (`final_eval.py` + `EVALUATION_REPORT.md`), not an afterthought.

---

## Further reading

| Doc | Content |
|-----|---------|
| [WORKFLOW.md](WORKFLOW.md) | Exact CLI path and stuck-state table |
| [DEPLOY.md](DEPLOY.md) | Streamlit Cloud deployment |
| `results/EVALUATION_REPORT.md` | Metrics narrative after you run eval |
| `.env.example` | All supported environment variables |
| `.streamlit/secrets.toml.example` | Cloud secrets template |

---

## License / academic use

Intended for coursework, portfolio, and research prototyping. Respect arXiv’s terms of use and the licenses of models (EmbeddingGemma, Gemini, failsafe providers) and third-party packages.

---

**Quick start (copy-paste)**

```bash
cd ArXiv_Paper_Suggester
uv sync
cp .env.example .env   # set GEMINI_API_KEY and GEMINI_MODEL=gemini-2.0-flash

uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python -m src.rag.cli ingest 2501.16496 2304.14997 1806.07572
uv run python -m src.rag.cli ask "What are open problems in mechanistic interpretability?"
uv run python -m streamlit run src/app.py
```
