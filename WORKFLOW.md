# Exact workflow — CLI and web

All commands assume you are in the project root (the folder that contains `pyproject.toml` and `src/`).

---

## 0. One-time setup

```bash
cd "/path/to/ArXiv_Paper_Suggester"

uv sync

# Create .env if needed (do not commit secrets)
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-3.8-flash   # or whatever your key supports
# EMBEDDING_DIM=0                # 0 = full 768; use 512 only after reset-index

# Confirm Streamlit is installed
uv run python -c "import streamlit; print('streamlit', streamlit.__version__)"
```

Start the web UI (prefer the module form):

```bash
uv run python -m streamlit run src/app.py
```

If you see `Failed to spawn: streamlit`, run `uv sync` (or `uv add streamlit`) and use the command above.

---

## 1. Build a paper library (metadata)

### Option A — search by keywords

```bash
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index
```

- Saves papers into `data/corpus.json`
- `--build-index` embeds titles/abstracts into `data/chroma_corpus` for ranking

### Option B — explicit arXiv IDs

```bash
uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 2304.14997 --build-index
```

### Status

```bash
uv run python src/corpus_builder.py status
```

### Rebuild recommend index only (after editing corpus)

```bash
uv run python src/corpus_builder.py rebuild-index
```

---

## 2. Rank papers for an interest

```bash
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
```

Useful flags:

```bash
uv run python src/recommend.py "neural tangent kernel" -k 10 --prefer-recent
uv run python src/recommend.py "circuits" -k 8 --weights 0.65,0.10,0.20,0.05
uv run python src/recommend.py "NTK" -k 8 --json
```

Requires a non-empty library **and** a built recommend index (`--build-index` or `rebuild-index`).

---

## 3. Ingest full text for Q&A (RAG)

Downloads PDF → Markdown → chunks → EmbeddingGemma 2 → `data/chroma_rag`.

```bash
uv run python -m src.rag.cli ingest 1806.07572 2501.16496 2304.14997
```

Force re-download / re-embed:

```bash
uv run python -m src.rag.cli ingest --force 1806.07572 2501.16496
```

From personal ranking CSV (if you have rankings):

```bash
# uses rankings/personal_importance.csv via the Python API / Library tab
```

### RAG status

```bash
uv run python -m src.rag.cli status
uv run python -m src.rag.cli papers
```

### If you changed `EMBEDDING_DIM`

```bash
uv run python -m src.rag.cli reset-index
uv run python -m src.rag.cli ingest --force <ids…>
```

---

## 4. Ask a question

```bash
uv run python -m src.rag.cli ask "What is the neural tangent kernel and why does it matter?"
```

Typical wall time: **10–20 seconds** (LLM). Retrieval alone is tens of milliseconds.

---

## 5. Evaluation

```bash
# Retrieval + dim sweep + report (no LLM cost)
uv run python src/final_eval.py --retrieval-only

# Cold generation + refusal + citation validity
uv run python src/final_eval.py --with-generation --no-cache
```

Outputs:

- `results/EVALUATION_REPORT.md`
- `results/rag_eval/final_eval.json`
- `results/rag_eval/dim_sweep.json`

---

## End-to-end minimal path (copy-paste)

```bash
uv sync

uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index

uv run python src/recommend.py "mechanistic interpretability circuits" -k 8

uv run python -m src.rag.cli ingest 2501.16496 2304.14997 1806.07572

uv run python -m src.rag.cli status

uv run python -m src.rag.cli ask "What are open problems in mechanistic interpretability?"

uv run python -m streamlit run src/app.py
```

---

## When something feels stuck

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| No output for 1–2 min | First EmbeddingGemma load | Wait; check terminal |
| Rank fails | Empty library or no index | `corpus_builder.py status` then `rebuild-index` |
| Ask says not covered | No full-text ingest | `rag.cli ingest <ids>` |
| Dimension error | `EMBEDDING_DIM` changed | `rag.cli reset-index` then re-ingest |
| Streamlit spawn error | Deps not synced | `uv sync` then `python -m streamlit run …` |
| Gemini errors | Key / model name | Check `.env` `GEMINI_API_KEY`, `GEMINI_MODEL` |

In the **web UI**, open **Activity log** at the top of the page — every major step is timestamped there and mirrored to the terminal.
