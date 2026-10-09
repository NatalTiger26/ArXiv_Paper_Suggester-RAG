# ArXiv Paper Suggester

Cold-start paper library → multi-signal **recommend** → full-text **RAG** with citations.

**Embeddings:** EmbeddingGemma 2 only (local).  
**Generation:** Gemini primary; OpenAI-compatible failsafe (e.g. NVIDIA Llama).

---

## Install

```bash
cd ArXiv_Paper_Suggester
uv sync
cp .env.example .env
# set GEMINI_API_KEY=...
# optional: NVIDIA_API_KEY=...   (or FALLBACK_LLM_API_KEY)
# optional: HF_TOKEN=...
```

`torchvision` is required for EmbeddingGemma 2.

---

## End-to-end from empty data

```bash
# 1) Build library
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 10 --build-index
# or: uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 --build-index

# 2) Recommend
uv run python src/recommend.py "neural tangent kernel" -k 5

# 3) RAG
uv run python -m src.rag.cli ingest 1806.07572 2501.16496
uv run python -m src.rag.cli ask "What is the neural tangent kernel and why does it matter?"

# 4) UI
uv run streamlit run src/app.py
```

---

## Project modules

| Module | Role |
|--------|------|
| `src/settings.py` | Shared paths, EmbeddingGemma, Gemini, fallback LLM |
| `src/corpus_builder.py` | arXiv search / IDs → `data/corpus.json` |
| `src/corpus_index.py` | Title/abstract embeddings + Chroma for recommend |
| `src/ranking.py` | Multi-signal scoring |
| `src/recommend.py` | Recommend CLI |
| `src/evaluate_recommender.py` | Metrics vs personal ranking |
| `src/app.py` | Unified Streamlit |
| `src/rag/` | PDF → MD → chunk → embed → answer + citations |

---

## Embedding dimension (Matryoshka)

Default is full model dim (`EMBEDDING_DIM=0`).

```bash
# in .env — cheaper / smaller vectors
EMBEDDING_DIM=256
```

Then **rebuild both indexes** (old vectors are incompatible):

```bash
uv run python src/corpus_index.py build
uv run python -m src.rag.cli ingest --force <ids…>
```

---

## Generation failsafe

If Gemini is missing, rate-limited, or errors:

1. Set `NVIDIA_API_KEY` (or `FALLBACK_LLM_API_KEY`) from [build.nvidia.com](https://build.nvidia.com)
2. Defaults use `meta/llama-3.1-8b-instruct` via `https://integrate.api.nvidia.com/v1`
3. Any OpenAI-compatible server works (`FALLBACK_LLM_BASE_URL` + `FALLBACK_LLM_MODEL`)

---

## Eval

```bash
uv run python -m src.rag.cli eval --retrieval-only
uv run python src/evaluate_recommender.py run
```
