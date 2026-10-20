# Workflow (CLI)

All commands from the project root. Use **`uv run`**.

## Setup

```bash
uv sync
cp .env.example .env   # add GEMINI_API_KEY for generation
```

## 1. Library

```bash
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 15 --build-index
uv run python src/corpus_builder.py from-ids 1806.07572 2501.16496 2304.14997 --build-index
uv run python src/corpus_builder.py status
uv run python src/corpus_builder.py rebuild-index
```

## 2. Rank

```bash
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python src/recommend.py "Grad-CAM" -k 10 --prefer-recent
```

## 3. Full-text ingest

```bash
uv run python -m src.rag.cli ingest 2304.14997 2501.16496
uv run python -m src.rag.cli status
```

Overnight / large (resumable):

```bash
bash scripts/reingest_overnight.sh
# Ctrl+C stops after current batch; resume with the same command.
# FRESH=1 only when intentionally resetting the RAG index.
```

## 4. Ask

```bash
uv run python -m src.rag.cli ask "What is circuit discovery?"
```

## 5. UI

```bash
uv run python -m streamlit run src/app.py
```

## 6. Evaluate

```bash
uv run python src/final_eval.py --retrieval-only
```

## Reminders

- Library ≠ RAG — ingest before full-text answers.
- Rebuild indexes after embedding model or `EMBEDDING_DIM` changes.
- Cloud demo may use MiniLM and a smaller library than local.
