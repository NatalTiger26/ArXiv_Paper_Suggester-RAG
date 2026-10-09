.PHONY: help setup health recommend eval build-index \
        corpus-status corpus-search rag-status rag-ingest rag-ask rag-eval web clean

UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
  PY := python
  RUN := python
else
  PY := uv run python
  RUN := uv run
endif

help:
	@echo "ArXiv Paper Suggester"
	@echo "  make setup | health | web"
	@echo "  make corpus-search Q=\"…\" N=10"
	@echo "  make recommend Q=\"…\" K=8"
	@echo "  make rag-ingest IDS=\"1806.07572 2501.16496\""
	@echo "  make rag-ask Q=\"What is the NTK?\""
	@echo "  make rag-eval | eval"

setup:
	@if command -v uv >/dev/null 2>&1; then uv sync || uv pip install -r requirements.txt; \
	else python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt; fi
	@echo "Copy .env.example → .env and set GEMINI_API_KEY (and optional FALLBACK_LLM_API_KEY)"

health:
	$(PY) -c "from pathlib import Path; import json; \
p=Path('data/corpus.json') if Path('data/corpus.json').exists() else Path('data/corpus.json'); \
print('corpus', p, 'n=', len(json.loads(p.read_text())) if p.exists() else 0); \
print('chroma_corpus', Path('data/chroma_corpus').exists(), 'chroma_rag', Path('data/chroma_rag').exists())"
	$(PY) -m src.rag.cli status || true

corpus-status:
	$(PY) src/corpus_builder.py status

corpus-search:
	$(PY) src/corpus_builder.py search "$(Q)" -n $(or $(N),15) --build-index

build-index:
	$(PY) src/corpus_index.py build

recommend:
	$(PY) src/recommend.py "$(Q)" -k $(or $(K),10)

eval:
	$(PY) src/evaluate_recommender.py run

rag-status:
	$(PY) -m src.rag.cli status

rag-ingest:
	$(PY) -m src.rag.cli ingest $(IDS)

rag-ask:
	$(PY) -m src.rag.cli ask "$(Q)"

rag-eval:
	$(PY) -m src.rag.cli eval --retrieval-only

web:
	$(RUN) streamlit run src/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
