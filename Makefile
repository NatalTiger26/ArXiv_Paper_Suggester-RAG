.PHONY: help setup sync lock lock-check health recommend eval build-index index-rebuild \
        corpus-status corpus-search rag-status rag-ingest ingest rag-ask ask rag-eval web clean \
        confai-presence confai-sweep check test poster-numbers failure-dashboard budget-planner

UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
  PY := python
  RUN := python
else
  PY := uv run python
  RUN := uv run
endif

help:
	@echo "ArXiv Paper Suggester (uv-native ML workflow)"
	@echo "  make setup | sync | lock | lock-check"
	@echo "  make health | check | test | web"
	@echo "  make corpus-search Q=\"…\" N=10"
	@echo "  make recommend Q=\"…\" K=8"
	@echo "  make ingest IDS=\"1806.07572 2501.16496\""
	@echo "  make ask Q=\"What is the NTK?\""
	@echo "  make failure-dashboard | poster-numbers | budget-planner"
	@echo "  make eval | final-eval | final-eval-full"
	@echo "  make confai-presence | confai-sweep"

setup:
	@if command -v uv >/dev/null 2>&1; then uv sync; \
	else python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt; fi
	@echo "Environment synced via uv. Copy .env.example → .env and configure keys."

sync:
	uv sync

lock:
	uv lock

lock-check:
	uv lock --check

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

index-rebuild:
	$(PY) src/corpus_builder.py rebuild-index

recommend:
	$(PY) src/recommend.py "$(Q)" -k $(or $(K),10)

eval:
	$(PY) src/evaluate_recommender.py run

rag-status:
	$(PY) -m src.rag.cli status

rag-ingest:
	$(PY) -m src.rag.cli ingest $(IDS)

ingest:
	$(PY) -m src.rag.cli ingest $(IDS)

rag-ask:
	$(PY) -m src.rag.cli ask "$(Q)"

ask:
	$(PY) -m src.rag.cli ask "$(Q)"

rag-eval:
	$(PY) -m src.rag.cli eval --retrieval-only

confai-presence:
	$(PY) results/confai_analysis/scripts/A_presence_check.py

confai-sweep:
	$(PY) results/confai_analysis/scripts/B_index_size_sweep.py --sizes $(or $(SIZES),10,100) --seeds $(or $(SEEDS),1)

check:
	bash scripts/check_repo.sh

test:
	$(PY) -m unittest discover -s tests

poster-numbers:
	$(PY) scripts/export_poster_numbers.py

failure-dashboard:
	$(PY) scripts/failure_dashboard.py

budget-planner:
	$(PY) scripts/ingest_budget_planner.py

web:
	$(PY) -m streamlit run src/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true


eval-suite:
	$(PY) src/eval_suite.py --all

eval-suite-full:
	$(PY) src/eval_suite.py --all --with-generation

dim-sweep:
	$(PY) src/eval_suite.py --dim-sweep

eval-advanced:
	$(PY) src/eval_advanced.py --write-report

final-eval:
	$(PY) src/final_eval.py --retrieval-only

final-eval-full:
	$(PY) src/final_eval.py --with-generation --no-cache
