#!/usr/bin/env bash
# Local health check for ArXiv Paper Suggester (CLI + Streamlit readiness).
# Usage (from project root):
#   bash scripts/check_local.sh
#   bash scripts/check_local.sh --quick      # skip model load / network
#   bash scripts/check_local.sh --with-model # also load embedding model (slow)
#
# Requires: uv  (https://docs.astral.sh/uv/)

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK=0
WITH_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --with-model) WITH_MODEL=1 ;;
    -h|--help)
      echo "Usage: bash scripts/check_local.sh [--quick] [--with-model]"
      exit 0
      ;;
  esac
done

PASS=0
FAIL=0
WARN=0
SKIP=0

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
blue()  { printf '\033[34m%s\033[0m\n' "$*"; }

ok()   { green "  ✓ $*"; PASS=$((PASS + 1)); }
bad()  { red   "  ✗ $*"; FAIL=$((FAIL + 1)); }
warn() { yellow "  ! $*"; WARN=$((WARN + 1)); }
skip() { printf '  – %s\n' "$*"; SKIP=$((SKIP + 1)); }

section() { echo; blue "== $* =="; }

run_uv() {
  uv run "$@"
}

# ── 0. Project root ───────────────────────────────────────────────────────
section "Project root"
if [[ -f "$ROOT/pyproject.toml" && -d "$ROOT/src" ]]; then
  ok "Root looks valid: $ROOT"
else
  bad "Run this from the repo (missing pyproject.toml or src/)"
  exit 1
fi

# ── 1. uv ─────────────────────────────────────────────────────────────────
section "Package manager (uv)"
if command -v uv >/dev/null 2>&1; then
  ok "uv found: $(uv --version 2>/dev/null | head -1)"
else
  bad "uv not on PATH — install: https://docs.astral.sh/uv/"
  exit 1
fi

if [[ -d "$ROOT/.venv" ]]; then
  ok ".venv exists"
else
  warn ".venv missing — run: uv sync"
fi

# ── 2. Core imports ───────────────────────────────────────────────────────
section "Python imports (uv run)"
if run_uv python -c "
import sys
mods = [
    'streamlit', 'chromadb', 'sentence_transformers', 'numpy', 'pandas',
    'dotenv', 'requests', 'torch', 'google.genai', 'openai',
]
failed = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        failed.append(f'{m}: {e}')
if failed:
    print('FAIL')
    for f in failed:
        print(f)
    sys.exit(1)
print('OK')
print('python', sys.version.split()[0])
" 2>/tmp/aps_check_import.err; then
  ok "Core packages import"
  pyv=$(run_uv python -c "import sys; print(sys.version.split()[0])" 2>/dev/null || true)
  [[ -n "$pyv" ]] && ok "Python $pyv"
else
  bad "Import failure — run: uv sync"
  sed -n '1,20p' /tmp/aps_check_import.err 2>/dev/null || true
fi

# Optional pymupdf4llm
if run_uv python -c "import pymupdf4llm" 2>/dev/null; then
  ok "pymupdf4llm import"
else
  warn "pymupdf4llm not importable (PDF→MD may fail)"
fi

# ── 3. Project modules ────────────────────────────────────────────────────
section "Project modules"
if run_uv python -c "
import sys
from pathlib import Path
root = Path('.').resolve()
sys.path[:0] = [str(root), str(root / 'src')]
from settings import (
    PROJECT_ROOT, DATA_DIR, RAG_CACHE_DIR, EMBEDDING_MODEL,
    GEMINI_MODEL, CORPUS_PATH, CHROMA_RAG_DIR,
)
assert RAG_CACHE_DIR is not None
print('EMBEDDING_MODEL=', EMBEDDING_MODEL)
print('GEMINI_MODEL=', GEMINI_MODEL)
print('PROJECT_ROOT=', PROJECT_ROOT)
" 2>/tmp/aps_check_settings.err; then
  ok "settings.py loads (paths + models)"
  run_uv python -c "
import sys
from pathlib import Path
sys.path[:0] = ['.', 'src']
from settings import EMBEDDING_MODEL, GEMINI_MODEL
print('  embedder:', EMBEDDING_MODEL)
print('  gemini:  ', GEMINI_MODEL)
" 2>/dev/null || true
else
  bad "settings.py failed"
  cat /tmp/aps_check_settings.err 2>/dev/null | tail -20
fi

for mod in corpus_builder corpus_index ranking recommend; do
  if run_uv python -c "
import sys
sys.path[:0] = ['.', 'src']
import $mod
" 2>/tmp/aps_mod.err; then
    ok "import $mod"
  else
    bad "import $mod"
    tail -5 /tmp/aps_mod.err 2>/dev/null || true
  fi
done

if run_uv python -c "
import sys
sys.path[:0] = ['.', 'src']
from rag import embed_store, rag_pipeline, cli, ingest
" 2>/tmp/aps_rag.err; then
  ok "import rag package (embed_store, rag_pipeline, cli, ingest)"
else
  bad "import rag package"
  tail -8 /tmp/aps_rag.err 2>/dev/null || true
fi

if run_uv python -c "
import sys
sys.path[:0] = ['.', 'src']
import importlib.util
spec = importlib.util.spec_from_file_location('app', 'src/app.py')
# Do not exec Streamlit app (would start UI); only check file exists + syntax
" 2>/dev/null && run_uv python -m py_compile src/app.py 2>/tmp/aps_app.err; then
  ok "src/app.py syntax (py_compile)"
else
  bad "src/app.py syntax"
  cat /tmp/aps_app.err 2>/dev/null | tail -10
fi

# ── 4. Environment ────────────────────────────────────────────────────────
section "Environment (.env / keys)"
if [[ -f "$ROOT/.env" ]]; then
  ok ".env present"
else
  warn ".env missing — copy from .env.example and set GEMINI_API_KEY"
fi

run_uv python -c "
import os, sys
from pathlib import Path
sys.path[:0] = ['.', 'src']
from dotenv import load_dotenv
load_dotenv()
key = (os.getenv('GEMINI_API_KEY') or '').strip()
if key:
    print('GEMINI_OK')
else:
    print('GEMINI_MISSING')
fb = (os.getenv('FALLBACK_LLM_API_KEY') or os.getenv('NVIDIA_API_KEY') or '').strip()
print('FALLBACK', 'OK' if fb else 'MISSING')
" 2>/tmp/aps_env.err | while read -r line; do
  case "$line" in
    GEMINI_OK) ok "GEMINI_API_KEY is set" ;;
    GEMINI_MISSING) warn "GEMINI_API_KEY not set — Ask generation will fail" ;;
    FALLBACK\ OK) ok "Fallback LLM key is set" ;;
    FALLBACK\ MISSING) skip "Fallback LLM key not set (optional)" ;;
  esac
done

# ── 5. Data layout ────────────────────────────────────────────────────────
section "Data directories"
run_uv python -c "
import sys
sys.path[:0] = ['.', 'src']
from settings import DATA_DIR, CORPUS_PATH, CHROMA_RAG_DIR, CORPUS_CHROMA_DIR
from pathlib import Path
print('DATA_DIR', DATA_DIR, DATA_DIR.is_dir())
print('CORPUS', CORPUS_PATH, CORPUS_PATH.exists())
print('CHROMA_CORPUS', CORPUS_CHROMA_DIR.exists())
print('CHROMA_RAG', CHROMA_RAG_DIR.exists())
" 2>/dev/null | while read -r a b c; do
  case "$a" in
    DATA_DIR)
      if [[ "$c" == "True" ]]; then ok "data/ exists"; else warn "data/ missing (created on first run)"; fi
      ;;
    CORPUS)
      if [[ "$c" == "True" ]]; then ok "data/corpus.json exists"; else warn "No corpus yet — run corpus_builder search/from-ids"; fi
      ;;
    CHROMA_CORPUS)
      if [[ "$c" == "True" ]]; then ok "recommend index dir exists"; else warn "Recommend index not built yet"; fi
      ;;
    CHROMA_RAG)
      if [[ "$c" == "True" ]]; then ok "RAG index dir exists"; else warn "RAG index empty — ingest papers for Ask"; fi
      ;;
  esac
done

# ── 6. CLI entrypoints (help / status, no heavy model unless asked) ───────
section "CLI entrypoints"

if run_uv python src/corpus_builder.py --help >/tmp/aps_cb_help.out 2>/tmp/aps_cb_help.err; then
  ok "corpus_builder.py --help"
else
  bad "corpus_builder.py --help"
  tail -5 /tmp/aps_cb_help.err 2>/dev/null || true
fi

if run_uv python src/corpus_builder.py status >/tmp/aps_cb_st.out 2>/tmp/aps_cb_st.err; then
  ok "corpus_builder.py status"
  head -5 /tmp/aps_cb_st.out 2>/dev/null | sed 's/^/    /'
else
  # status may exit non-zero if empty; still show
  if grep -qi "paper\|corpus\|n_papers\|0" /tmp/aps_cb_st.out /tmp/aps_cb_st.err 2>/dev/null; then
    ok "corpus_builder.py status (empty or partial is OK)"
    head -5 /tmp/aps_cb_st.out /tmp/aps_cb_st.err 2>/dev/null | sed 's/^/    /'
  else
    warn "corpus_builder.py status returned an error (OK if no data yet)"
    tail -3 /tmp/aps_cb_st.err 2>/dev/null || true
  fi
fi

if run_uv python src/recommend.py --help >/tmp/aps_rec_help.out 2>/tmp/aps_rec_help.err; then
  ok "recommend.py --help"
else
  bad "recommend.py --help"
fi

if run_uv python -m src.rag.cli --help >/tmp/aps_rag_help.out 2>/tmp/aps_rag_help.err \
   || run_uv python -m src.rag.cli status >/tmp/aps_rag_st.out 2>/tmp/aps_rag_st.err; then
  ok "src.rag.cli responds"
else
  # try status alone
  if run_uv python -m src.rag.cli status >/tmp/aps_rag_st.out 2>/tmp/aps_rag_st.err; then
    ok "src.rag.cli status"
    head -8 /tmp/aps_rag_st.out 2>/dev/null | sed 's/^/    /'
  else
    warn "rag.cli status failed (OK if no index yet)"
    tail -5 /tmp/aps_rag_st.err 2>/dev/null || true
  fi
fi

# Explicit status when help path is weak
if run_uv python -m src.rag.cli status >/tmp/aps_rag_st2.out 2>/tmp/aps_rag_st2.err; then
  ok "rag.cli status"
  head -10 /tmp/aps_rag_st2.out 2>/dev/null | sed 's/^/    /'
else
  skip "rag.cli status skipped or empty index"
  head -5 /tmp/aps_rag_st2.out /tmp/aps_rag_st2.err 2>/dev/null | sed 's/^/    /' || true
fi

# ── 7. Streamlit app readiness ────────────────────────────────────────────
section "Streamlit app readiness"
if run_uv python -c "import streamlit; print(streamlit.__version__)" 2>/tmp/aps_st.err; then
  ver=$(run_uv python -c "import streamlit; print(streamlit.__version__)" 2>/dev/null)
  ok "streamlit $ver"
else
  bad "streamlit not importable"
fi

# Compile-only already done; check app can be located
if [[ -f "$ROOT/src/app.py" ]]; then
  ok "Main module path: src/app.py"
else
  bad "src/app.py missing"
fi

# Dry-run: ensure streamlit CLI module works
if run_uv python -m streamlit --version >/tmp/aps_stv.out 2>/tmp/aps_stv.err; then
  ok "python -m streamlit --version → $(tr -d '\n' </tmp/aps_stv.out)"
else
  bad "python -m streamlit failed — use: uv run python -m streamlit run src/app.py"
  cat /tmp/aps_stv.err 2>/dev/null | tail -5
fi

# ── 8. Optional: light network / model ────────────────────────────────────
section "Optional deeper checks"
if [[ "$QUICK" -eq 1 ]]; then
  skip "--quick: skipping arXiv ping and model load"
elif [[ "$WITH_MODEL" -eq 1 ]]; then
  echo "  Loading embedding model (may take 1–2 minutes)…"
  if run_uv python -c "
import sys
sys.path[:0] = ['.', 'src']
from rag.embed_store import load_embedding_model, embed_texts
m = load_embedding_model()
v = embed_texts(['neural tangent kernel'], model=m)
assert v and len(v[0]) > 8
print('dim', len(v[0]))
" 2>/tmp/aps_model.err; then
    ok "Embedding model loads and encodes a test string"
  else
    bad "Embedding model load/encode failed"
    tail -15 /tmp/aps_model.err 2>/dev/null || true
  fi
else
  skip "Model load not run (pass --with-model to test EmbeddingGemma / configured embedder)"
fi

if [[ "$QUICK" -eq 1 ]]; then
  skip "--quick: skipping arXiv HTTP check"
else
  if run_uv python -c "
import requests
r = requests.get('https://export.arxiv.org/api/query', params={'search_query': 'all:test', 'max_results': 1}, timeout=20)
r.raise_for_status()
assert 'feed' in r.text.lower() or 'entry' in r.text.lower() or len(r.text) > 50
print('arxiv_ok')
" 2>/tmp/aps_arxiv.err; then
    ok "arXiv API reachable"
  else
    warn "arXiv API check failed (network/firewall) — search may not work offline"
    tail -3 /tmp/aps_arxiv.err 2>/dev/null || true
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo
blue "======== Summary ========"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "  Warnings: $WARN"
echo "  Skipped: $SKIP"
echo

if [[ "$FAIL" -gt 0 ]]; then
  red "Some checks failed. Fix the items marked ✗ then re-run."
  echo
  echo "Common fixes:"
  echo "  uv sync"
  echo "  cp .env.example .env   # set GEMINI_API_KEY"
  echo "  uv run python src/corpus_builder.py search \"your topic\" -n 10 --build-index"
  exit 1
fi

green "Local environment looks usable for CLI + Streamlit."
echo
echo "Next steps:"
echo "  uv run python src/corpus_builder.py search \"mechanistic interpretability\" -n 12 --build-index"
echo "  uv run python src/recommend.py \"mechanistic interpretability circuits\" -k 8"
echo "  uv run python -m src.rag.cli ingest <arxiv_id>"
echo "  uv run python -m src.rag.cli ask \"Your question\""
echo "  uv run python -m streamlit run src/app.py"
echo
echo "Deeper model check:  bash scripts/check_local.sh --with-model"
exit 0
