#!/usr/bin/env bash
# Fast repository hygiene & smoke check.
# Usage:
#   bash scripts/check_repo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "======================================================="
echo " Repository Hygiene & Smoke Check"
echo "======================================================="

PASS=0
FAIL=0

ok()   { printf "\033[32m  ✓ %s\033[0m\n" "$*"; PASS=$((PASS + 1)); }
bad()  { printf "\033[31m  ✗ %s\033[0m\n" "$*"; FAIL=$((FAIL + 1)); }

# 1. Git Secrets & Ignored Files Check
echo "[1/6] Checking git tracking and secrets..."
if git ls-files .env | grep -q ".env"; then
  bad ".env file is tracked by git! Unstage/remove it immediately."
else
  ok ".env is not tracked by git"
fi

if git ls-files data/ | grep -E "(chroma_|\.pdf$|\.sqlite3$|\.sqlite$)" | grep -q .; then
  bad "Large binary or index data is tracked in git!"
else
  ok "No binary/index files tracked under data/"
fi

# Secret patterns scan
if grep -rnE --exclude=".env" --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="data" --exclude-dir="backups" --exclude="*.sqlite3" --exclude="*.sqlite" "AIzaSy[A-Za-z0-9_-]{33}" . | grep -v "\.env\.example" | grep -q .; then
  bad "Potential Gemini API key detected in source code!"
else
  ok "No raw Google API keys found in tracked source files"
fi

# 2. uv Package Manager & Lockfile Check
echo "[2/6] Verifying uv package manager & uv.lock sync..."
if command -v uv >/dev/null 2>&1; then
  uv lock --check >/dev/null 2>&1 && ok "uv.lock is synchronized with pyproject.toml" || bad "uv.lock out of sync! Run: uv lock"
else
  bad "uv is not installed on PATH! Install via: curl -LsSf https://astral.sh/uv/install.sh"
fi

# 3. Module Import Smoke Tests
echo "[3/6] Running module import smoke tests..."
uv run python -c "
import sys
modules = [
    'src.settings',
    'src.corpus_builder',
    'src.corpus_index',
    'src.ranking',
    'src.recommend',
    'src.rag.embed_store',
    'src.rag.rag_pipeline',
    'src.rag.cli',
    'src.final_eval',
    'src.citation_library.store',
    'src.novelty.refs',
    'src.telemetry',
]
for m in modules:
    try:
        __import__(m)
        print(f'  ✓ {m}')
    except Exception as e:
        print(f'  ✗ {m}: {e}', file=sys.stderr)
        sys.exit(1)
" && ok "All core modules import cleanly" || bad "Module import error"

# 4. CLI --help smoke tests
echo "[4/6] Running CLI entrypoint --help tests..."
uv run python src/corpus_builder.py --help >/dev/null && ok "corpus_builder.py --help" || bad "corpus_builder.py --help failed"
uv run python src/recommend.py --help >/dev/null && ok "recommend.py --help" || bad "recommend.py --help failed"
uv run python -m src.rag.cli --help >/dev/null && ok "python -m src.rag.cli --help" || bad "rag.cli --help failed"
uv run python src/final_eval.py --help >/dev/null && ok "final_eval.py --help" || bad "final_eval.py --help failed"

# 5. Unit Test Suite
echo "[5/6] Running core unit test suite..."
TEST_OUT=$(uv run python -m unittest discover -s tests 2>&1)
if [ $? -eq 0 ]; then
  TEST_COUNT=$(echo "$TEST_OUT" | grep -o 'Ran [0-9]* tests' | awk '{print $2}')
  ok "Unit test suite passed (${TEST_COUNT:-all} tests)"
else
  bad "Unit test failures"
fi

# 6. Optional Ruff / formatting check
echo "[6/6] Checking linter/formatter availability..."
if command -v ruff >/dev/null 2>&1; then
  ruff check src/ scripts/ --select E,F,W --ignore E501,E402 && ok "ruff linter passed" || bad "ruff linter warnings"
elif uv run ruff --version >/dev/null 2>&1; then
  uv run ruff check src/ scripts/ --select E,F,W --ignore E501,E402 && ok "uv ruff linter passed" || bad "uv ruff linter warnings"
else
  echo "  – ruff not installed in env (skipped)"
fi

echo "======================================================="
if [ "$FAIL" -eq 0 ]; then
  printf "\033[32mPassed: %d, Failed: 0. Repository is clean and healthy!\033[0m\n" "$PASS"
  exit 0
else
  printf "\033[31mPassed: %d, Failed: %d.\033[0m\n" "$PASS" "$FAIL"
  exit 1
fi
