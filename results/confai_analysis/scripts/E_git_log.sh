#!/usr/bin/env bash
# E. Git history for gold, eval, prompts, settings
# Run from project root (your git clone):
#   bash results/confai_analysis/E_git_log.sh | tee results/confai_analysis/E_git_log.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/confai_analysis/E_git_log.txt"
{
  echo "=== git rev-parse ==="
  git rev-parse --show-toplevel 2>/dev/null || echo "NOT A GIT REPO"
  echo
  for f in \
    results/rag_eval/gold_questions.json \
    src/final_eval.py \
    src/settings.py \
    src/rag/rag_pipeline.py \
    src/rag/config.py \
    src/rag/evaluate.py
  do
    echo "======== $f ========"
    if [[ -f "$f" ]]; then
      git log --follow --format='%h %ci %an %s' -- "$f" | head -30
    else
      echo "(missing)"
    fi
    echo
  done
  echo "======== commits touching results/rag_eval after 2026-10-08 ========"
  git log --since='2026-10-08' --format='%h %ci %s' -- results/rag_eval/ | head -40
} | tee "$OUT"
echo "Wrote $OUT"
