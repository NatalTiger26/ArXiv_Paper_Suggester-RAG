#!/usr/bin/env bash
# Resumable in-library citation graph → ranked candidates to ingest.
#
# Usage (project root, uv env OK):
#   bash scripts/build_citation_library.sh
#   SOURCE=rag MIN_COUNT=5 bash scripts/build_citation_library.sh
#   LIMIT=50 bash scripts/build_citation_library.sh   # smoke test
#
# Citation counts only need reference *edges* (ids). Abstract enrichment is
# off by default so collect does not hang for minutes per paper.
# Optional: ENRICH=1 to also fill abstracts (novelty path only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SOURCE="${SOURCE:-library}"
MIN_COUNT="${MIN_COUNT:-3}"
SLEEP="${SLEEP:-0.35}"
LIMIT="${LIMIT:-0}"
TOP="${TOP:-40}"
ENRICH="${ENRICH:-0}"

echo "== citation library =="
echo "SOURCE=$SOURCE MIN_COUNT=$MIN_COUNT SLEEP=$SLEEP LIMIT=$LIMIT ENRICH=$ENRICH"
echo "Stop with Ctrl+C — re-run resumes (skips papers already ok)."
echo "Fast mode: no abstract enrichment (set ENRICH=1 only if you need novelty)."
echo

if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python)
fi

COLLECT_ARGS=(--source "$SOURCE" --sleep "$SLEEP" --limit "$LIMIT")
if [[ "$ENRICH" == "1" || "$ENRICH" == "true" || "$ENRICH" == "yes" ]]; then
  COLLECT_ARGS+=(--enrich-abstracts)
fi

"${PY[@]}" -m src.citation_library.cli collect "${COLLECT_ARGS[@]}"

"${PY[@]}" -m src.citation_library.cli report \
  --min-count "$MIN_COUNT" \
  --top "$TOP"

echo
echo "Outputs under results/citation_library/"
echo "  citation_counts.csv"
echo "  candidates_not_in_library.csv"
echo "  candidates_arxiv_ids.txt   ← use for ingest"
echo "  importance_corpus.json"
echo "  summary.md"
