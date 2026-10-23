#!/usr/bin/env bash
# Adaptive stable overnight re-ingest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARGS=(--timeout "${PAPER_TIMEOUT_SEC:-900}" --max-micro "${MAX_MICRO:-4}")
if [[ "${FRESH:-0}" == "1" ]]; then
  ARGS+=(--fresh-chroma)
fi

exec caffeinate -dims uv run python scripts/reingest_stable.py "${ARGS[@]}" "$@"
