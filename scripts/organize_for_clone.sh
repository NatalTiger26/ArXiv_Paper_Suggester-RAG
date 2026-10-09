#!/usr/bin/env bash
# Move one-off collection scripts out of src/ so the active pipeline is obvious.
# Run from the project root:  bash scripts/organize_for_clone.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p legacy

LEGACY_SRC=(
  arxiv_paper_puller.py
  collect.py
  compute_impact_scores.py
  consolidate_corpus.py
  literature_collector.py
  literature_collector_v2.py
  phase2_personal_ranking.py
  quick_search.py
  test_embedding_chroma.py
)

for f in "${LEGACY_SRC[@]}"; do
  if [[ -f "src/$f" ]]; then
    mv "src/$f" "legacy/$f"
    echo "moved src/$f -> legacy/$f"
  fi
done

mkdir -p rankings/archive
for f in to_rank.csv ranking_sample.json ranking_notes.txt; do
  if [[ -f "rankings/$f" ]]; then
    mv "rankings/$f" "rankings/archive/$f"
    echo "moved rankings/$f -> rankings/archive/$f"
  fi
done

for f in data/corpus_all.jsonl data/corpus_100.csv data/papers.db; do
  if [[ -f "$f" ]]; then
    rm -f "$f"
    echo "removed $f"
  fi
done

echo ""
echo "Done. Active runtime code is only:"
echo "  src/recommend.py"
echo "  src/phase3_embeddings.py"
echo "  src/phase4_scoring.py"
echo "  src/phase5_evaluation.py"
echo "  src/rag/*"
