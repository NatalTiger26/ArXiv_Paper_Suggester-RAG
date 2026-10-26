#!/usr/bin/env bash
# Run on the local machine with full Chroma + GEMINI key as needed.
# Does not modify src/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/confai_analysis

echo "=== E git log ==="
bash results/confai_analysis/E_git_log.sh || true

echo "=== A presence ==="
uv run python results/confai_analysis/A_presence_check.py

echo "=== C top-1 mismatch sample ==="
uv run python results/confai_analysis/C_top1_mismatch_sample.py --seed 42

echo "=== B index-size sweep (long) ==="
uv run python results/confai_analysis/B_index_size_sweep.py --sizes 10,100,1000,full --seeds 3

echo "=== D OOS full answers (needs GEMINI_API_KEY; long) ==="
uv run python results/confai_analysis/D_oos_full_answers.py --no-cache

echo "Done. Outputs under results/confai_analysis/"
