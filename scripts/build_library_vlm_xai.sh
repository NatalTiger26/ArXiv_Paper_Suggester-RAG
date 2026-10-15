#!/usr/bin/env bash
# Build / extend library for VLM explainability, Score-CAM, stability, attention.
# Uses arXiv-friendly term ANDs (not Scholar-only exact phrases).
#
# Usage (project root):
#   chmod +x scripts/build_library_vlm_xai.sh
#   caffeinate -dims bash scripts/build_library_vlm_xai.sh
#
# Env:
#   N=25              hits per (query, category) pair  (default 25)
#   SLEEP=3           pause between API calls
#   CATEGORIES=...    space-separated cats (default: cs.CV cs.LG cs.AI cs.CL)
#   NO_CATEGORY=1     also run each query once with no category filter

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH" >&2
  exit 1
fi

N="${N:-25}"
SLEEP="${SLEEP:-3}"
NO_CATEGORY="${NO_CATEGORY:-1}"
# shellcheck disable=SC2206
CATEGORIES=(${CATEGORIES:-cs.CV cs.LG cs.AI cs.CL})

mkdir -p logs
LOG="logs/library_build_vlm_xai_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Project     : $ROOT"
echo "Log         : $LOG"
echo "N           : $N"
echo "SLEEP       : ${SLEEP}s"
echo "CATEGORIES  : ${CATEGORIES[*]}"
echo "NO_CATEGORY : $NO_CATEGORY"
echo "Started     : $(date -Iseconds 2>/dev/null || date)"
echo ""

# Short, high-signal phrases. Long Scholar Boolean strings are expanded by
# corpus_builder into term-ANDs; keep each query to ~2–5 content words.
QUERIES=(
  # Core VLM / multimodal XAI
  "vision-language explainability"
  "vision-language interpretability"
  "multimodal explainability"
  "multimodal interpretability"
  "VLM explainability"
  "VLM interpretability"
  "explainable AI multimodal"
  "multimodal XAI"

  # Method names (phrase-friendly)
  "Score-CAM"
  "Grad-CAM"
  "Grad-CAM++"
  "class activation mapping"
  "HiResCAM"
  "Layer-CAM"
  "Eigen-CAM"

  # Attribution / saliency
  "feature attribution"
  "visual attribution"
  "saliency maps"
  "pixel attribution"
  "gradient-free attribution"
  "perturbation-based explanation"
  "occlusion sensitivity"
  "multimodal feature attribution"
  "visual token attribution"
  "patch relevance propagation"

  # Stability / robustness / guarantees
  "explanation stability"
  "attribution robustness"
  "saliency robustness"
  "certified explanations"
  "certified attribution"
  "explanation consistency"
  "saliency invariance"
  "perturbation stability explanation"
  "formal guarantees attribution"
  "provable explainability"
  "Lipschitz saliency"
  "attributional robustness"

  # Attention (esp. multimodal)
  "attention faithfulness"
  "attention attribution"
  "attention rollout"
  "cross-attention interpretability"
  "cross-modal attention"
  "attention intervention"
  "causal attention"
  "token-to-region alignment"

  # Grounding / CLIP / VLM-specific
  "image-text grounding"
  "vision-language grounding"
  "CLIP interpretability"
  "CLIP saliency"
  "CLIP explainability"
  "prompt sensitivity CLIP"
  "text-conditioned saliency"
  "language-conditioned saliency"
  "cross-modal attribution"
  "multimodal attribution consistency"
  "vision-language robustness"

  # Evaluation of explanations
  "explanation faithfulness"
  "faithfulness metrics"
  "deletion insertion explanation"
  "saliency sanity checks"
  "explanation fidelity"
  "counterfactual explanation multimodal"

  # Concept / causal
  "concept-based explanations"
  "concept activation"
  "causal attribution"
  "causal intervention vision-language"
  "counterfactual faithfulness"

  # Mechanistic (optional related)
  "mechanistic interpretability vision"
  "mechanistic interpretability multimodal"

  # Compact combos (2–4 words; term-AND works on arXiv)
  "Score-CAM stability"
  "Score-CAM robustness"
  "Grad-CAM vision transformer"
  "saliency certified robustness"
  "attention faithfulness multimodal"
  "CLIP attribution"
  "CLIP prompt sensitivity"
  "vision-language causal"
  "multimodal explanation faithfulness"
  "cross-modal explanation"
)

# Build work list: query × category (+ optional uncategorized)
WORK=()
for q in "${QUERIES[@]}"; do
  for cat in "${CATEGORIES[@]}"; do
    WORK+=("${q}|||${cat}")
  done
  if [[ "$NO_CATEGORY" == "1" ]]; then
    WORK+=("${q}|||")
  fi
done

TOTAL="${#WORK[@]}"
echo "=== ${#QUERIES[@]} base queries × categories → ${TOTAL} API calls (n=${N}) ==="
echo "=== Merges into existing library (does not wipe) ==="
echo ""

ok=0
fail=0
empty=0
i=0

for item in "${WORK[@]}"; do
  i=$((i + 1))
  q="${item%|||*}"
  cat="${item##*|||}"

  echo "----------------------------------------------------------------"
  if [[ -n "$cat" ]]; then
    echo ">>> [${i}/${TOTAL}] [$cat] ${q}"
  else
    echo ">>> [${i}/${TOTAL}] [all-cats] ${q}"
  fi
  echo "----------------------------------------------------------------"

  if [[ -n "$cat" ]]; then
    out="$(uv run python src/corpus_builder.py search "$q" -n "$N" --category "$cat" 2>&1)" || {
      echo "$out"
      fail=$((fail + 1))
      echo "[fail] $q ($cat)"
      sleep "$SLEEP"
      continue
    }
  else
    out="$(uv run python src/corpus_builder.py search "$q" -n "$N" 2>&1)" || {
      echo "$out"
      fail=$((fail + 1))
      echo "[fail] $q (no cat)"
      sleep "$SLEEP"
      continue
    }
  fi

  echo "$out"
  if echo "$out" | grep -q "Found 0 papers"; then
    empty=$((empty + 1))
  else
    ok=$((ok + 1))
  fi
  sleep "$SLEEP"
done

echo ""
echo "=== Search summary: nonempty=${ok} empty=${empty} fail=${fail} / ${TOTAL} ==="
echo ""
echo "=== Rebuild recommend index (one pass) ==="
if uv run python src/corpus_builder.py rebuild-index; then
  echo "[ok] index rebuilt"
else
  echo "[fail] rebuild-index failed"
  echo "  uv run python src/corpus_builder.py rebuild-index"
  exit 1
fi

echo ""
uv run python src/corpus_builder.py status
echo ""
echo "Finished: $(date -Iseconds 2>/dev/null || date)"
echo "Log: $LOG"
echo "Note: library = rank only. Full-text still needs RAG ingest."
