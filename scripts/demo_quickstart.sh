#!/usr/bin/env bash
# One-command quickstart demo for ArXiv Paper Suggester
# ConfAI 2026 Demonstration Script

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================================================================="
echo " ArXiv Paper Suggester — One-Command Quickstart Demo"
echo " ConfAI 2026: Multi-Signal Ranking & Citation-Grounded RAG"
echo "================================================================="

echo -e "\n[Step 1/4] Ensuring curated papers are ingested..."
uv run python -m src.rag.cli ingest 1806.07572 2501.16496

echo -e "\n[Step 2/4] Multi-Signal Ranking for Research Interest:"
echo " Query: 'mechanistic interpretability circuits' (Top-5 with inspectable weights)"
uv run python src/recommend.py "mechanistic interpretability circuits" -k 5

echo -e "\n[Step 3/4] Grounded Q&A with Citations (Answerable query):"
echo " Question: 'What is the neural tangent kernel?'"
uv run python -m src.rag.cli ask "What is the neural tangent kernel?" -k 4

echo -e "\n[Step 4/4] Out-of-Scope Query (Honest refusal demonstration):"
echo " Question: 'What is the exact critical temperature of the 2D Ising model?'"
uv run python -m src.rag.cli ask "What is the exact critical temperature of the 2D Ising model?" -k 4

echo -e "\n================================================================="
echo " Demo complete! Launch interactive web UI:"
echo "   make web    (or: uv run streamlit run src/app.py)"
echo "================================================================="
