"""
Central project settings (paths + models).

Embeddings — EmbeddingGemma 2 only (local):
  EMBEDDING_MODEL=google/embeddinggemma-2
  EMBEDDING_DIM=0          # 0 = full model dim; try 512 / 256 / 128 to save cost

Generation — Gemini primary, OpenAI-compatible failsafe (e.g. NVIDIA):
  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-3.8-flash
  FALLBACK_LLM_API_KEY=...           # or NVIDIA_API_KEY
  FALLBACK_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
  FALLBACK_LLM_MODEL=meta/llama-3.1-8b-instruct
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RANKINGS_DIR = PROJECT_ROOT / "rankings"

CORPUS_PATH = DATA_DIR / "corpus.json"
CORPUS_CHROMA_DIR = DATA_DIR / "chroma_corpus"
CORPUS_EMB_CACHE = DATA_DIR / "embeddings" / "corpus_embeddings.json"
CORPUS_COLLECTION = "arxiv_corpus"

PAPERS_DIR = DATA_DIR / "papers"
MARKDOWN_DIR = DATA_DIR / "markdown"
CHROMA_RAG_DIR = DATA_DIR / "chroma_rag"
RAG_CACHE_DIR = DATA_DIR / "rag_cache"
RAG_COLLECTION = "paper_chunks"
RANKINGS_PATH = RANKINGS_DIR / "personal_importance.csv"
EVAL_DIR = RESULTS_DIR / "rag_eval"

# ── Embeddings: EmbeddingGemma 2 only ─────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or os.getenv(
    "RAG_EMBEDDING_MODEL", "google/embeddinggemma-2"
)
# Matryoshka truncate. 0 = full dimension from the model (typically 768).
# Lower values (512, 256, 128) reduce storage/compute; rebuild indexes after change.
try:
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0"))
except ValueError:
    EMBEDDING_DIM = 0

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

# ── Generation ────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

FALLBACK_LLM_API_KEY = (
    os.getenv("FALLBACK_LLM_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("NGC_API_KEY")
)
FALLBACK_LLM_BASE_URL = os.getenv(
    "FALLBACK_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
FALLBACK_LLM_MODEL = os.getenv(
    "FALLBACK_LLM_MODEL", "meta/llama-3.1-8b-instruct"
)

# RAG retrieval / chunking
DEFAULT_TOP_K = 6
DEFAULT_MAX_CONTEXT_CHARS = 12_000
MIN_RELEVANCE_SCORE = 0.55
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
MIN_CHUNK_CHARS = 80

SEMANTIC_CACHE_ENABLED = os.getenv("RAG_SEMANTIC_CACHE", "1") not in ("0", "false", "False")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("RAG_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_PATH = RAG_CACHE_DIR / "semantic_answers.json"

DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 90

for d in (
    DATA_DIR,
    PAPERS_DIR,
    MARKDOWN_DIR,
    CHROMA_RAG_DIR,
    RAG_CACHE_DIR,
    EVAL_DIR,
    DATA_DIR / "embeddings",
):
    d.mkdir(parents=True, exist_ok=True)
