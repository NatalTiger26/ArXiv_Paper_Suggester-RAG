"""RAG config — re-exports shared project settings."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from settings import (  # noqa: E402
    CHROMA_RAG_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    DOWNLOAD_RETRIES,
    DOWNLOAD_TIMEOUT,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EVAL_DIR,
    FALLBACK_LLM_API_KEY,
    FALLBACK_LLM_BASE_URL,
    FALLBACK_LLM_MODEL,
    GEMINI_MODEL,
    HF_TOKEN,
    MARKDOWN_DIR,
    MIN_CHUNK_CHARS,
    MIN_RELEVANCE_SCORE,
    PAPERS_DIR,
    PROJECT_ROOT,
    RAG_COLLECTION as COLLECTION_NAME,
    RANKINGS_PATH,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_THRESHOLD,
    TEMPERATURE,
)
from settings import RAG_CACHE_DIR as CACHE_DIR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
