"""RAG configuration facade: re-exports central project settings from src.settings."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from settings import (  # type: ignore
        CHROMA_RAG_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        CORPUS_CHROMA_DIR,
        CORPUS_PATH,
        DATA_DIR,
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
        IS_STREAMLIT_CLOUD,
        MARKDOWN_DIR,
        MIN_CHUNK_CHARS,
        MIN_RELEVANCE_SCORE,
        PAPERS_DIR,
        PROJECT_ROOT,
        RAG_CACHE_DIR,
        RAG_COLLECTION,
        RANKINGS_PATH,
        SEMANTIC_CACHE_ENABLED,
        SEMANTIC_CACHE_PATH,
        SEMANTIC_CACHE_THRESHOLD,
        TEMPERATURE,
        get_gemini_model,
    )
except ImportError:
    from src.settings import (
        CHROMA_RAG_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        CORPUS_CHROMA_DIR,
        CORPUS_PATH,
        DATA_DIR,
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
        IS_STREAMLIT_CLOUD,
        MARKDOWN_DIR,
        MIN_CHUNK_CHARS,
        MIN_RELEVANCE_SCORE,
        PAPERS_DIR,
        PROJECT_ROOT,
        RAG_CACHE_DIR,
        RAG_COLLECTION,
        RANKINGS_PATH,
        SEMANTIC_CACHE_ENABLED,
        SEMANTIC_CACHE_PATH,
        SEMANTIC_CACHE_THRESHOLD,
        TEMPERATURE,
        get_gemini_model,
    )

COLLECTION_NAME = RAG_COLLECTION
CACHE_DIR = RAG_CACHE_DIR
