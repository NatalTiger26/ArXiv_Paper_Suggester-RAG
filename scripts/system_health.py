#!/usr/bin/env python3
"""Comprehensive system health report for ArXiv_Paper_Suggester."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.settings import (
    CHROMA_RAG_DIR,
    CORPUS_CHROMA_DIR,
    CORPUS_PATH,
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FALLBACK_LLM_BASE_URL,
    FALLBACK_LLM_MODEL,
    IS_STREAMLIT_CLOUD,
    MARKDOWN_DIR,
    PAPERS_DIR,
    RAG_CACHE_DIR,
    get_gemini_model,
)


def _fmt_size(bytes_n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_n < 1024.0:
            return f"{bytes_n:.1f} {unit}"
        bytes_n /= 1024.0
    return f"{bytes_n:.1f} TB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def print_health() -> None:
    print("\n" + "=" * 65)
    print(" ArXiv Paper Suggester — System Health & Diagnostics")
    print("=" * 65)

    # 1. Platform & Environment
    print("\n[Platform & Environment]")
    print(f"  Python Version      : {sys.version.split()[0]}")
    print(f"  Streamlit Cloud Mode: {IS_STREAMLIT_CLOUD}")
    print(f"  Project Root        : {ROOT}")

    # 2. Embedding Configuration
    print("\n[Embedding Configuration]")
    print(f"  Configured Model    : {EMBEDDING_MODEL}")
    print(f"  Active Dimension    : {'Full (default)' if EMBEDDING_DIM == 0 else f'{EMBEDDING_DIM}-d (Matryoshka)'}")

    # 3. LLM Generation
    print("\n[LLM Generation]")
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    gemini_model = get_gemini_model()
    print(f"  Gemini API Key      : {'Configured ✓' if has_gemini else 'MISSING ✗'}")
    print(f"  Gemini Model        : {gemini_model}")
    has_fallback = bool(os.getenv("FALLBACK_LLM_API_KEY") or os.getenv("NVIDIA_API_KEY"))
    print(f"  Fallback LLM        : {'Configured ✓' if has_fallback else 'Unset (optional)'}")
    if has_fallback:
        print(f"    Base URL          : {FALLBACK_LLM_BASE_URL}")
        print(f"    Model             : {FALLBACK_LLM_MODEL}")

    # 4. External Discovery APIs
    print("\n[External Discovery APIs]")
    has_s2 = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY"))
    print(f"  Semantic Scholar Key: {'Configured ✓' if has_s2 else 'Unset (polite rate limits apply)'}")
    oa_mail = os.getenv("OPENALEX_MAILTO")
    print(f"  OpenAlex Mailto     : {oa_mail if oa_mail else 'Unset'}")

    # 5. Dual Vector Store Indexes
    print("\n[Vector Store Indexes (Dual Chroma)]")
    # Recommend Index
    rec_n = 0
    if CORPUS_PATH.exists():
        try:
            rec_n = len(json.loads(CORPUS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    print(f"  Recommend Corpus    : {CORPUS_PATH} ({rec_n} papers)")
    print(f"  Recommend Index     : {CORPUS_CHROMA_DIR} ({'Exists ✓' if CORPUS_CHROMA_DIR.exists() else 'Missing ✗'})")

    # RAG Index
    try:
        from src.rag.embed_store import index_stats
        rs = index_stats()
        rag_chunks = rs.get("chunk_count", 0)
        rag_papers = rs.get("paper_count", 0)
    except Exception as e:
        rag_chunks = 0
        rag_papers = 0
    print(f"  RAG Chunk Index     : {CHROMA_RAG_DIR}")
    print(f"    Total Chunks      : {rag_chunks}")
    print(f"    Ingested Papers   : {rag_papers}")

    # 6. Caches
    print("\n[Cache Subsystems]")
    sem_cache = RAG_CACHE_DIR / "semantic_answers.json"
    sem_cnt = 0
    if sem_cache.exists():
        try:
            sem_cnt = len(json.loads(sem_cache.read_text(encoding="utf-8")))
        except Exception:
            pass
    print(f"  Semantic QA Cache   : {sem_cnt} entries ({_fmt_size(sem_cache.stat().st_size if sem_cache.exists() else 0)})")

    http_cache = DATA_DIR / "cache" / "http_cache.sqlite"
    http_cnt = 0
    if http_cache.exists():
        try:
            import sqlite3
            c = sqlite3.connect(str(http_cache))
            http_cnt = c.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0]
        except Exception:
            pass
    print(f"  API HTTP Cache      : {http_cnt} responses ({_fmt_size(http_cache.stat().st_size if http_cache.exists() else 0)})")

    # 7. Disk Usage Breakdown
    print("\n[Disk Usage Breakdown]")
    for name, p in [
        ("PDF Papers (data/papers)", PAPERS_DIR),
        ("Markdown Texts (data/markdown)", MARKDOWN_DIR),
        ("Recommend Chroma (data/chroma_corpus)", CORPUS_CHROMA_DIR),
        ("RAG Chroma (data/chroma_rag)", CHROMA_RAG_DIR),
    ]:
        size = _dir_size(p)
        print(f"  {name:<38}: {_fmt_size(size)}")

    total_data = _dir_size(DATA_DIR)
    print(f"  ---------------------------------------------------")
    print(f"  Total `data/` Directory Size          : {_fmt_size(total_data)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print_health()
