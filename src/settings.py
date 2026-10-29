"""
Central project settings (paths + models).

Embeddings — local default EmbeddingGemma 2; Cloud defaults to MiniLM:
  EMBEDDING_MODEL=google/embeddinggemma-2  # or all-MiniLM-L6-v2 on Cloud
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
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env first (local). Streamlit Secrets must win on Cloud — do NOT use setdefault
# after dotenv, or a stale GEMINI_MODEL in the environment shadows Secrets.
load_dotenv(override=False)





def _detect_streamlit_cloud() -> bool:
    """True on Streamlit Community Cloud (and similar constrained hosts)."""
    if os.getenv("FORCE_CLOUD_EMBED") == "1":
        return True
    if os.getenv("FORCE_LOCAL_EMBED") == "1":
        return False
    # Common Cloud markers
    if os.path.exists("/home/adminuser"):
        return True
    if os.path.exists("/mount/src"):
        return True
    if os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT") == "cloud":
        return True
    # Hostname / path heuristics
    try:
        import socket

        hn = socket.gethostname() or ""
        if "streamlit" in hn.lower():
            return True
    except Exception:
        pass
    cwd = os.getcwd()
    if "/mount/src/" in str(cwd).replace(chr(92), "/"):
        return True
    return False


IS_STREAMLIT_CLOUD = _detect_streamlit_cloud()


def _should_check_streamlit_secrets() -> bool:
    """Only import streamlit if we are running under streamlit or on Streamlit Cloud."""
    if "streamlit" in sys.modules:
        return True
    if any("streamlit" in str(arg).lower() for arg in sys.argv):
        return True
    if IS_STREAMLIT_CLOUD:
        return True
    return False


def _apply_streamlit_secrets(*, override: bool = True) -> None:
    """Copy st.secrets into os.environ. override=True so Cloud Secrets beat .env."""
    if not _should_check_streamlit_secrets():
        return
    try:
        import contextlib
        import io

        # Suppress noisy server/CORS warnings during import in non-server contexts
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if not secrets:
            return
        keys = (
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "FALLBACK_LLM_API_KEY",
            "NVIDIA_API_KEY",
            "EMBEDDING_MODEL",
            "RAG_EMBEDDING_MODEL",
            "EMBEDDING_DIM",
            "ALLOW_LARGE_EMBED",
            "FORCE_CLOUD_EMBED",
        )
        for key in keys:
            try:
                if hasattr(secrets, "get"):
                    val = secrets.get(key)
                else:
                    val = secrets[key]
            except Exception:
                continue
            if val is None or str(val).strip() == "":
                continue
            val_s = str(val).strip()
            if override or key not in os.environ:
                prev = os.environ.get(key)
                os.environ[key] = val_s
                if prev and prev != val_s and key in ("GEMINI_MODEL", "EMBEDDING_MODEL"):
                    if os.getenv("DEBUG_SETTINGS", "0") == "1":
                        print(
                            f"[settings] Secret overrode {key}: {prev!r} → {val_s!r}",
                            flush=True,
                        )
    except Exception:
        pass


_apply_streamlit_secrets(override=True)



# ── Embeddings ────────────────────────────────────────────────────────────
# Local default: EmbeddingGemma 2 (~1GB — fine on a laptop, deadly on free Cloud).
# On Streamlit Community Cloud we ALWAYS use a small embedder unless the user
# explicitly sets ALLOW_LARGE_EMBED=1 (not recommended on free tier).
_DEFAULT_LOCAL_EMBED = "google/embeddinggemma-2"
_DEFAULT_CLOUD_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
_HEAVY_EMBED_MARKERS = (
    "embeddinggemma",
    "gemma-2",
    "e5-large",
    "gtr-t5-large",
)

_requested = (
    os.getenv("EMBEDDING_MODEL")
    or os.getenv("RAG_EMBEDDING_MODEL")
    or ""
).strip()

if IS_STREAMLIT_CLOUD and os.getenv("ALLOW_LARGE_EMBED") != "1":
    # Ignore a heavy model from .env / Secrets on free Cloud — prevents OOM kills.
    if not _requested or any(m in _requested.lower() for m in _HEAVY_EMBED_MARKERS):
        EMBEDDING_MODEL = _DEFAULT_CLOUD_EMBED
        if _requested and _requested != _DEFAULT_CLOUD_EMBED:
            print(
                f"[settings] Streamlit Cloud: refusing heavy embedder {_requested!r}; "
                f"using {_DEFAULT_CLOUD_EMBED!r}. Set ALLOW_LARGE_EMBED=1 to override.",
                flush=True,
            )
    else:
        EMBEDDING_MODEL = _requested
elif _requested:
    EMBEDDING_MODEL = _requested
else:
    EMBEDDING_MODEL = _DEFAULT_LOCAL_EMBED if not IS_STREAMLIT_CLOUD else _DEFAULT_CLOUD_EMBED

# Matryoshka truncate. Default 256 dimensions for high throughput and compact storage.
# Set EMBEDDING_DIM=768 in .env to use uncompressed full dimensions.
try:
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
except ValueError:
    EMBEDDING_DIM = 256

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)


# ── Paths ─────────────────────────────────────────────────────────────────
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

# ── Generation ────────────────────────────────────────────────────────────
def get_gemini_model() -> str:
    """Always re-read so late secret injection / env edits are visible."""
    _apply_streamlit_secrets(override=True)
    return (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()


GEMINI_MODEL = get_gemini_model()
if os.getenv("DEBUG_SETTINGS", "0") == "1":
    print(f"[settings] GEMINI_MODEL resolved to {GEMINI_MODEL!r}", flush=True)

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
