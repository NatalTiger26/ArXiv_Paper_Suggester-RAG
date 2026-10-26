"""Lightweight semantic cache for repeated RAG questions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_THRESHOLD,
)

_cached_entries: list[dict[str, Any]] | None = None
_cached_mtime: float = 0.0


def _load() -> list[dict[str, Any]]:
    global _cached_entries, _cached_mtime
    if not SEMANTIC_CACHE_PATH.exists():
        _cached_entries = []
        return []
    try:
        mtime = SEMANTIC_CACHE_PATH.stat().st_mtime
        if _cached_entries is not None and mtime == _cached_mtime:
            return _cached_entries
        raw = json.loads(SEMANTIC_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            _cached_entries = raw
            _cached_mtime = mtime
            return raw
        _cached_entries = []
        return []
    except Exception:
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    global _cached_entries, _cached_mtime
    try:
        SEMANTIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        trimmed = entries[-250:]
        SEMANTIC_CACHE_PATH.write_text(
            json.dumps(trimmed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _cached_entries = trimmed
        _cached_mtime = SEMANTIC_CACHE_PATH.stat().st_mtime
    except Exception as e:
        print(f"[cache] save warning: {e}", flush=True)


def lookup(
    question: str,
    paper_ids: list[str] | None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    query_vector: list[float] | None = None,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a cached answer if a near-duplicate question exists.

    Accepts precomputed query_vector for single-pass embedding efficiency.
    """
    if not SEMANTIC_CACHE_ENABLED:
        return None
    entries = _load()
    if not entries:
        return None

    if query_vector is None:
        if embed_fn is None:
            return None
        query_vector = embed_fn([question])[0]

    q_scope = ",".join(sorted(str(p).strip() for p in (paper_ids or []) if p))
    qvec = query_vector

    best = None
    best_score = -1.0
    for e in entries:
        if e.get("scope") != q_scope:
            continue
        # If cache entry specifies a generation model and caller specified one, ensure match
        if model_id and e.get("model_id") and e.get("model_id") != model_id:
            continue
        ev = e.get("embedding")
        if not ev or len(ev) != len(qvec):
            continue
        # cosine similarity (both vectors L2-normalized)
        score = sum(a * b for a, b in zip(qvec, ev))
        if score > best_score:
            best_score = score
            best = e

    if best is not None and best_score >= SEMANTIC_CACHE_THRESHOLD:
        out = dict(best["result"])
        out["cache_hit"] = True
        out["cache_score"] = round(float(best_score), 4)
        return out
    return None


def store(
    question: str,
    paper_ids: list[str] | None,
    result: dict[str, Any],
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    query_vector: list[float] | None = None,
    model_id: str | None = None,
) -> None:
    if not SEMANTIC_CACHE_ENABLED:
        return

    # Do not cache explicit "not covered", refusals, or empty/low coverage answers
    ans_lower = (result.get("answer") or "").lower()
    cov = str(result.get("coverage") or "").lower()
    if cov in ("none", "low"):
        return
    if not result.get("references") and any(
        w in ans_lower
        for w in ("not covered", "do not contain", "cannot answer", "no relevant")
    ):
        return

    if query_vector is None:
        if embed_fn is None:
            return
        query_vector = embed_fn([question])[0]

    qvec = query_vector
    entries = _load()
    entries.append(
        {
            "question": question,
            "scope": ",".join(sorted(str(p).strip() for p in (paper_ids or []) if p)),
            "model_id": model_id or "",
            "embedding": qvec,
            "result": {
                "answer": result.get("answer"),
                "references": result.get("references"),
                "coverage": result.get("coverage"),
            },
            "ts": time.time(),
        }
    )
    _save(entries)
