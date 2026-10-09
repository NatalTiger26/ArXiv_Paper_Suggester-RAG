"""Lightweight semantic cache for repeated RAG questions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_THRESHOLD,
)


def _load() -> list[dict[str, Any]]:
    if not SEMANTIC_CACHE_PATH.exists():
        return []
    try:
        return json.loads(SEMANTIC_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    SEMANTIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEMANTIC_CACHE_PATH.write_text(
        json.dumps(entries[-200:], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def lookup(
    question: str,
    paper_ids: list[str] | None,
    embed_fn,
) -> dict[str, Any] | None:
    """Return a cached answer if a near-duplicate question exists."""
    if not SEMANTIC_CACHE_ENABLED:
        return None
    entries = _load()
    if not entries:
        return None

    q_scope = ",".join(sorted(paper_ids or []))
    qvec = embed_fn([question])[0]

    best = None
    best_score = -1.0
    for e in entries:
        if e.get("scope") != q_scope:
            continue
        ev = e.get("embedding")
        if not ev or len(ev) != len(qvec):
            continue
        # cosine similarity (vectors are L2-normalised)
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
    embed_fn,
) -> None:
    if not SEMANTIC_CACHE_ENABLED:
        return
    # Do not cache explicit "not covered" empty answers as strongly
    if not result.get("references") and "not covered" in (result.get("answer") or "").lower():
        return

    qvec = embed_fn([question])[0]
    entries = _load()
    entries.append(
        {
            "question": question,
            "scope": ",".join(sorted(paper_ids or [])),
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
