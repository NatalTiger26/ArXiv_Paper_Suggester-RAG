"""High-level ingest: download → markdown → chunk → embed for one or more papers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .chunk import chunk_markdown, is_bad_title
from .config import RANKINGS_PATH
from .download import download_paper, fetch_arxiv_title, normalise_arxiv_id
from .embed_store import load_embedding_model, upsert_chunks
from .pdf_to_md import pdf_to_markdown


def resolve_title(arxiv_id: str, md_path: Path | None = None) -> str:
    """Prefer arXiv API title; fall back to Markdown extraction."""
    title = fetch_arxiv_title(arxiv_id)
    if title and not is_bad_title(title):
        return title
    if md_path and md_path.exists():
        from .chunk import _extract_title

        t = _extract_title(md_path.read_text(encoding="utf-8"))
        if not is_bad_title(t):
            return t
    return title or "Untitled"


def ingest_paper(
    arxiv_id: str,
    model=None,
    force_download: bool = False,
) -> dict:
    """Full pipeline for a single paper."""
    aid = normalise_arxiv_id(arxiv_id)
    if not aid:
        return {"arxiv_id": arxiv_id, "ok": False, "error": "invalid id"}

    try:
        pdf = download_paper(aid, force=force_download)
        md_path = pdf_to_markdown(pdf, arxiv_id=aid, force=force_download)
        title = resolve_title(aid, md_path)
        chunks = chunk_markdown(md_path, paper_id=aid, title=title)
        n = upsert_chunks(chunks, model=model)
        return {
            "arxiv_id": aid,
            "ok": True,
            "title": title,
            "pdf": str(pdf),
            "markdown": str(md_path),
            "chunks": n,
        }
    except Exception as e:
        return {"arxiv_id": aid, "ok": False, "error": str(e)}


def ingest_papers(ids: Iterable[str], force_download: bool = False) -> list[dict]:
    """Ingest many papers; load the embedding model once."""
    model = load_embedding_model()
    results = []
    for raw in ids:
        print(f"[ingest] {raw} …")
        results.append(ingest_paper(raw, model=model, force_download=force_download))
    return results


def ids_from_personal_ranking(
    top_n: int = 10,
    min_score: float = 4.0,
    path: Path | str | None = None,
) -> list[str]:
    """
    Read rankings/personal_importance.csv and return arXiv IDs
    for papers with score >= min_score, ordered by score desc.
    """
    p = Path(path) if path else RANKINGS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Personal ranking not found: {p}")

    rows = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_raw = row.get("score_1_to_5") or row.get("score") or ""
            try:
                score = float(score_raw)
            except ValueError:
                continue
            if score < min_score:
                continue
            aid = (
                normalise_arxiv_id(row.get("arxiv_id") or "")
                or normalise_arxiv_id(row.get("paper_id") or "")
            )
            if not aid:
                continue
            rows.append((score, aid, row.get("title") or ""))

    rows.sort(key=lambda x: (-x[0], x[1]))
    seen = set()
    out = []
    for _, aid, _ in rows:
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
        if len(out) >= top_n:
            break
    return out


def ingest_from_personal_ranking(
    top_n: int = 10,
    min_score: float = 4.0,
    force_download: bool = False,
) -> list[dict]:
    ids = ids_from_personal_ranking(top_n=top_n, min_score=min_score)
    print(f"[ingest] personal ranking → {len(ids)} ids (score>={min_score}, top_n={top_n})")
    return ingest_papers(ids, force_download=force_download)
