"""High-level ingest: download → markdown → chunk → embed for one or more papers."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .chunk import chunk_markdown, is_bad_title
from .config import PROJECT_ROOT, RANKINGS_PATH
from .download import download_paper, fetch_arxiv_title, normalise_arxiv_id, pdf_path_for
from .embed_store import load_embedding_model, upsert_chunks
from .pdf_to_md import _has_usable_text_layer, markdown_path_for, pdf_to_markdown


def _record_ingest_profile(record: dict) -> None:
    try:
        profile_file = PROJECT_ROOT / "results" / "ingest_profile.jsonl"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_pdf_page_count(pdf_path: Path) -> int:
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


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


def _paper_in_rag(aid: str) -> tuple[bool, int]:
    """Return (present, chunk_count) for this arXiv id in chroma_rag."""
    try:
        from .embed_store import get_collection

        coll = get_collection(create=False)
        # Prefer metadata filter; fall back to full list for older stores
        try:
            res = coll.get(where={"paper_id": aid}, include=["metadatas"], limit=5)
            n = len(res.get("ids") or [])
            if n == 0:
                # count all matching via get without limit if small
                res = coll.get(where={"paper_id": aid}, include=[])
                n = len(res.get("ids") or [])
            return (n > 0, n)
        except Exception:
            from .embed_store import list_ingested_papers

            ids = {
                str(p.get("paper_id") or p.get("arxiv_id") or "").strip()
                for p in list_ingested_papers()
            }
            return (aid in ids, -1)
    except Exception:
        return (False, 0)


def ingest_paper(
    arxiv_id: str,
    model=None,
    force_download: bool = False,
    force_embed: bool = False,
) -> dict:
    """
    Full pipeline for a single paper.

    Skips work when artifacts already exist (unless force_*):
      • PDF     — data/papers/{id}.pdf present and >1KB
      • Markdown— data/markdown/{id}.md present, non-trivial, not older than PDF
      • Embed   — paper_id already has chunks in chroma_rag (skip chunk+embed)
    """
    aid = normalise_arxiv_id(arxiv_id)
    if not aid:
        return {"arxiv_id": arxiv_id, "ok": False, "error": "invalid id"}

    t0 = time.perf_counter()
    try:
        # --- PDF ---
        dest = pdf_path_for(aid)
        pdf_reused = (
            dest.exists() and dest.stat().st_size > 1000 and not force_download
        )
        pdf = download_paper(aid, force=force_download)
        if pdf_reused:
            print(f"  [skip-download] PDF already on disk: {pdf.name}", flush=True)
        else:
            print(f"  [download] wrote {pdf.name}", flush=True)

        # --- Markdown ---
        md_dest = markdown_path_for(aid)
        md_reused = (
            not force_download
            and md_dest.exists()
            and md_dest.stat().st_size > 200
            and md_dest.stat().st_mtime >= pdf.stat().st_mtime
        )
        used_ocr = not md_reused and not _has_usable_text_layer(pdf)
        md_path = pdf_to_markdown(pdf, arxiv_id=aid, force=force_download)
        if md_reused:
            print(f"  [skip-pdf2md] Markdown already present: {md_path.name}", flush=True)
        else:
            print(f"  [pdf2md] wrote {md_path.name}", flush=True)

        pages = _get_pdf_page_count(pdf)

        # --- Already embedded in RAG? ---
        in_rag, n_existing = _paper_in_rag(aid)
        if in_rag and not force_embed and not force_download:
            title = resolve_title(aid, md_path)
            duration = round(time.perf_counter() - t0, 3)
            print(
                f"  [skip-embed] already in RAG"
                + (f" (~{n_existing} chunks)" if n_existing > 0 else ""),
                flush=True,
            )
            _record_ingest_profile({
                "arxiv_id": aid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pages": pages,
                "used_ocr": used_ocr,
                "chunks": n_existing if n_existing > 0 else 0,
                "duration_s": duration,
                "skipped_embed": True,
                "reused_pdf": pdf_reused,
                "reused_md": md_reused,
            })
            return {
                "arxiv_id": aid,
                "ok": True,
                "title": title,
                "pdf": str(pdf),
                "markdown": str(md_path),
                "chunks": n_existing if n_existing > 0 else 0,
                "skipped_embed": True,
                "reused_pdf": pdf_reused,
                "reused_md": md_reused,
                "pages": pages,
                "duration_s": duration,
            }

        title = resolve_title(aid, md_path)
        chunks = chunk_markdown(md_path, paper_id=aid, title=title)
        n = upsert_chunks(chunks, model=model)
        duration = round(time.perf_counter() - t0, 3)
        print(f"  [embed] upserted {n} chunks in {duration:.2f}s", flush=True)
        _record_ingest_profile({
            "arxiv_id": aid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
            "used_ocr": used_ocr,
            "chunks": n,
            "duration_s": duration,
            "skipped_embed": False,
            "reused_pdf": pdf_reused,
            "reused_md": md_reused,
        })
        return {
            "arxiv_id": aid,
            "ok": True,
            "title": title,
            "pdf": str(pdf),
            "markdown": str(md_path),
            "chunks": n,
            "skipped_embed": False,
            "reused_pdf": pdf_reused,
            "reused_md": md_reused,
            "pages": pages,
            "duration_s": duration,
        }
    except Exception as e:
        duration = round(time.perf_counter() - t0, 3)
        _record_ingest_profile({
            "arxiv_id": aid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "duration_s": duration,
        })
        return {"arxiv_id": aid, "ok": False, "error": str(e), "duration_s": duration}


def ingest_papers(
    ids: Iterable[str],
    force_download: bool = False,
    force_embed: bool = False,
) -> list[dict]:
    """Ingest many papers; load the embedding model once."""
    model = load_embedding_model()
    results = []
    for raw in ids:
        print(f"[ingest] {raw} …")
        results.append(
            ingest_paper(
                raw,
                model=model,
                force_download=force_download,
                force_embed=force_embed,
            )
        )
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
    return [aid for _, aid, _ in rows[:top_n]]


def ingest_from_personal_ranking(
    top_n: int = 10,
    min_score: float = 4.0,
    force_download: bool = False,
    force_embed: bool = False,
) -> list[dict]:
    ids = ids_from_personal_ranking(top_n=top_n, min_score=min_score)
    return ingest_papers(ids, force_download=force_download, force_embed=force_embed)
