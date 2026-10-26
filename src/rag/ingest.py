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
    """Prefer local markdown / corpus title; fall back to arXiv API only if needed."""
    if md_path and md_path.exists():
        try:
            from .chunk import _extract_title
            text = md_path.read_text(encoding="utf-8", errors="ignore")[:3000]
            t = _extract_title(text)
            if t and not is_bad_title(t):
                return t
        except Exception:
            pass
    try:
        from .embed_store import _titles_from_corpus
        ct = _titles_from_corpus([arxiv_id])
        if ct.get(arxiv_id) and not is_bad_title(ct[arxiv_id]):
            return ct[arxiv_id]
    except Exception:
        pass

    title = fetch_arxiv_title(arxiv_id)
    if title and not is_bad_title(title):
        return title
    return title or "Untitled"


def _paper_in_rag(aid: str) -> tuple[bool, int]:
    """Return (present, chunk_count) for this arXiv id in chroma_rag."""
    try:
        from .embed_store import _progress_done_ids
        done = set(_progress_done_ids())
        if aid in done:
            return (True, -1)
    except Exception:
        pass

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
    prefer_html: bool = True,
    force_pdf: bool = False,
) -> dict:
    """
    Full pipeline for a single paper.

    Fast HTML-first strategy with automatic PDF fallback:
      1. If prefer_html and not force_pdf:
         Fetch arXiv native HTML (LaTeXML/ar5iv). If 200 OK, convert directly
         to clean Markdown + structured section chunks, bypassing heavy PDF downloads.
      2. If HTML is unavailable (404 / legacy paper) or force_pdf:
         Fall back to PDF download → PyMuPDF/OCR → chunk_markdown.
      3. Skips work when paper is already in RAG unless force_* is specified.
    """
    aid = normalise_arxiv_id(arxiv_id)
    if not aid:
        return {"arxiv_id": arxiv_id, "ok": False, "error": "invalid id", "source_format": None}

    t0 = time.perf_counter()
    try:
        md_dest = markdown_path_for(aid)
        dest = pdf_path_for(aid)
        in_rag, n_existing = _paper_in_rag(aid)

        # Skip embed if already present and artifacts exist
        if in_rag and not force_embed and not force_download and md_dest.exists() and md_dest.stat().st_size > 200:
            title = resolve_title(aid, md_dest)
            duration = round(time.perf_counter() - t0, 3)
            print(
                f"  [skip-embed] already in RAG"
                + (f" (~{n_existing} chunks)" if n_existing > 0 else ""),
                flush=True,
            )
            _record_ingest_profile({
                "arxiv_id": aid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pages": 0,
                "used_ocr": False,
                "chunks": n_existing if n_existing > 0 else 0,
                "duration_s": duration,
                "skipped_embed": True,
                "reused_md": True,
            })
            return {
                "arxiv_id": aid,
                "ok": True,
                "title": title,
                "source_format": "cached",
                "pdf": str(dest) if dest.exists() else None,
                "markdown": str(md_dest),
                "chunks": n_existing if n_existing > 0 else 0,
                "skipped_embed": True,
                "duration_s": duration,
            }

        # --- Route 1: HTML-First Ingestion ---
        if prefer_html and not force_pdf:
            from .html_ingest import fetch_arxiv_html, html_to_markdown_and_chunks

            html_text = fetch_arxiv_html(aid)
            if html_text:
                title = resolve_title(aid)
                md_content, parsed_title, html_chunks = html_to_markdown_and_chunks(
                    html_text, arxiv_id=aid, title=title if title != "Untitled" else None
                )
                md_dest.parent.mkdir(parents=True, exist_ok=True)
                md_dest.write_text(md_content, encoding="utf-8")
                print(
                    f"  [html-ingest] extracted {len(html_chunks)} structured chunks from arXiv HTML",
                    flush=True,
                )

                n = upsert_chunks(html_chunks, model=model)
                duration = round(time.perf_counter() - t0, 3)
                print(f"  [embed] upserted {n} chunks in {duration:.2f}s", flush=True)

                final_title = parsed_title or title
                _record_ingest_profile({
                    "arxiv_id": aid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_format": "html",
                    "pages": 0,
                    "used_ocr": False,
                    "chunks": n,
                    "duration_s": duration,
                    "skipped_embed": False,
                })
                return {
                    "arxiv_id": aid,
                    "ok": True,
                    "title": final_title,
                    "source_format": "html",
                    "pdf": str(dest) if dest.exists() else None,
                    "markdown": str(md_dest),
                    "chunks": n,
                    "skipped_embed": False,
                    "duration_s": duration,
                }
            else:
                print(f"  [html] not available on arXiv; falling back to PDF download...", flush=True)

        # --- Route 2: PDF Fallback ---
        pdf_reused = (
            dest.exists() and dest.stat().st_size > 1000 and not force_download
        )
        pdf = download_paper(aid, force=force_download)
        if pdf_reused:
            print(f"  [skip-download] PDF already on disk: {pdf.name}", flush=True)
        else:
            print(f"  [download] wrote {pdf.name}", flush=True)

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
        title = resolve_title(aid, md_path)
        chunks = chunk_markdown(md_path, paper_id=aid, title=title)
        n = upsert_chunks(chunks, model=model)
        duration = round(time.perf_counter() - t0, 3)
        print(f"  [embed] upserted {n} chunks in {duration:.2f}s", flush=True)
        _record_ingest_profile({
            "arxiv_id": aid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_format": "pdf",
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
            "source_format": "pdf",
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
        return {"arxiv_id": aid, "ok": False, "error": str(e), "source_format": None, "duration_s": duration}


def ingest_papers(
    ids: Iterable[str],
    force_download: bool = False,
    force_embed: bool = False,
    prefer_html: bool = True,
    force_pdf: bool = False,
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
                prefer_html=prefer_html,
                force_pdf=force_pdf,
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
    prefer_html: bool = True,
    force_pdf: bool = False,
) -> list[dict]:
    ids = ids_from_personal_ranking(top_n=top_n, min_score=min_score)
    return ingest_papers(
        ids,
        force_download=force_download,
        force_embed=force_embed,
        prefer_html=prefer_html,
        force_pdf=force_pdf,
    )
