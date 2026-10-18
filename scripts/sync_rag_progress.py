#!/usr/bin/env python3
"""
Align reingest progress with the RAG index (source of truth).

Problem: done.txt can list papers that never made it into chroma_rag, so
reingest_library.sh skips work that still needs full-text ingest.

This script:
  1) Lists every paper actually in the RAG Chroma collection
  2) Intersects with data/corpus.json (the library)
  3) Rewrites data/rag_ingest_progress/done.txt so ONLY truly indexed
     library papers are "done"
  4) Prints remaining = library − RAG

Usage (project root):
  uv run python scripts/sync_rag_progress.py
  uv run python scripts/sync_rag_progress.py --dry-run
  uv run python scripts/sync_rag_progress.py --keep-extra-done   # do not shrink done.txt

After this:
  caffeinate -dims bash scripts/reingest_library.sh
only processes library papers not already in the RAG index (when done.txt
matches RAG, which is the default mode of this script).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROGRESS = ROOT / "data" / "rag_ingest_progress"
DONE = PROGRESS / "done.txt"
FAILED = PROGRESS / "failed.txt"
CORPUS_CANDIDATES = [
    ROOT / "data" / "corpus.json",
    ROOT / "data" / "corpus_100.json",
]


def strip_ver(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def load_library_ids() -> set[str]:
    path = next((p for p in CORPUS_CANDIDATES if p.exists()), None)
    if path is None:
        sys.exit("No data/corpus.json found")
    raw = json.loads(path.read_text(encoding="utf-8"))
    papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
    out: set[str] = set()
    for p in papers:
        aid = strip_ver(str(p.get("arxiv_id") or p.get("id") or ""))
        if re.match(r"^\d{4}\.\d{4,5}$", aid) or re.match(r"^[a-z\-]+/\d{7}$", aid, re.I):
            out.add(aid)
    print(f"[sync] corpus={path}  library_ids={len(out)}")
    return out


def load_rag_ids() -> set[str]:
    try:
        from rag.embed_store import list_ingested_papers, get_collection
    except ImportError:
        from src.rag.embed_store import list_ingested_papers, get_collection  # type: ignore

    try:
        coll = get_collection(create=False)
        chunk_count = int(coll.count() or 0)
    except Exception as e:
        print(f"[sync] RAG collection unavailable: {e}")
        chunk_count = 0

    papers = list_ingested_papers()
    out: set[str] = set()
    for p in papers:
        # list_ingested_papers uses paper_id; older rows may use arxiv_id
        aid = strip_ver(
            str(p.get("paper_id") or p.get("arxiv_id") or p.get("id") or "")
        )
        if aid:
            out.add(aid)
    print(f"[sync] RAG chunks={chunk_count}  unique_papers={len(out)}")
    if chunk_count > 0 and len(out) == 0:
        print(
            "[sync] WARNING: collection has chunks but no paper_id/arxiv_id in metadata — "
            "cannot mark done; check ingest metadata keys."
        )
    return out


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_lines(path: Path, ids: set[str] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({strip_ver(x) for x in ids if x})
    path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync done.txt to RAG index ∩ library")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not write done.txt",
    )
    ap.add_argument(
        "--keep-extra-done",
        action="store_true",
        help="Only ADD library∩RAG into done.txt; do not remove false done lines",
    )
    ap.add_argument(
        "--clear-failed-if-indexed",
        action="store_true",
        default=True,
        help="Drop failed.txt entries that are now in RAG (default on)",
    )
    ap.add_argument(
        "--no-clear-failed",
        action="store_true",
        help="Leave failed.txt unchanged",
    )
    args = ap.parse_args()

    library = load_library_ids()
    rag = load_rag_ids()

    in_lib_and_rag = library & rag
    remaining = library - rag

    old_done = {strip_ver(x) for x in load_lines(DONE)}
    false_done = (old_done & library) - rag  # marked done but not in RAG
    done_outside_lib = old_done - library

    print(f"[sync] library ∩ RAG (truly done): {len(in_lib_and_rag)}")
    print(f"[sync] library − RAG (still need ingest): {len(remaining)}")
    print(f"[sync] old done.txt lines: {len(old_done)}")
    print(f"[sync] false done (in done.txt ∩ library but NOT in RAG): {len(false_done)}")
    if false_done and len(false_done) <= 20:
        print(f"         examples: {sorted(false_done)[:20]}")
    elif false_done:
        print(f"         examples: {sorted(false_done)[:10]} …")
    print(f"[sync] done.txt entries outside current library: {len(done_outside_lib)}")

    if args.keep_extra_done:
        new_done = old_done | in_lib_and_rag
        mode = "merge (keep extra done lines)"
    else:
        # Authoritative: only RAG∩library count as done for the library.
        # Keep non-library done lines optional — drop them to avoid confusion.
        new_done = set(in_lib_and_rag)
        mode = "authoritative (done.txt := library ∩ RAG only)"

    print(f"[sync] mode={mode} → new done.txt size={len(new_done)}")

    if args.dry_run:
        print("[sync] dry-run — no files written")
        print(f"[sync] after sync, reingest REMAINING should be ≈ {len(remaining)}")
        return

    write_lines(DONE, new_done)
    print(f"[sync] wrote {DONE}")

    if not args.no_clear_failed and args.clear_failed_if_indexed:
        failed = {strip_ver(x) for x in load_lines(FAILED)}
        still_failed = failed - rag
        removed = failed & rag
        write_lines(FAILED, still_failed)
        print(f"[sync] failed.txt: kept={len(still_failed)} removed_indexed={len(removed)}")

    # Small machine-readable summary
    summary = {
        "library": len(library),
        "rag_indexed": len(rag),
        "done_true": len(in_lib_and_rag),
        "remaining": len(remaining),
        "false_done_cleared": len(false_done) if not args.keep_extra_done else 0,
        "done_file": str(DONE),
    }
    summary_path = PROGRESS / "sync_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sync] summary → {summary_path}")
    print()
    print("Next:")
    print("  caffeinate -dims bash scripts/reingest_library.sh")
    print(f"  Expected REMAINING ≈ {len(remaining)} (library papers not in RAG)")


if __name__ == "__main__":
    main()
