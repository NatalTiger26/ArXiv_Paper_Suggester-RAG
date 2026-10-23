#!/usr/bin/env python3
"""
Safely UNION arXiv IDs into data/corpus.json — never deletes existing entries.

Sources (all optional, combined):
  • current data/corpus.json
  • one or more ID list files (terminal recovery, etc.)
  • data/papers/*.pdf stems
  • data/markdown/*.md stems
  • data/rag_ingest_progress/done.txt
  • RAG index (paper_ids)

Usage:
  uv run python scripts/merge_library_ids.py \\
    --from-file data/rag_ingest_progress/recovered_from_terminal_3054.txt \\
    --from-pdfs --from-markdown --from-done --from-rag

  # dry run
  uv run python scripts/merge_library_ids.py --from-file ... --from-pdfs --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data" / "corpus.json"
PAPERS = ROOT / "data" / "papers"
MARKDOWN = ROOT / "data" / "markdown"
DONE = ROOT / "data" / "rag_ingest_progress" / "done.txt"


def strip_ver(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def is_aid(s: str) -> bool:
    s = strip_ver(s)
    return bool(
        re.match(r"^\d{4}\.\d{4,5}$", s)
        or re.match(r"^[a-z][\w\-]*/\d{7}$", s, re.I)
    )


def load_ids_file(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        print(f"[merge] missing file: {path}")
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d{4}\.\d{4,5}|[a-z][\w\-]*/\d{7})\b", line, re.I)
        if m and is_aid(m.group(1)):
            out.add(strip_ver(m.group(1)))
    return out


def load_corpus() -> tuple[list[dict], dict[str, dict]]:
    if not CORPUS.exists():
        return [], {}
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
    by: dict[str, dict] = {}
    for p in papers:
        if not isinstance(p, dict):
            continue
        aid = strip_ver(str(p.get("arxiv_id") or p.get("id") or ""))
        if is_aid(aid):
            by[aid] = p
    return papers, by


def main() -> None:
    ap = argparse.ArgumentParser(description="Union IDs into corpus.json (never shrink)")
    ap.add_argument("--from-file", action="append", default=[], help="ID list file(s)")
    ap.add_argument("--from-pdfs", action="store_true")
    ap.add_argument("--from-markdown", action="store_true")
    ap.add_argument("--from-done", action="store_true")
    ap.add_argument("--from-rag", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Write corpus.json.bak_TIMESTAMP before overwrite (default on)",
    )
    args = ap.parse_args()

    _, by = load_corpus()
    before = len(by)
    print(f"[merge] corpus before: {before}")

    added_from: dict[str, int] = {}

    def add_many(ids: set[str], label: str) -> None:
        n = 0
        for aid in ids:
            if not is_aid(aid):
                continue
            if aid not in by:
                by[aid] = {
                    "arxiv_id": aid,
                    "title": "",
                    "source": f"merge:{label}",
                }
                n += 1
        added_from[label] = n
        print(f"[merge] +{n} new from {label} (source size {len(ids)})")

    for f in args.from_file:
        add_many(load_ids_file(Path(f)), Path(f).name)

    if args.from_pdfs and PAPERS.is_dir():
        ids = set()
        for p in PAPERS.glob("*.pdf"):
            stem = strip_ver(p.stem.replace("arXiv:", ""))
            if is_aid(stem):
                ids.add(stem)
        add_many(ids, "pdfs")

    if args.from_markdown and MARKDOWN.is_dir():
        ids = set()
        for p in MARKDOWN.glob("*.md"):
            stem = strip_ver(p.stem.replace("arXiv:", ""))
            if is_aid(stem):
                ids.add(stem)
        add_many(ids, "markdown")

    if args.from_done and DONE.exists():
        add_many(load_ids_file(DONE), "done.txt")

    if args.from_rag:
        try:
            from rag.embed_store import list_ingested_papers

            ids = set()
            for p in list_ingested_papers():
                aid = strip_ver(str(p.get("paper_id") or p.get("arxiv_id") or ""))
                if is_aid(aid):
                    ids.add(aid)
            add_many(ids, "rag")
        except Exception as e:
            print(f"[merge] RAG skip: {e}")

    after = len(by)
    print(f"[merge] corpus after:  {after}  (net +{after - before})")
    print(f"[merge] breakdown new: {added_from}")

    if args.dry_run:
        print("[merge] dry-run — corpus.json not written")
        return

    if after < before:
        print("[merge] REFUSING to write: would shrink corpus. Bug?")
        sys.exit(1)

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and CORPUS.exists():
        import time

        bak = CORPUS.with_name(f"corpus.json.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        bak.write_bytes(CORPUS.read_bytes())
        print(f"[merge] backup → {bak}")

    out = list(by.values())
    CORPUS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[merge] wrote {CORPUS} n={len(out)}")
    print()
    print("Next:")
    print("  uv run python scripts/sync_rag_progress.py")
    print("  caffeinate -dims bash scripts/reingest_library.sh")


if __name__ == "__main__":
    main()
