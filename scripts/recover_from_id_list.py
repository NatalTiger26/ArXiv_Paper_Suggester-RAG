#!/usr/bin/env python3
"""
Recover progress from a saved list of ingested arXiv IDs (e.g. terminal status dump).

Does NOT re-download or re-embed by default. It:
  1) Loads IDs from a text file (one id per line)
  2) Optionally merges them into data/corpus.json if missing
  3) Writes data/rag_ingest_progress/done.txt from (list ∩ current RAG)
     OR trusts the list as done if --trust-list
  4) Prints how many still need full-text ingest

Usage:
  # After RAG index is healthy again:
  uv run python scripts/recover_from_id_list.py data/rag_ingest_progress/recovered_from_terminal_3054.txt

  # Trust the list as done even if RAG was wiped (queue only the rest of library):
  uv run python scripts/recover_from_id_list.py data/rag_ingest_progress/recovered_from_terminal_3054.txt --trust-list

  # Also ensure every id is in corpus.json:
  uv run python scripts/recover_from_id_list.py ... --merge-corpus
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
CORPUS = ROOT / "data" / "corpus.json"


def strip_ver(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def load_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow "id  title" lines
        m = re.match(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})\b", line, re.I)
        if m:
            ids.append(strip_ver(m.group(1)))
    return sorted(set(ids))


def load_corpus_ids() -> set[str]:
    if not CORPUS.exists():
        return set()
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
    out = set()
    for p in papers:
        aid = strip_ver(str(p.get("arxiv_id") or p.get("id") or ""))
        if aid:
            out.add(aid)
    return out


def load_rag_ids() -> set[str]:
    try:
        from rag.embed_store import list_ingested_papers, get_collection
        coll = get_collection(create=False)
        print(f"[recover] RAG chunks={coll.count()}")
        out = set()
        for p in list_ingested_papers():
            aid = strip_ver(str(p.get("paper_id") or p.get("arxiv_id") or ""))
            if aid:
                out.add(aid)
        return out
    except Exception as e:
        print(f"[recover] RAG list failed: {e}")
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id_file", type=Path, help="Text file of arXiv ids (one per line)")
    ap.add_argument("--trust-list", action="store_true",
                    help="Mark all listed ids as done even if not in RAG now")
    ap.add_argument("--merge-corpus", action="store_true",
                    help="Add missing ids into data/corpus.json as stub entries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    saved = load_ids(args.id_file)
    print(f"[recover] loaded {len(saved)} ids from {args.id_file}")

    library = load_corpus_ids()
    print(f"[recover] corpus.json library={len(library)}")
    rag = load_rag_ids()
    print(f"[recover] RAG unique papers={len(rag)}")

    in_lib = [a for a in saved if a in library]
    not_in_lib = [a for a in saved if a not in library]
    in_rag = [a for a in saved if a in rag]
    print(f"[recover] saved ∩ library={len(in_lib)}  saved not in library={len(not_in_lib)}")
    print(f"[recover] saved ∩ RAG={len(in_rag)}")

    if args.merge_corpus and not_in_lib:
        raw = json.loads(CORPUS.read_text(encoding="utf-8")) if CORPUS.exists() else []
        papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
        by = {}
        for p in papers:
            aid = strip_ver(str(p.get("arxiv_id") or p.get("id") or ""))
            if aid:
                by[aid] = p
        for aid in not_in_lib:
            by[aid] = {"arxiv_id": aid, "title": "", "source": "recovered_terminal"}
        out = list(by.values())
        if not args.dry_run:
            CORPUS.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"[recover] wrote corpus.json n={len(out)}")
        else:
            print(f"[recover] dry-run would write corpus n={len(out)}")
        library = set(by.keys())

    if args.trust_list:
        new_done = set(saved)
        mode = "trust-list (all saved ids → done.txt)"
    else:
        # only mark truly present in RAG
        new_done = set(saved) & rag
        mode = "RAG ∩ saved list → done.txt"
        if not rag:
            print("[recover] WARNING: RAG empty — use --trust-list if you only want to "
                  "shrink the reingest queue without claiming embed is done.")

    print(f"[recover] mode={mode} done_size={len(new_done)}")

    if args.dry_run:
        still = len(library - new_done) if library else "n/a"
        print(f"[recover] dry-run remaining library−done ≈ {still}")
        return

    PROGRESS.mkdir(parents=True, exist_ok=True)
    DONE.write_text("\n".join(sorted(new_done)) + ("\n" if new_done else ""), encoding="utf-8")
    print(f"[recover] wrote {DONE}")

    # backup the list
    bak = PROGRESS / "recovered_ingested_ids_backup.txt"
    bak.write_text("\n".join(sorted(saved)) + "\n", encoding="utf-8")
    print(f"[recover] backup list → {bak}")

    if library:
        remaining = sorted(library - new_done)
        print(f"[recover] library remaining to ingest: {len(remaining)}")
    print("Next: caffeinate -dims bash scripts/reingest_library.sh")
    print("  (FORCE=0 — existing PDF/MD reused; RAG skip if present)")


if __name__ == "__main__":
    main()
