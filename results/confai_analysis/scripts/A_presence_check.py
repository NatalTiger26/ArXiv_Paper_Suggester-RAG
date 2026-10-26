#!/usr/bin/env python3
"""A. Gold expected_paper_id presence in full Chroma RAG index.
Run from project root. Does not modify src/.
  uv run python results/confai_analysis/A_presence_check.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def find_project_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists() or (p / "src").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]

ROOT = find_project_root()
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "results" / "confai_analysis"


def main() -> None:
    from rag.embed_store import get_collection, list_ingested_papers

    gold = json.loads((ROOT / "results/rag_eval/gold_questions.json").read_text())
    expected: set[str] = set()
    q_by_paper: dict[str, list[str]] = defaultdict(list)
    for q in gold:
        for pid in q.get("expected_paper_ids") or []:
            expected.add(str(pid).strip())
            q_by_paper[str(pid).strip()].append(q.get("id", "?"))

    # Prefer metadata scan via collection.get in batches
    coll = get_collection(create=False)
    counts: Counter[str] = Counter()
    titles: dict[str, str] = {}
    offset = 0
    batch = 5000
    total = coll.count()
    print(f"[A] collection count={total}", flush=True)
    while offset < total:
        try:
            res = coll.get(
                include=["metadatas"],
                limit=min(batch, total - offset),
                offset=offset,
            )
        except Exception as e:
            print(f"[A] get failed offset={offset}: {e}", flush=True)
            break
        metas = res.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            if not m:
                continue
            pid = str(m.get("paper_id") or "").strip()
            if not pid:
                continue
            counts[pid] += 1
            if pid not in titles and m.get("title"):
                titles[pid] = str(m.get("title"))[:120]
        offset += len(metas)
        if offset % 50000 < batch:
            print(f"[A] scanned {offset}/{total}", flush=True)

    # Also cross-check list_ingested
    try:
        listed = {str(p.get("paper_id") or p.get("arxiv_id")) for p in list_ingested_papers()}
    except Exception:
        listed = set()

    rows = []
    for pid in sorted(expected):
        n = int(counts.get(pid, 0))
        rows.append(
            {
                "paper_id": pid,
                "n_chunks": n,
                "in_list_ingested": pid in listed,
                "title": titles.get(pid, ""),
                "gold_question_ids": q_by_paper.get(pid, []),
                "n_gold_questions": len(q_by_paper.get(pid, [])),
            }
        )

    absent = [r for r in rows if r["n_chunks"] == 0]
    low = [r for r in rows if 0 < r["n_chunks"] < 5]
    qs_absent = sorted({qid for r in absent for qid in r["gold_question_ids"]})
    qs_low = sorted({qid for r in low for qid in r["gold_question_ids"]})

    out = {
        "index_total_chunks_scanned": int(sum(counts.values())),
        "collection_count": total,
        "n_expected_unique": len(expected),
        "papers": rows,
        "absent_papers": absent,
        "low_chunk_papers_lt5": low,
        "gold_question_ids_depending_on_absent": qs_absent,
        "gold_question_ids_depending_on_low_chunk": qs_low,
    }
    path = OUT / "A_presence.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = OUT / "A_presence.md"
    lines = [
        "# A. Gold paper presence in full RAG index",
        "",
        f"- Collection count: **{total}**",
        f"- Unique expected papers in gold: **{len(expected)}**",
        f"- Absent (0 chunks): **{len(absent)}**",
        f"- Low (&lt;5 chunks): **{len(low)}**",
        "",
        "## Per expected paper",
        "",
        "| paper_id | n_chunks | n_gold_Qs | title |",
        "|----------|---------:|----------:|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['paper_id']} | {r['n_chunks']} | {r['n_gold_questions']} | "
            f"{(r['title'] or '')[:50]} |"
        )
    lines += ["", "## Questions depending on absent papers", "", ", ".join(qs_absent) or "(none)"]
    lines += ["", "## Questions depending on low-chunk papers", "", ", ".join(qs_low) or "(none)"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[A] wrote {path}")
    print(f"[A] wrote {md}")
    print(f"[A] absent={len(absent)} low={len(low)} qs_absent={len(qs_absent)}")


if __name__ == "__main__":
    main()
