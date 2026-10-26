#!/usr/bin/env python3
"""C. Sample 20 answerable Qs where top-1 paper is not expected; show titles + snippet.
  uv run python results/confai_analysis/C_top1_mismatch_sample.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
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


def _norm(pid: str) -> str:
    s = (pid or "").strip().replace("arXiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    from rag.embed_store import load_embedding_model, query_chunks

    gold = json.loads((ROOT / "results/rag_eval/gold_questions.json").read_text())
    answerable = [q for q in gold if not q.get("should_refuse")]
    model = load_embedding_model()

    mismatches = []
    for q in answerable:
        hits = query_chunks(q["question"], top_k=args.top_k, paper_ids=None, model=model)
        if not hits:
            continue
        top = hits[0]
        exp = {_norm(x) for x in (q.get("expected_paper_ids") or [])}
        if _norm(top.get("paper_id") or "") in exp:
            continue
        # title for expected: best-effort from any hit or empty
        exp_titles = []
        for pid in q.get("expected_paper_ids") or []:
            exp_titles.append(pid)
        mismatches.append(
            {
                "id": q.get("id"),
                "kind": q.get("kind"),
                "question": q.get("question"),
                "expected_paper_ids": q.get("expected_paper_ids") or [],
                "top1_paper_id": top.get("paper_id"),
                "top1_title": top.get("title") or "",
                "top1_score": top.get("score"),
                "top1_section": top.get("section") or "",
                "top1_snippet": (top.get("text") or "")[:300],
            }
        )

    rng = random.Random(args.seed)
    sample = mismatches if len(mismatches) <= args.n else rng.sample(mismatches, args.n)

    # Enrich expected titles via one-chunk query filter if possible
    for item in sample:
        titles = []
        for pid in item["expected_paper_ids"]:
            try:
                h = query_chunks(
                    item["question"], top_k=1, paper_ids=[_norm(pid)], model=model
                )
                titles.append(
                    {
                        "paper_id": pid,
                        "title": (h[0].get("title") if h else "") or "(not in index or no title)",
                    }
                )
            except Exception as e:
                titles.append({"paper_id": pid, "title": f"(error: {e})"})
        item["expected_papers"] = titles

    path = OUT / "C_top1_mismatch_sample.json"
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    lines = [
        "# C. Top-1 ≠ expected (sample for human judgment)",
        "",
        f"Pool of mismatches: {len(mismatches)}; sample n={len(sample)}; seed={args.seed}",
        "",
    ]
    for i, item in enumerate(sample, 1):
        lines += [
            f"## {i}. {item['id']} ({item['kind']})",
            f"**Q:** {item['question']}",
            f"**Expected:** {item['expected_papers']}",
            f"**Top-1:** {item['top1_paper_id']} — {item['top1_title']} (score={item['top1_score']})",
            f"**Snippet:** {item['top1_snippet']}",
            "",
        ]
    (OUT / "C_top1_mismatch_sample.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[C] mismatches={len(mismatches)} wrote sample n={len(sample)} → {path}")


if __name__ == "__main__":
    main()
