#!/usr/bin/env python3
"""D. Regenerate 20 out-of-scope gold questions; save FULL answer text.
  uv run python results/confai_analysis/D_oos_full_answers.py --no-cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from rag.rag_pipeline import ask

    gold = json.loads((ROOT / "results/rag_eval/gold_questions.json").read_text())
    oos = [q for q in gold if q.get("should_refuse")]
    print(f"[D] OOS questions={len(oos)}", flush=True)

    rows = []
    for i, q in enumerate(oos, 1):
        print(f"[D] {i}/{len(oos)} {q.get('id')} …", flush=True)
        t0 = time.time()
        try:
            result = ask(
                q["question"],
                use_cache=not args.no_cache,
            )
            # result is typically a dict with answer, references, coverage
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("text") or ""
                coverage = result.get("coverage")
                refs = result.get("references") or result.get("sources") or []
            else:
                answer = str(result)
                coverage = None
                refs = []
            err = None
        except Exception as e:
            answer = ""
            coverage = None
            refs = []
            err = str(e)
        rows.append(
            {
                "id": q.get("id"),
                "kind": q.get("kind"),
                "question": q.get("question"),
                "should_refuse": True,
                "answer_full": answer,
                "coverage": coverage,
                "n_references": len(refs) if isinstance(refs, list) else None,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "error": err,
            }
        )
        (OUT / "D_oos_full_answers_partial.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    path = OUT / "D_oos_full_answers.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[D] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
