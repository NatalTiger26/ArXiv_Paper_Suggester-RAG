#!/usr/bin/env python3
"""G. Full-library generation eval WITHOUT modifying src/.
Saves FULL answer text per gold question to G_full_gen_rows.json.

  uv run python results/confai_analysis/G_full_gen_eval.py --no-cache

Uses the same official looks_like_refusal keys as src/final_eval.py (copied here
as a constant — not imported from a patched module). Also reports a broader
phrase grader for comparison only.
"""
from __future__ import annotations

import argparse
import json
import re
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

TOP_K = 6
CITATION_RE = re.compile(r"\[(\d+)\]")

OFFICIAL_REFUSAL_KEYS = [
    "not covered",
    "do not contain",
    "does not provide",
    "no relevant",
    "not enough information",
    "ingested papers do not",
    "cannot answer",
    "outside the scope",
    "not present in",
    "i don't know",
    "i do not know",
    "unable to answer",
]
BROADER_EXTRA = [
    "there is no information",
    "no mention",
    "insufficient context",
    "insufficient evidence",
    "insufficient information",
    "excerpts do not",
    "not in the ingested",
]


def looks_like_refusal_official(answer: str) -> bool:
    a = (answer or "").lower()
    return any(k in a for k in OFFICIAL_REFUSAL_KEYS)


def looks_like_refusal_broader(answer: str) -> bool:
    a = (answer or "").lower()
    return any(k in a for k in OFFICIAL_REFUSAL_KEYS + BROADER_EXTRA)


def citation_validity(answer: str, n_sources: int) -> dict:
    nums = [int(x) for x in CITATION_RE.findall(answer or "")]
    if not nums:
        return {
            "has_citation": False,
            "valid": n_sources == 0 or looks_like_refusal_official(answer),
            "n_citations": 0,
        }
    invalid = [n for n in nums if n < 1 or n > n_sources]
    return {
        "has_citation": True,
        "valid": len(invalid) == 0,
        "n_citations": len(nums),
        "invalid_indices": sorted(set(invalid)),
    }


def _norm(pid: str) -> str:
    s = (pid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def retrieval_metrics(expected, retrieved):
    exp = {_norm(x) for x in expected if x}
    ret = [_norm(x) for x in retrieved]
    if not exp:
        return {"recall@k": None, "mrr": None, "hit_rate@1": None}
    hit = any(r in exp for r in ret[:TOP_K])
    mrr = 0.0
    for i, r in enumerate(ret[:TOP_K], 1):
        if r in exp:
            mrr = 1.0 / i
            break
    return {
        "recall@k": 1.0 if hit else 0.0,
        "mrr": mrr,
        "hit_rate@1": 1.0 if ret and ret[0] in exp else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from rag.embed_store import get_collection, list_ingested_papers
    from rag.rag_pipeline import ask

    gold = json.loads((ROOT / "results/rag_eval/gold_questions.json").read_text())
    coll = get_collection(create=False)
    n_chunks = coll.count()
    n_papers = len(list_ingested_papers())
    print(f"[G] index papers≈{n_papers} chunks={n_chunks}", flush=True)

    rows = []
    for i, q in enumerate(gold, 1):
        print(f"[G] {i}/{len(gold)} {q.get('id')} …", flush=True)
        t0 = time.time()
        err = None
        try:
            result = ask(q["question"], use_cache=not args.no_cache, return_chunks=True)
        except Exception as e:
            result = {"answer": "", "references": [], "coverage": None, "chunks": []}
            err = str(e)
        answer = (result.get("answer") or "") if isinstance(result, dict) else str(result)
        chunks = result.get("chunks") or []
        refs = result.get("references") or []
        n_sources = max(len(refs), len(chunks))
        retrieved = []
        for c in chunks:
            pid = c.get("paper_id") or ""
            if pid and pid not in retrieved:
                retrieved.append(pid)
        for r in refs:
            pid = r.get("paper_id") or r.get("arxiv_id") or ""
            if pid and pid not in retrieved:
                retrieved.append(pid)

        should = bool(q.get("should_refuse"))
        did_off = looks_like_refusal_official(answer)
        did_br = looks_like_refusal_broader(answer)
        cites = citation_validity(answer, n_sources)
        rm = retrieval_metrics(q.get("expected_paper_ids") or [], retrieved)

        rows.append(
            {
                "id": q.get("id"),
                "kind": q.get("kind"),
                "difficulty": q.get("difficulty"),
                "question": q.get("question"),
                "expected_paper_ids": q.get("expected_paper_ids") or [],
                "should_refuse": should,
                "answer_full": answer,
                "coverage": result.get("coverage"),
                "retrieved_paper_ids": retrieved,
                "n_sources": n_sources,
                "did_refuse_official": did_off,
                "did_refuse_broader": did_br,
                "citation_has": cites["has_citation"],
                "citation_valid": cites["valid"],
                "n_citations": cites["n_citations"],
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "cache_hit": bool(result.get("cache_hit")),
                "error": err,
                **rm,
            }
        )
        (OUT / "G_full_gen_rows_partial.json").write_text(
            json.dumps({"index_papers": n_papers, "index_chunks": n_chunks, "rows": rows}, indent=2),
            encoding="utf-8",
        )

    # Aggregate
    answerable = [r for r in rows if not r["should_refuse"]]
    oos = [r for r in rows if r["should_refuse"]]
    lats = sorted(r["latency_ms"] for r in rows)

    def pct(p):
        if not lats:
            return None
        return lats[int(round((p / 100) * (len(lats) - 1)))]

    def confusion(key):
        TP = sum(1 for r in oos if r[key])
        FN = sum(1 for r in oos if not r[key])
        FP = sum(1 for r in answerable if r[key])
        TN = sum(1 for r in answerable if not r[key])
        prec = TP / (TP + FP) if (TP + FP) else None
        rec = TP / (TP + FN) if (TP + FN) else None
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "precision": prec, "recall": rec}

    cited_valid = sum(1 for r in rows if r["citation_has"] and r["citation_valid"])
    cited_invalid = sum(1 for r in rows if r["citation_has"] and not r["citation_valid"])
    uncited_ref = sum(1 for r in rows if (not r["citation_has"]) and r["did_refuse_official"])
    uncited_non = sum(1 for r in rows if (not r["citation_has"]) and (not r["did_refuse_official"]))

    summary = {
        "index_papers": n_papers,
        "index_chunks": n_chunks,
        "n_questions": len(rows),
        "citation": {
            "cited_valid": cited_valid,
            "cited_invalid": cited_invalid,
            "uncited_refusal_official": uncited_ref,
            "uncited_non_refusal_official": uncited_non,
        },
        "refusal_official": confusion("did_refuse_official"),
        "refusal_broader": confusion("did_refuse_broader"),
        "answerable_wrongly_refused_official": sum(
            1 for r in answerable if r["did_refuse_official"]
        ),
        "answerable_wrongly_refused_broader": sum(
            1 for r in answerable if r["did_refuse_broader"]
        ),
        "latency_p50_ms": pct(50),
        "latency_p95_ms": pct(95),
        "mean_mrr_answerable": sum(r["mrr"] or 0 for r in answerable) / max(1, len(answerable)),
        "mean_recall_answerable": sum(r["recall@k"] or 0 for r in answerable)
        / max(1, len(answerable)),
    }
    out = {"summary": summary, "rows": rows}
    path = OUT / "G_full_gen_rows.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "G_full_gen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[G] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
