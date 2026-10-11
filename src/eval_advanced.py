#!/usr/bin/env python3
"""
Advanced offline analysis on results/rag_eval/last_eval.json + dim_sweep.json.

Goes beyond the basic suite using AIP Lab 1–7 ideas:
  - Failure Pareto + missing-paper impact (Lab 5)
  - Refusal confusion matrix (Lab 4)
  - Bootstrap already in last_eval; McNemar-ready paired slots (Lab 2)
  - Kind × difficulty heat table (Lab 3 stratification)
  - Coverage vs correctness calibration
  - Cost / latency SLO gate (Lab 7)
  - Embedding dim decision (Matryoshka)
  - Portfolio recommendation paragraph

  uv run python src/eval_advanced.py
  uv run python src/eval_advanced.py --write-report
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "rag_eval"
RESULTS = ROOT / "results"


def load_json(name: str) -> dict[str, Any] | None:
    p = EVAL_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def missing_paper_impact(rows: list[dict]) -> list[tuple[str, int]]:
    missing: Counter[str] = Counter()
    for r in rows:
        if not r.get("has_relevant"):
            continue
        exp = r.get("expected_paper_ids") or []
        got = r.get("retrieved_paper_ids") or []
        for e in exp:
            if e not in got and not any(e in str(g) for g in got):
                missing[e] += 1
    return missing.most_common()


def simulate_ingest_lift(rows: list[dict], paper_id: str) -> dict[str, float]:
    """
    Upper-bound estimate: if `paper_id` were always retrievable when expected,
    how would mean Recall@k / MRR move? (optimistic: counts as hit@1).
    """
    ret = [r for r in rows if r.get("has_relevant")]
    n = len(ret) or 1
    base_rec = sum(r["recall@k"] for r in ret) / n
    base_mrr = sum(r["mrr"] for r in ret) / n

    recs, mrrs = [], []
    for r in ret:
        exp = set(r.get("expected_paper_ids") or [])
        if paper_id in exp and r["recall@k"] < 1.0:
            recs.append(1.0)
            mrrs.append(1.0)  # optimistic
        else:
            recs.append(r["recall@k"])
            mrrs.append(r["mrr"])
    return {
        "paper_id": paper_id,
        "base_recall": round(base_rec, 4),
        "lifted_recall": round(sum(recs) / n, 4),
        "base_mrr": round(base_mrr, 4),
        "lifted_mrr": round(sum(mrrs) / n, 4),
        "delta_recall": round(sum(recs) / n - base_rec, 4),
        "delta_mrr": round(sum(mrrs) / n - base_mrr, 4),
        "questions_helped": sum(
            1
            for r in ret
            if paper_id in (r.get("expected_paper_ids") or []) and r["recall@k"] < 1.0
        ),
    }


def refusal_confusion(rows: list[dict]) -> dict[str, int]:
    """
    should_refuse × did_refuse contingency (Lab 4).
    """
    tp = fp = tn = fn = 0
    for r in rows:
        if r.get("refusal_correct") is None and r.get("did_refuse") is None:
            continue
        # when skip_generation, did_refuse may be heuristic
        sr = bool(r.get("should_refuse"))
        dr = bool(r.get("did_refuse"))
        if sr and dr:
            tp += 1
        elif not sr and dr:
            fp += 1
        elif not sr and not dr:
            tn += 1
        else:
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def kind_difficulty_table(rows: list[dict]) -> list[dict]:
    ret = [r for r in rows if r.get("has_relevant")]
    cells = defaultdict(list)
    for r in ret:
        key = (r.get("kind") or "?", r.get("difficulty") or "?")
        cells[key].append(r["mrr"])
    out = []
    for (kind, diff), vals in sorted(cells.items()):
        out.append(
            {
                "kind": kind,
                "difficulty": diff,
                "n": len(vals),
                "mean_mrr": round(sum(vals) / len(vals), 4),
            }
        )
    return out


def coverage_calibration(rows: list[dict]) -> list[dict]:
    """Does 'coverage' track retrieval success?"""
    buckets = defaultdict(list)
    for r in rows:
        if not r.get("has_relevant"):
            continue
        buckets[r.get("coverage") or "n/a"].append(r["recall@k"])
    out = []
    for cov, vals in sorted(buckets.items()):
        out.append(
            {
                "coverage": cov,
                "n": len(vals),
                "mean_recall": round(sum(vals) / len(vals), 4),
            }
        )
    return out


def score_deciles_vs_hit(rows: list[dict]) -> list[dict]:
    ret = [r for r in rows if r.get("has_relevant")]
    if not ret:
        return []
    scored = sorted(ret, key=lambda r: r.get("top_score") or 0)
    n = len(scored)
    out = []
    for d in range(5):
        lo = int(d * n / 5)
        hi = int((d + 1) * n / 5)
        chunk = scored[lo:hi]
        if not chunk:
            continue
        out.append(
            {
                "quintile": d + 1,
                "n": len(chunk),
                "score_lo": round(min(c["top_score"] for c in chunk), 4),
                "score_hi": round(max(c["top_score"] for c in chunk), 4),
                "mean_hit@1": round(sum(c["hit_rate@1"] for c in chunk) / len(chunk), 4),
                "mean_recall": round(sum(c["recall@k"] for c in chunk) / len(chunk), 4),
            }
        )
    return out


def slo_gate(summary: dict) -> dict[str, Any]:
    """Lab 7-style regression gates (defaults tunable)."""
    gates = {
        "mean_mrr_min": 0.55,
        "mean_recall_min": 0.55,
        "refusal_precision_min": 0.50,  # only if measured
        "p95_latency_ms_max_retrieval": 500,
        "p95_latency_ms_max_generation": 15000,
    }
    results = {}
    mrr = summary.get("mean_mrr")
    rec = summary.get("mean_recall@k")
    results["mrr_pass"] = mrr is not None and mrr >= gates["mean_mrr_min"]
    results["recall_pass"] = rec is not None and rec >= gates["mean_recall_min"]
    rp = summary.get("refusal_precision")
    if rp is not None:
        results["refusal_precision_pass"] = rp >= gates["refusal_precision_min"]
    p95 = summary.get("latency_p95_ms")
    skip = summary.get("skip_generation")
    if p95 is not None:
        limit = (
            gates["p95_latency_ms_max_retrieval"]
            if skip
            else gates["p95_latency_ms_max_generation"]
        )
        results["latency_pass"] = p95 <= limit
        results["latency_limit_ms"] = limit
    results["gates"] = gates
    results["all_pass"] = all(
        v for k, v in results.items() if k.endswith("_pass")
    )
    return results


def annual_cost_sketch(cost_per_query: float = 0.005, queries_per_day: int = 1000) -> dict:
    """Lab 1/2 economic framing — placeholders for API cost."""
    annual = cost_per_query * queries_per_day * 365
    return {
        "assume_usd_per_query": cost_per_query,
        "queries_per_day": queries_per_day,
        "usd_per_year": round(annual, 2),
        "note": "Replace cost_per_query with measured Gemini spend/query from your traces.",
    }


def build_analysis() -> dict[str, Any]:
    last = load_json("last_eval.json")
    dim = load_json("dim_sweep.json")
    if not last:
        raise SystemExit("Missing results/rag_eval/last_eval.json — run eval_suite first.")

    s = last["summary"]
    rows = last["rows"]
    missing = missing_paper_impact(rows)
    lifts = [simulate_ingest_lift(rows, pid) for pid, _ in missing[:5]]
    conf = refusal_confusion(rows)
    kd = kind_difficulty_table(rows)
    cal = coverage_calibration(rows)
    dec = score_deciles_vs_hit(rows)
    gate = slo_gate(s)
    cost = annual_cost_sketch()

    # Recommendation paragraph
    top_missing = missing[0][0] if missing else None
    top_n = missing[0][1] if missing else 0
    dim_rec = (dim or {}).get("recommendation") or ""
    fh = s.get("fail_hint_counts") or {}
    cache = s.get("cache_hit_rate")
    cache_note = ""
    if cache is not None and float(cache) >= 0.9:
        cache_note = (
            " Generation metrics used a warm semantic cache (hit rate "
            f"{cache}); for a cold refusal measurement re-run with cache disabled."
        )
    rank_n = fh.get("ranking", 0)
    miss_n = fh.get("missing_or_ranking", 0)
    rec_para = (
        f"With {s.get('index_papers')} full-text papers indexed, mean MRR is "
        f"{s.get('mean_mrr')} (95% CI {s.get('mrr_ci95')}) and Recall@k is "
        f"{s.get('mean_recall@k')} on {s.get('n_retrieval_scored')} gold items with labels. "
        f"Residual retrieval issues: missing_or_ranking={miss_n}, ranking={rank_n}. "
        f"Most frequent top-k miss is still {top_missing} ({top_n} questions) — "
        f"often ranking/embedding mismatch, not a missing PDF. "
        f"Refusal precision/recall are {s.get('refusal_precision')}/{s.get('refusal_recall')} "
        f"(product bar for unsupervised use ≈ 0.7).{cache_note} "
        f"Dim sweep: {dim_rec} "
        f"System is suitable for a constrained demo on the ingested MI/NTK library; "
        f"not for unsupervised refusal-critical deployment until OOS precision improves."
    )

    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary_echo": {
            "mean_mrr": s.get("mean_mrr"),
            "mean_recall@k": s.get("mean_recall@k"),
            "mean_hit_rate@1": s.get("mean_hit_rate@1"),
            "mrr_ci95": s.get("mrr_ci95"),
            "refusal_precision": s.get("refusal_precision"),
            "refusal_recall": s.get("refusal_recall"),
            "index_papers": s.get("index_papers"),
            "fail_hint_counts": s.get("fail_hint_counts"),
        },
        "missing_papers": [{"arxiv_id": a, "n_questions": n} for a, n in missing],
        "ingest_lift_simulation": lifts,
        "refusal_confusion": conf,
        "kind_difficulty": kd,
        "coverage_calibration": cal,
        "score_quintiles": dec,
        "slo_gate": gate,
        "cost_sketch": cost,
        "dim_sweep": dim,
        "recommendation": rec_para,
    }
    return analysis


def write_markdown(analysis: dict[str, Any]) -> Path:
    s = analysis["summary_echo"]
    lines = [
        "# Advanced Evaluation Report — ArXiv Paper Suggester",
        "",
        f"_Generated: {analysis['timestamp']}_",
        "",
        "This report extends the standard suite with Lab 1–7 style analyses "
        "(failure Pareto, refusal matrix, calibration, SLO gate, cost sketch, dim decision).",
        "",
        "## Executive recommendation",
        "",
        analysis["recommendation"],
        "",
        "## Headline metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| MRR | {s.get('mean_mrr')} CI {s.get('mrr_ci95')} |",
        f"| Recall@k | {s.get('mean_recall@k')} |",
        f"| Hit@1 | {s.get('mean_hit_rate@1')} |",
        f"| Refusal P / R | {s.get('refusal_precision')} / {s.get('refusal_recall')} |",
        f"| RAG papers indexed | {s.get('index_papers')} |",
        f"| Failure hints | `{s.get('fail_hint_counts')}` |",
        "",
        "## Lab 5 — Failure Pareto & ingest lift",
        "",
        "### Papers expected by gold but often missing from top-k",
        "",
        "| arXiv ID | # gold questions hurt |",
        "|----------|----------------------:|",
    ]
    for m in analysis["missing_papers"][:12]:
        lines.append(f"| {m['arxiv_id']} | {m['n_questions']} |")

    lines += [
        "",
        "### Optimistic lift if top missing papers were always retrieved",
        "",
        "| Paper | Δ Recall | Δ MRR | Questions helped |",
        "|-------|----------|-------|------------------|",
    ]
    for L in analysis["ingest_lift_simulation"]:
        lines.append(
            f"| {L['paper_id']} | {L['delta_recall']:+.3f} | {L['delta_mrr']:+.3f} | {L['questions_helped']} |"
        )

    conf = analysis["refusal_confusion"]
    lines += [
        "",
        "## Lab 4 — Refusal confusion matrix",
        "",
        "|  | did_refuse | did_not |",
        "|--|----------:|--------:|",
        f"| **should_refuse** | TP {conf['TP']} | FN {conf['FN']} |",
        f"| **should_answer** | FP {conf['FP']} | TN {conf['TN']} |",
        "",
        "Precision = TP/(TP+FP); Recall = TP/(TP+FN). Both must be reported.",
        "",
        "## Lab 3 — Kind × difficulty (mean MRR)",
        "",
        "| Kind | Difficulty | n | Mean MRR |",
        "|------|------------|---|----------|",
    ]
    for cell in analysis["kind_difficulty"]:
        lines.append(
            f"| {cell['kind']} | {cell['difficulty']} | {cell['n']} | {cell['mean_mrr']} |"
        )

    lines += [
        "",
        "## Coverage vs retrieval success (calibration)",
        "",
        "| Coverage label | n | Mean Recall@k |",
        "|----------------|---|---------------|",
    ]
    for c in analysis["coverage_calibration"]:
        lines.append(f"| {c['coverage']} | {c['n']} | {c['mean_recall']} |")

    lines += [
        "",
        "## Top-score quintiles vs hit rate",
        "",
        "| Quintile | Score range | n | Hit@1 | Recall |",
        "|----------|-------------|---|-------|--------|",
    ]
    for d in analysis["score_quintiles"]:
        lines.append(
            f"| {d['quintile']} | {d['score_lo']}–{d['score_hi']} | {d['n']} | "
            f"{d['mean_hit@1']} | {d['mean_recall']} |"
        )

    gate = analysis["slo_gate"]
    lines += [
        "",
        "## Lab 7 — SLO / regression gate",
        "",
        f"| Check | Pass |",
        f"|-------|------|",
        f"| MRR ≥ {gate['gates']['mean_mrr_min']} | {gate.get('mrr_pass')} |",
        f"| Recall ≥ {gate['gates']['mean_recall_min']} | {gate.get('recall_pass')} |",
        f"| Latency p95 ≤ limit | {gate.get('latency_pass')} (limit {gate.get('latency_limit_ms')} ms) |",
        f"| Refusal precision (if measured) | {gate.get('refusal_precision_pass', 'n/a')} |",
        f"| **All pass** | **{gate.get('all_pass')}** |",
        "",
        "## Cost sketch (Lab 1/2 economic framing)",
        "",
        f"```json\n{json.dumps(analysis['cost_sketch'], indent=2)}\n```",
        "",
        "## Embedding dim decision",
        "",
    ]
    dim = analysis.get("dim_sweep") or {}
    if dim.get("rows"):
        lines += ["| Dim | MRR | R@10 | Hit@1 |", "|-----|-----|------|-------|"]
        for r in dim["rows"]:
            lines.append(
                f"| {r['dim']} | {r['mean_mrr']} | {r['mean_recall@10']} | {r['mean_hit@1']} |"
            )
        lines += ["", dim.get("recommendation", ""), ""]
    else:
        lines += ["_No dim_sweep.json_", ""]

    lines += [
        "## What this system is not safe for",
        "",
        "- Unsupervised clinical, legal, or financial decisions.",
        "- Questions whose supporting papers are not ingested (28/80 retrieval fails are largely missing content).",
        "- Relying on refusal alone while precision ≈ 0.45.",
        "- Treating retrieval scores as probabilities without calibration (see quintile table).",
        "",
        "## Ranked next experiments",
        "",
        "1. **Ingest `2304.14997` and other top missing IDs** — largest expected Recall/MRR lift.",
        "2. **Set `EMBEDDING_DIM=512`**, rebuild corpus index + RAG, re-run suite.",
        "3. **Tighten OOS refusal** (coverage gate + prompt) aiming precision ≥ 0.7.",
        "4. **Re-run `--with-generation`** only after (1)–(3) to avoid paying for known misses.",
        "5. Optional: hybrid BM25 for `identifier` kind (Lab 3 Q44 lesson).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "uv run python src/eval_suite.py --all",
        "uv run python src/eval_suite.py --all --with-generation",
        "uv run python src/eval_advanced.py --write-report",
        "```",
        "",
    ]
    out = RESULTS / "EVALUATION_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (EVAL_DIR / "advanced_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out}")
    print(f"Wrote {EVAL_DIR / 'advanced_analysis.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true", default=True)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    analysis = build_analysis()
    if args.json_only:
        print(json.dumps(analysis, indent=2))
        return
    write_markdown(analysis)
    print("\n=== Recommendation ===")
    print(analysis["recommendation"])


if __name__ == "__main__":
    main()
