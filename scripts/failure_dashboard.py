#!/usr/bin/env python3
"""
Failure Taxonomy Dashboard (ConfAI / Lab 5 Diagnostic).
Analyzes an evaluation run JSON and stratifies failures into 4 actionable buckets:
  1. missing_or_ranking: Expected paper not in top-k (disambiguated into uningested vs distractor suppression).
  2. ranking: Expected paper retrieved in top-k, but displaced from top-1.
  3. false_answer: Out-of-scope query was answered instead of refused (under-refusal).
  4. over_refuse: Answerable query was erroneously declined (over-refusal).

Usage:
  python scripts/failure_dashboard.py
  python scripts/failure_dashboard.py --eval-json results/rag_eval/final_eval.json --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "results" / "rag_eval" / "final_eval.json"
PRESENCE_PATH = ROOT / "results" / "confai_analysis" / "A_presence.json"


def load_presence_map() -> set[str]:
    """Returns set of normalized arXiv IDs present in the RAG index."""
    if PRESENCE_PATH.exists():
        try:
            data = json.loads(PRESENCE_PATH.read_text(encoding="utf-8"))
            return {
                str(row.get("paper_id", "")).replace("arXiv:", "").strip()
                for row in data
                if row.get("chunks", 0) > 0
            }
        except Exception:
            pass
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure Taxonomy Dashboard")
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=DEFAULT_EVAL,
        help="Path to evaluation JSON (default: results/rag_eval/final_eval.json)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "json"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    if not args.eval_json.exists():
        print(f"Error: Evaluation file not found at {args.eval_json}", file=sys.stderr)
        sys.exit(1)

    eval_data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    summary = eval_data.get("summary", {})
    rows = eval_data.get("rows", [])
    if not rows and isinstance(eval_data, list):
        rows = eval_data

    present_ids = load_presence_map()

    # Buckets
    buckets: dict[str, list[dict]] = {
        "missing_or_ranking": [],
        "ranking": [],
        "false_answer": [],
        "over_refuse": [],
    }

    for r in rows:
        hint = r.get("fail_hint")
        if hint in buckets:
            buckets[hint].append(r)

    total_eval = len(rows)
    total_failures = sum(len(b) for b in buckets.values())

    # Disambiguate missing_or_ranking into uningested vs ranked-down
    missing_uningested = 0
    missing_distractor = 0
    for r in buckets["missing_or_ranking"]:
        exp = [str(x).replace("arXiv:", "").strip() for x in r.get("expected_paper_ids") or []]
        if present_ids and any(x not in present_ids for x in exp):
            missing_uningested += 1
        else:
            missing_distractor += 1

    taxonomy_summary = {
        "total_questions": total_eval,
        "total_failures": total_failures,
        "success_rate": round(1.0 - (total_failures / max(1, total_eval)), 4),
        "buckets": {
            "missing_or_ranking": {
                "count": len(buckets["missing_or_ranking"]),
                "pct_of_all": round(len(buckets["missing_or_ranking"]) / max(1, total_eval) * 100, 1),
                "breakdown": {
                    "uningested_pdf": missing_uningested,
                    "distractor_suppression": missing_distractor,
                },
                "primary_action": "Enable hybrid BM25 / query rewrite or ingest missing PDFs",
            },
            "ranking": {
                "count": len(buckets["ranking"]),
                "pct_of_all": round(len(buckets["ranking"]) / max(1, total_eval) * 100, 1),
                "primary_action": "Tune dense top_k or apply cross-encoder reranker",
            },
            "false_answer": {
                "count": len(buckets["false_answer"]),
                "pct_of_all": round(len(buckets["false_answer"]) / max(1, total_eval) * 100, 1),
                "primary_action": "Tighten coverage threshold in prompt / increase refusal examples",
            },
            "over_refuse": {
                "count": len(buckets["over_refuse"]),
                "pct_of_all": round(len(buckets["over_refuse"]) / max(1, total_eval) * 100, 1),
                "primary_action": "Relax strict refusal prompt instructions or calibrate coverage cue",
            },
        },
    }

    if args.format == "json":
        print(json.dumps(taxonomy_summary, indent=2))
        return

    # Render Table or Markdown
    is_md = args.format == "markdown"
    bold = "**" if is_md else ""
    code = "`" if is_md else ""

    output = []
    output.append(f"{bold}=== Failure Taxonomy Dashboard ==={bold}")
    output.append(f"Source: {args.eval_json} (Total Questions: {total_eval})\n")

    if is_md:
        output.append("| Failure Bucket | Count | % Total | Diagnostic Root Cause | Actionable Mitigation |")
        output.append("| :--- | :---: | :---: | :--- | :--- |")
        b = taxonomy_summary["buckets"]
        output.append(
            f"| {code}missing_or_ranking{code} | {b['missing_or_ranking']['count']} | {b['missing_or_ranking']['pct_of_all']}% | Expected paper absent from top-k (distractors: {missing_distractor}, uningested: {missing_uningested}) | {b['missing_or_ranking']['primary_action']} |"
        )
        output.append(
            f"| {code}ranking{code} | {b['ranking']['count']} | {b['ranking']['pct_of_all']}% | Paper retrieved at rank 2–k; missed top-1 | {b['ranking']['primary_action']} |"
        )
        output.append(
            f"| {code}false_answer{code} | {b['false_answer']['count']} | {b['false_answer']['pct_of_all']}% | Out-of-scope query hallucinated or answered | {b['false_answer']['primary_action']} |"
        )
        output.append(
            f"| {code}over_refuse{code} | {b['over_refuse']['count']} | {b['over_refuse']['pct_of_all']}% | In-scope question rejected due to overly strict cue | {b['over_refuse']['primary_action']} |"
        )
    else:
        fmt = "{:<22} | {:<6} | {:<7} | {:<32} | {:<40}"
        output.append(fmt.format("Failure Bucket", "Count", "% Total", "Diagnostic Root Cause", "Actionable Mitigation"))
        output.append("-" * 118)
        b = taxonomy_summary["buckets"]
        output.append(
            fmt.format(
                "missing_or_ranking",
                b["missing_or_ranking"]["count"],
                f"{b['missing_or_ranking']['pct_of_all']}%",
                f"Absent top-k (dist={missing_distractor}, miss={missing_uningested})",
                b["missing_or_ranking"]["primary_action"],
            )
        )
        output.append(
            fmt.format(
                "ranking",
                b["ranking"]["count"],
                f"{b['ranking']['pct_of_all']}%",
                "Retrieved rank 2-k, missed top-1",
                b["ranking"]["primary_action"],
            )
        )
        output.append(
            fmt.format(
                "false_answer",
                b["false_answer"]["count"],
                f"{b['false_answer']['pct_of_all']}%",
                "Under-refusal (hallucinated)",
                b["false_answer"]["primary_action"],
            )
        )
        output.append(
            fmt.format(
                "over_refuse",
                b["over_refuse"]["count"],
                f"{b['over_refuse']['pct_of_all']}%",
                "Over-refusal on answerable item",
                b["over_refuse"]["primary_action"],
            )
        )

    # Top sample failures
    output.append(f"\n{bold}Top Sample Failures by Bucket:{bold}")
    for name, sample_rows in buckets.items():
        output.append(f"\n--- {bold}{name}{bold} ({len(sample_rows)} cases) ---")
        for item in sample_rows[:3]:
            qid = item.get("id", "?")
            q = (item.get("question") or "")[:75]
            exp = item.get("expected_paper_ids", [])
            ret = item.get("retrieved_paper_ids", [])
            output.append(f"  • [{qid}] {q}...")
            output.append(f"    Expected: {exp} | Retrieved: {ret[:3]}")

    print("\n".join(output))


if __name__ == "__main__":
    main()
