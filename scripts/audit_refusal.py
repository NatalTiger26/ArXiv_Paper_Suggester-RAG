#!/usr/bin/env python3
"""Audit refusal classifications across official vs broader substring lists.

Usage:
  python scripts/audit_refusal.py results/confai_analysis/G_full_gen_rows.json
  python scripts/audit_refusal.py results/rag_eval/final_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

BROADER_REFUSAL_EXTRA = [
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
    return any(k in a for k in (OFFICIAL_REFUSAL_KEYS + BROADER_REFUSAL_EXTRA))


def audit_rows(rows: list[dict[str, Any]]) -> None:
    print(f"\nAuditing {len(rows)} rows...")

    for label, grader in [
        ("OFFICIAL GRADER (Official Poster Definition)", looks_like_refusal_official),
        ("BROADER GRADER (Exploratory / Diagnostic)", looks_like_refusal_broader),
    ]:
        tp, fp, tn, fn = 0, 0, 0, 0
        fp_examples = []
        fn_examples = []

        for r in rows:
            # Check for answer text in various fields
            ans = r.get("answer_full") or r.get("answer") or r.get("answer_preview") or ""
            should_refuse = bool(r.get("should_refuse"))
            did_refuse = grader(ans)

            if should_refuse and did_refuse:
                tp += 1
            elif not should_refuse and did_refuse:
                fp += 1
                fp_examples.append((r.get("id", "?"), r.get("question", "")[:60], ans[:120]))
            elif not should_refuse and not did_refuse:
                tn += 1
            elif should_refuse and not did_refuse:
                fn += 1
                fn_examples.append((r.get("id", "?"), r.get("question", "")[:60], ans[:120]))

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        acc = (tp + tn) / len(rows) if rows else 0.0

        print(f"\n=======================================================")
        print(f" {label}")
        print(f"=======================================================")
        print(f" Confusion Matrix:")
        print(f"                 Actual Refuse (OOS)   Actual Answerable")
        print(f"  Did Refuse:    TP = {tp:<6}            FP = {fp:<6}")
        print(f"  Did Answer:    FN = {fn:<6}            TN = {tn:<6}")
        print(f"-------------------------------------------------------")
        print(f" Precision : {prec:.4f} ({tp}/{tp + fp})")
        print(f" Recall    : {rec:.4f} ({tp}/{tp + fn})")
        print(f" Accuracy  : {acc:.4f} ({tp + tn}/{len(rows)})")

        if fp_examples:
            print(f"\n Sample False Positives (Over-refusals) [max 3]:")
            for qid, q, snippet in fp_examples[:3]:
                print(f"   [{qid}] Q: {q}")
                print(f"         Snippet: {snippet.replace(chr(10), ' ')}...")
        if fn_examples:
            print(f"\n Sample False Negatives (Hallucinated/OOS answers) [max 3]:")
            for qid, q, snippet in fn_examples[:3]:
                print(f"   [{qid}] Q: {q}")
                print(f"         Snippet: {snippet.replace(chr(10), ' ')}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit refusal classifications")
    parser.add_argument("file", help="Path to final_eval.json, G_full_gen_rows.json, or full_answers_*.json")
    args = parser.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("rows") or data.get("_full_answers") or []

    if not rows:
        raise SystemExit(f"No rows found in {p}")

    audit_rows(rows)


if __name__ == "__main__":
    main()
