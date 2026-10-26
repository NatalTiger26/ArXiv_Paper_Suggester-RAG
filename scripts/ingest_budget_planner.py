#!/usr/bin/env python3
"""
Ingest Budget Planner (ConfAI / Phase 5 Resource Optimizer).
Given a pool of candidate arXiv papers and a time/disk budget, plans the optimal
subset of PDFs to download, parse, and embed for maximum Q&A question coverage.

Uses empirical costs recorded in `results/ingest_profile.jsonl` (or defaults:
  ~3.8s ingest time, ~1.5 MB disk per paper).

Usage:
  python scripts/ingest_budget_planner.py --max-time-s 60
  python scripts/ingest_budget_planner.py --max-disk-mb 100 --max-papers 15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "results" / "ingest_profile.jsonl"
GOLD_PATH = ROOT / "results" / "rag_eval" / "gold_questions.json"
CORPUS_PATH = ROOT / "data" / "corpus.json"


def get_empirical_costs() -> tuple[float, float]:
    """Returns (avg_seconds_per_paper, avg_mb_per_paper)."""
    default_secs = 4.0
    default_mb = 1.8

    if not PROFILE_PATH.exists():
        return default_secs, default_mb

    secs = []
    for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if "ingest_duration_s" in entry:
                secs.append(float(entry["ingest_duration_s"]))
        except Exception:
            continue

    avg_s = sum(secs) / len(secs) if secs else default_secs
    return round(avg_s, 2), default_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Budget Planner")
    parser.add_argument(
        "--max-time-s",
        type=float,
        default=60.0,
        help="Maximum allowable ingest time in seconds (default: 60s)",
    )
    parser.add_argument(
        "--max-disk-mb",
        type=float,
        default=150.0,
        help="Maximum allowable disk space in MB (default: 150 MB)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=25,
        help="Hard cap on number of papers to ingest (default: 25)",
    )
    args = parser.parse_args()

    avg_s_per_paper, avg_mb_per_paper = get_empirical_costs()

    # Load gold question demands
    paper_to_q_ids: dict[str, set[str]] = {}
    if GOLD_PATH.exists():
        gold_data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        for q in gold_data:
            qid = q.get("id", "")
            for pid in q.get("expected_paper_ids", []):
                norm = pid.replace("arXiv:", "").strip()
                paper_to_q_ids.setdefault(norm, set()).add(qid)

    # Load corpus titles
    corpus_titles: dict[str, str] = {}
    if CORPUS_PATH.exists():
        try:
            corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
            for p in corpus:
                pid = str(p.get("arxiv_id", "")).strip()
                title = p.get("title", pid)
                if pid:
                    corpus_titles[pid] = title
        except Exception:
            pass

    # Score candidates: prioritize papers that unlock the most questions
    # Fallback to corpus order if not in gold set
    candidates = sorted(
        paper_to_q_ids.keys(),
        key=lambda k: len(paper_to_q_ids[k]),
        reverse=True,
    )

    # Greedy budget allocation
    planned = []
    accum_time = 0.0
    accum_disk = 0.0
    covered_q_ids: set[str] = set()

    for pid in candidates:
        if len(planned) >= args.max_papers:
            break
        if accum_time + avg_s_per_paper > args.max_time_s:
            break
        if accum_disk + avg_mb_per_paper > args.max_disk_mb:
            break

        planned.append(pid)
        accum_time += avg_s_per_paper
        accum_disk += avg_mb_per_paper
        covered_q_ids.update(paper_to_q_ids.get(pid, set()))

    print("=== Ingest Budget Planner ===")
    print(f"Cost Model: {avg_s_per_paper}s / paper, {avg_mb_per_paper} MB / paper")
    print(f"Budget: Max {args.max_time_s}s | Max {args.max_disk_mb} MB | Max {args.max_papers} papers\n")

    print(f"Proposed Ingest Plan: {len(planned)} papers")
    print(f"Estimated Duration: {accum_time:.1f}s (budget remaining: {args.max_time_s - accum_time:.1f}s)")
    print(f"Estimated Disk Use: {accum_disk:.1f} MB (budget remaining: {args.max_disk_mb - accum_disk:.1f} MB)")
    print(f"Evaluation Question Coverage: {len(covered_q_ids)} questions enabled\n")

    fmt = "{:<4} | {:<12} | {:<12} | {:<50}"
    print(fmt.format("#", "arXiv ID", "Qs Unlocked", "Title"))
    print("-" * 86)
    for i, pid in enumerate(planned, 1):
        q_count = len(paper_to_q_ids.get(pid, set()))
        title = (corpus_titles.get(pid) or "—")[:48]
        print(fmt.format(i, pid, q_count, title))


if __name__ == "__main__":
    main()
