#!/usr/bin/env python3
"""
Enrich citation graph database (data/citation_library/graph.sqlite3) with global citation counts.

Operations:
1. Sync: Copies citation counts from data/corpus.json for all matching papers into graph.sqlite3.
2. Enrich: Batch queries Semantic Scholar for the top landmark papers (highest in-library citers)
   to populate global_citation_count.

Usage:
    uv run python scripts/enrich_citation_graph.py --top 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DB = PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3"
CORPUS_JSON = PROJECT_ROOT / "data" / "corpus.json"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


def clean_arxiv_id(val: str | None) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if s.lower().startswith("arxiv:"):
        s = s[6:].strip()
    return re.sub(r"v\d+$", "", s).strip()


def s2_request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    return headers


def fetch_s2_batch(
    paper_keys: list[str],
    max_retries: int = 4,
) -> list[dict[str, Any] | None]:
    if not paper_keys:
        return []
    headers = s2_request_headers()
    params = {"fields": "paperId,externalIds,citationCount,title,year"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                S2_BATCH_URL,
                params=params,
                json={"ids": paper_keys},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                return [None] * len(paper_keys)
            elif resp.status_code == 429:
                wait_sec = (2 ** attempt) * 4.0
                print(f"  [S2 429 Rate Limit] Backing off for {wait_sec:.1f}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr, flush=True)
                time.sleep(wait_sec)
                continue
            else:
                return [None] * len(paper_keys)
        except Exception as e:
            if attempt == max_retries - 1:
                return [None] * len(paper_keys)
            time.sleep(2.0 * (attempt + 1))
    return [None] * len(paper_keys)


def sync_corpus_citations_to_graph(conn: sqlite3.Connection, corpus_path: Path) -> int:
    """Sync citation counts from corpus.json into papers table in graph.sqlite3."""
    if not corpus_path.exists():
        return 0
    try:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load corpus: {e}", file=sys.stderr)
        return 0

    updates = []
    for p in corpus:
        c = p.get("citation_count")
        if c is not None and c != "":
            try:
                c_int = int(c)
                aid = clean_arxiv_id(p.get("arxiv_id"))
                if aid:
                    updates.append((c_int, f"arxiv:{aid}", aid))
            except (ValueError, TypeError):
                continue

    if not updates:
        return 0

    with conn:
        for c_int, cid, aid in updates:
            conn.execute(
                """
                UPDATE papers
                SET citation_count = ?
                WHERE canonical_id = ? OR arxiv_id = ?
                """,
                (c_int, cid, aid),
            )
    return len(updates)


def enrich_top_landmark_hubs(
    conn: sqlite3.Connection,
    top_n: int = 100,
    batch_size: int = 50,
    delay: float = 1.0,
) -> int:
    """Find top cited landmark papers in graph.sqlite3 missing global citation counts and enrich them."""
    cur = conn.execute(
        """
        SELECT
            e.cited_id,
            COUNT(DISTINCT e.citer_id) AS lib_citers,
            p.arxiv_id,
            p.s2_id,
            p.citation_count
        FROM edges e
        LEFT JOIN papers p ON p.canonical_id = e.cited_id
        GROUP BY e.cited_id
        ORDER BY lib_citers DESC
        LIMIT ?
        """,
        (top_n * 2,),
    )
    rows = cur.fetchall()

    targets: list[tuple[str, str]] = []  # (canonical_id, s2_lookup_key)
    for r in rows:
        cid = r[0]
        c_cites = r[4]
        if c_cites is not None and c_cites > 0:
            continue
        # Determine S2 lookup key
        aid = clean_arxiv_id(r[2])
        if not aid and cid.startswith("arxiv:"):
            aid = clean_arxiv_id(cid[6:])
        s2_id = r[3]
        if not s2_id and cid.startswith("s2:"):
            s2_id = cid[3:].strip()

        if aid:
            targets.append((cid, f"ARXIV:{aid}"))
        elif s2_id:
            targets.append((cid, s2_id))
        if len(targets) >= top_n:
            break

    if not targets:
        print("All target landmark papers already have citation counts.")
        return 0

    print(f"Querying Semantic Scholar for {len(targets)} landmark hub papers...")
    enriched_count = 0

    for i in range(0, len(targets), batch_size):
        sub = targets[i : i + batch_size]
        sub_keys = [k for _, k in sub]
        results = fetch_s2_batch(sub_keys)

        with conn:
            for (cid, _), node in zip(sub, results):
                if not node:
                    continue
                cnt = node.get("citationCount")
                if cnt is not None:
                    try:
                        c_val = int(cnt)
                        conn.execute(
                            "UPDATE papers SET citation_count = ? WHERE canonical_id = ?",
                            (c_val, cid),
                        )
                        enriched_count += 1
                    except (ValueError, TypeError):
                        pass

        print(f"  [Batch {i // batch_size + 1}] Processed {len(sub)} papers (total enriched: {enriched_count})")
        if i + batch_size < len(targets) and delay > 0:
            time.sleep(delay)

    return enriched_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich citation graph with global citation counts.")
    parser.add_argument("--top", type=int, default=100, help="Number of top landmark hubs to enrich via S2.")
    parser.add_argument("--db", type=Path, default=GRAPH_DB, help="Path to graph.sqlite3")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: {args.db} does not exist", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        print("1. Syncing known citations from data/corpus.json into graph.sqlite3...")
        synced = sync_corpus_citations_to_graph(conn, CORPUS_JSON)
        print(f"   ✓ Synced {synced} papers from corpus.json.")

        print(f"2. Enriching top {args.top} landmark papers via Semantic Scholar batch API...")
        enriched = enrich_top_landmark_hubs(conn, top_n=args.top)
        print(f"   ✓ Enriched {enriched} landmark papers.")

        # Print summary
        cur = conn.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0")
        total_with_cites = cur.fetchone()[0]
        print(f"\nCompleted! Total papers in graph.sqlite3 with global citation counts: {total_with_cites:,}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
