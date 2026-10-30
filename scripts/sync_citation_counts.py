#!/usr/bin/env python3
"""
Sync Authoritative Citation Counts into SQLite Catalog and Corpus JSON.

Extracts real pre-cached Semantic Scholar citation counts from data/cache/http_cache.sqlite,
falls back to local citation graph in-degrees from data/citation_library/graph.sqlite3,
updates data/catalog.sqlite3 in bulk transactions, and exports to data/corpus.json.

Usage:
    python3 scripts/sync_citation_counts.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog import clean_arxiv_id, connect, sync_to_corpus_json


def extract_cached_s2_citations(cache_db_path: Path) -> dict[str, int]:
    """Extract citation counts from cached Semantic Scholar API responses."""
    if not cache_db_path.exists():
        print(f"[sync-citations] Cache DB not found at {cache_db_path}")
        return {}

    conn = sqlite3.connect(str(cache_db_path))
    s2_cites: dict[str, int] = {}
    query = "SELECT cache_key, response_json FROM http_cache WHERE cache_key LIKE '%semanticscholar.org%'"

    cur = conn.cursor()
    for row in cur.execute(query):
        key, val = row[0], row[1]
        m = re.search(r"ARXIV:([0-9]{4}\.[0-9]{4,5})", key)
        if not m:
            continue
        aid = m.group(1)

        data = None
        if isinstance(val, (bytes, bytearray)):
            try:
                if val.startswith(b"ZLIB:"):
                    decomp = zlib.decompress(val[5:]).decode("utf-8")
                else:
                    decomp = zlib.decompress(val).decode("utf-8")
                data = json.loads(decomp)
            except Exception:
                continue
        elif isinstance(val, str):
            try:
                data = json.loads(val)
            except Exception:
                continue

        if isinstance(data, dict):
            c = data.get("citationCount")
            if c is not None:
                try:
                    count = int(c)
                    if count >= 0:
                        if aid not in s2_cites or count > s2_cites[aid]:
                            s2_cites[aid] = count
                except (ValueError, TypeError):
                    pass

    conn.close()
    return s2_cites


def extract_graph_in_degrees(graph_db_path: Path) -> dict[str, int]:
    """Extract local citation graph in-degrees as lower-bound citation counts."""
    if not graph_db_path.exists():
        return {}

    conn = sqlite3.connect(str(graph_db_path))
    graph_cites: dict[str, int] = {}
    query = "SELECT cited_id, COUNT(*) FROM edges GROUP BY cited_id"

    for row in conn.execute(query):
        cited_id, cnt = str(row[0]), int(row[1])
        clean = re.sub(r"^(?:arxiv:|ARXIV:)", "", cited_id).strip()
        aid = clean_arxiv_id(clean)
        if aid:
            graph_cites[aid] = cnt

    conn.close()
    return graph_cites


def run_sync(
    catalog_path: Path | None = None,
    cache_path: Path | None = None,
    graph_path: Path | None = None,
    corpus_json_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete citation synchronization across catalog and corpus."""
    cat_db = catalog_path or (PROJECT_ROOT / "data" / "catalog.sqlite3")
    cache_db = cache_path or (PROJECT_ROOT / "data" / "cache" / "http_cache.sqlite")
    graph_db = graph_path or (PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3")
    corpus_json = corpus_json_path or (PROJECT_ROOT / "data" / "corpus.json")

    print("[sync-citations] Starting citation count synchronization...")
    t0 = time.perf_counter()

    conn = connect(cat_db)
    cur = conn.cursor()

    before_total = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    before_with_cites = cur.execute(
        "SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0"
    ).fetchone()[0]
    before_null = cur.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NULL").fetchone()[0]

    print(f"[sync-citations] Catalog before sync: {before_total} total, {before_with_cites} with citations, {before_null} NULL")

    s2_cites = extract_cached_s2_citations(cache_db)
    print(f"[sync-citations] Extracted {len(s2_cites)} citation counts from Semantic Scholar cache.")

    graph_cites = extract_graph_in_degrees(graph_db)
    print(f"[sync-citations] Extracted {len(graph_cites)} in-degrees from citation graph.")

    all_papers = cur.execute("SELECT arxiv_id, citation_count FROM papers").fetchall()
    updates: list[tuple[int, float, str]] = []
    now = time.time()

    matched_s2 = 0
    matched_graph = 0
    defaulted_zero = 0

    for row in all_papers:
        aid = row["arxiv_id"]
        curr_c = row["citation_count"]

        new_c = None
        if aid in s2_cites:
            new_c = s2_cites[aid]
            matched_s2 += 1
        elif aid in graph_cites:
            new_c = graph_cites[aid]
            matched_graph += 1
        elif curr_c is not None:
            new_c = curr_c
        else:
            new_c = 0
            defaulted_zero += 1

        if curr_c != new_c:
            updates.append((new_c, now, aid))

    if updates:
        print(f"[sync-citations] Applying {len(updates)} citation count updates to SQLite catalog...")
        with conn:
            conn.executemany(
                "UPDATE papers SET citation_count = ?, updated_at = ? WHERE arxiv_id = ?",
                updates,
            )

    after_total = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    after_with_cites = cur.execute(
        "SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0"
    ).fetchone()[0]
    after_null = cur.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NULL").fetchone()[0]
    after_zero = cur.execute("SELECT COUNT(*) FROM papers WHERE citation_count = 0").fetchone()[0]

    print(f"[sync-citations] Catalog after sync: {after_total} total, {after_with_cites} with citations > 0, {after_zero} zero, {after_null} NULL")
    print(f"[sync-citations] Breakdown: {matched_s2} from S2 cache, {matched_graph} from graph in-degrees, {defaulted_zero} defaulted to 0")

    # Sync changes to data/corpus.json
    print(f"[sync-citations] Syncing catalog to {corpus_json}...")
    synced_json_count = sync_to_corpus_json(corpus_json, conn=conn)
    print(f"[sync-citations] Exported {synced_json_count} papers to {corpus_json}.")

    conn.close()
    elapsed = time.perf_counter() - t0
    print(f"[sync-citations] Completed citation sync in {elapsed:.2f}s.")

    return {
        "before_with_cites": before_with_cites,
        "after_with_cites": after_with_cites,
        "matched_s2": matched_s2,
        "matched_graph": matched_graph,
        "defaulted_zero": defaulted_zero,
        "total_updates": len(updates),
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    run_sync()
