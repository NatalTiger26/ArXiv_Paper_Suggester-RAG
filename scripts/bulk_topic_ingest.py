#!/usr/bin/env python3
"""
Autonomous Multi-Topic Bulk Ingestion Pipeline for ArXiv Scientific Discovery Platform.

Fetches, filters, enriches, and ingests research papers from arXiv based on
multi-signal relevance parameters (topics, categories, recency, citations).
Optionally downloads PDFs, extracts Markdown, chunks, and updates the Chroma vector store.

Usage:
    python3 scripts/bulk_topic_ingest.py --topics "mechanistic interpretability" "diffusion models" --max-per-topic 15
    python3 scripts/bulk_topic_ingest.py --topics "state space models" --categories cs.LG,cs.AI --start-year 2022 --full-text
    python3 scripts/bulk_topic_ingest.py --topics-file my_topics.txt --max-per-topic 20 --enrich-citations
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_DB = PROJECT_ROOT / "data" / "catalog.sqlite3"
CORPUS_JSON = PROJECT_ROOT / "data" / "corpus.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "results" / "bulk_topic_ingest_report.md"

DEFAULT_TOPICS = [
    "mechanistic interpretability circuits",
    "diffusion models generative latent",
    "state space models selective SSM mamba",
    "transformer attention linear complexity",
    "reinforcement learning human feedback alignment",
]


def parse_topic_list(topics_arg: list[str] | None, file_arg: Path | None) -> list[str]:
    """Combine and deduplicate topics from CLI arguments and topics file."""
    topics: list[str] = []
    if file_arg and file_arg.exists():
        for line in file_arg.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                topics.append(line)
    if topics_arg:
        for t in topics_arg:
            parts = [p.strip() for p in t.split(",") if p.strip()]
            topics.extend(parts)
    if not topics:
        topics = DEFAULT_TOPICS

    seen = set()
    deduped = []
    for t in topics:
        if t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)
    return deduped


def get_existing_ids() -> set[str]:
    """Retrieve all existing normalized arXiv IDs in catalog and corpus."""
    from src.catalog import clean_arxiv_id

    ids: set[str] = set()
    if CATALOG_DB.exists():
        try:
            conn = sqlite3.connect(str(CATALOG_DB))
            rows = conn.execute("SELECT arxiv_id FROM papers").fetchall()
            conn.close()
            for r in rows:
                cleaned = clean_arxiv_id(r[0])
                if cleaned:
                    ids.add(cleaned)
        except Exception as e:
            print(f"[bulk-ingest] Warning reading catalog: {e}", file=sys.stderr)

    if CORPUS_JSON.exists():
        try:
            with open(CORPUS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        cleaned = clean_arxiv_id(item.get("arxiv_id") or item.get("paper_id"))
                        if cleaned:
                            ids.add(cleaned)
        except Exception:
            pass

    return ids


def fetch_topic_candidates(
    topic: str,
    max_results: int = 25,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
    start_year: int | None = None,
) -> list[dict[str, Any]]:
    """Search arXiv API for topic with polite retries."""
    from src.corpus_builder import normalise_arxiv_id, search_arxiv

    cat_clause = categories[0] if categories and len(categories) == 1 else None
    print(f"  [search] Querying arXiv: '{topic}' (max={max_results}, cat={cat_clause or 'any'})...")

    try:
        raw_papers = search_arxiv(
            query=topic,
            max_results=max_results,
            category=cat_clause,
            sort_by=sort_by,
        )
    except Exception as e:
        print(f"  [search] Error querying arXiv for '{topic}': {e}", file=sys.stderr)
        return []

    filtered: list[dict[str, Any]] = []
    for p in raw_papers:
        aid = normalise_arxiv_id(p.get("arxiv_id") or "")
        if not aid:
            continue
        p["arxiv_id"] = aid

        # Year filter
        year = None
        pub = p.get("published") or ""
        if pub and len(pub) >= 4:
            try:
                year = int(pub[:4])
            except ValueError:
                pass
        p["year"] = year

        if start_year and year and year < start_year:
            continue

        # Multi-category filter check
        if categories and len(categories) > 1:
            p_cats = p.get("categories") or []
            if isinstance(p_cats, str):
                p_cats = [p_cats]
            match = any(any(c.lower() in pcat.lower() for pcat in p_cats) for c in categories)
            if not match:
                continue

        p["search_topic"] = topic
        filtered.append(p)

    return filtered


def run_bulk_ingest(
    topics: list[str],
    max_per_topic: int = 25,
    categories: list[str] | None = None,
    start_year: int | None = None,
    min_citations: int = 0,
    full_text: bool = False,
    enrich_citations: bool = True,
    update_index: bool = False,
    report_md_path: Path = DEFAULT_REPORT_MD,
) -> dict[str, Any]:
    from src.catalog import clean_arxiv_id, connect, upsert_paper_record
    from src.corpus_builder import load_corpus, save_corpus

    t_start = time.perf_counter()
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("🚀 Starting Bulk Topic Ingestion Pipeline")
    print(f"   Topics:           {len(topics)} query phrases")
    print(f"   Max per topic:    {max_per_topic}")
    print(f"   Categories:       {', '.join(categories) if categories else 'All'}")
    print(f"   Start Year:       {start_year or 'Any'}")
    print(f"   Min Citations:    {min_citations}")
    print(f"   Full-Text Ingest: {'Yes (PDF -> Markdown -> Chroma)' if full_text else 'No (Metadata & Catalog only)'}")
    print(f"   Enrich Citations: {'Yes (Semantic Scholar batch sync)' if enrich_citations else 'No'}")
    print("=" * 76)

    existing_ids = get_existing_ids()
    print(f"[bulk-ingest] Found {len(existing_ids)} existing papers across catalog and corpus.\n")

    new_papers: dict[str, dict[str, Any]] = {}
    skipped_existing = 0

    # 1. Fetch papers for each topic
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] Processing Topic: '{topic}'")
        candidates = fetch_topic_candidates(
            topic,
            max_results=max_per_topic,
            categories=categories,
            start_year=start_year,
        )

        for p in candidates:
            aid = p["arxiv_id"]
            if aid in existing_ids:
                skipped_existing += 1
                continue
            if aid not in new_papers:
                new_papers[aid] = p

        # Be polite to arXiv API
        time.sleep(0.5)

    print(f"\n[bulk-ingest] Discovered {len(new_papers)} new unique papers (skipped {skipped_existing} already present).")

    if not new_papers:
        print("[bulk-ingest] No new papers to ingest.")
        _write_ingest_report(
            report_md_path,
            topics=topics,
            new_papers=[],
            total_fetched=0,
            skipped_existing=skipped_existing,
            duration=time.perf_counter() - t_start,
            full_text_count=0,
        )
        return {"new_papers": 0, "ingested_full_text": 0}

    # 2. Enrich Citations via Semantic Scholar batch if requested
    if enrich_citations:
        print("\n[bulk-ingest] Enriching new papers with global citation counts from Semantic Scholar...")
        from scripts.enrich_citation_graph import fetch_s2_batch

        aids_list = list(new_papers.keys())
        batch_size = 50
        enriched_count = 0

        for b_idx in range(0, len(aids_list), batch_size):
            chunk = aids_list[b_idx : b_idx + batch_size]
            keys = [f"ARXIV:{aid}" for aid in chunk]
            try:
                s2_data = fetch_s2_batch(keys)
                for item in s2_data:
                    if not item:
                        continue
                    ext_ids = item.get("externalIds") or {}
                    raw_aid = ext_ids.get("ArXiv") or ext_ids.get("arXiv")
                    cleaned = clean_arxiv_id(raw_aid)
                    c_count = item.get("citationCount")
                    if cleaned and cleaned in new_papers and c_count is not None:
                        new_papers[cleaned]["citation_count"] = c_count
                        enriched_count += 1
            except Exception as e:
                print(f"  [enrich] Batch query error: {e}")
            time.sleep(0.2)

        print(f"[bulk-ingest] Successfully retrieved citation counts for {enriched_count} papers.")

    # 3. Apply min_citations filter if configured
    eligible_papers = list(new_papers.values())
    if min_citations > 0:
        eligible_papers = [
            p for p in eligible_papers if (p.get("citation_count") or 0) >= min_citations
        ]
        print(f"[bulk-ingest] {len(eligible_papers)} papers meet min_citations >= {min_citations}.")

    # 4. Upsert into catalog.sqlite3 and merge into corpus.json
    print("\n[bulk-ingest] Upserting papers into SQLite FTS5 catalog and corpus.json...")
    catalog_conn = connect()
    for p in eligible_papers:
        aid = p["arxiv_id"]
        title = p.get("title") or "Untitled"
        abstract = p.get("summary") or p.get("abstract") or ""
        year = p.get("year")
        cites = p.get("citation_count") or 0
        cats = p.get("categories")
        if isinstance(cats, list):
            cats = " ".join(cats)
        upsert_paper_record(
            catalog_conn,
            arxiv_id=aid,
            title=title,
            abstract=abstract,
            year=year,
            citation_count=cites,
            categories=cats,
        )
    catalog_conn.close()

    # Also update corpus.json for backward compatibility
    existing_corpus = load_corpus()
    by_id = {clean_arxiv_id(p.get("arxiv_id") or ""): p for p in existing_corpus if p.get("arxiv_id")}
    for p in eligible_papers:
        aid = p["arxiv_id"]
        if aid not in by_id:
            by_id[aid] = {
                "arxiv_id": aid,
                "title": p.get("title"),
                "abstract": p.get("summary") or p.get("abstract"),
                "year": p.get("year"),
                "citation_count": p.get("citation_count") or 0,
                "categories": p.get("categories") if isinstance(p.get("categories"), list) else [p.get("categories")],
                "authors": p.get("authors") or [],
            }
    save_corpus(list(by_id.values()))
    print(f"[bulk-ingest] Catalog and corpus updated with {len(eligible_papers)} papers.")

    # 5. Full-Text Ingestion if requested
    full_text_success = 0
    if full_text and eligible_papers:
        print("\n[bulk-ingest] Performing full-text PDF/HTML ingestion and Chroma embedding...")
        from src.rag.ingest import ingest_papers

        ids_to_ingest = [p["arxiv_id"] for p in eligible_papers]
        ingest_results = ingest_papers(ids_to_ingest, prefer_html=True)
        for r in ingest_results:
            if r.get("ok"):
                full_text_success += 1
        print(f"[bulk-ingest] Full-text ingestion complete: {full_text_success}/{len(ids_to_ingest)} succeeded.")

    # 6. Rebuild corpus index if requested
    if update_index:
        print("\n[bulk-ingest] Rebuilding corpus embeddings index...")
        from src.corpus_builder import rebuild_index
        rebuild_index()

    total_duration = time.perf_counter() - t_start
    print("=" * 76)
    print(f"🎉 Bulk Ingestion Complete in {total_duration:.1f}s!")

    _write_ingest_report(
        report_md_path,
        topics=topics,
        new_papers=eligible_papers,
        total_fetched=len(new_papers),
        skipped_existing=skipped_existing,
        duration=total_duration,
        full_text_count=full_text_success,
    )
    print(f"[bulk-ingest] Ingestion report written to: {report_md_path}\n")

    return {
        "new_papers": len(eligible_papers),
        "ingested_full_text": full_text_success,
        "duration_s": round(total_duration, 2),
    }


def _write_ingest_report(
    path: Path,
    topics: list[str],
    new_papers: list[dict[str, Any]],
    total_fetched: int,
    skipped_existing: int,
    duration: float,
    full_text_count: int,
) -> None:
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Bulk Topic Ingestion Pipeline Report",
        "",
        f"**Execution Timestamp:** `{now_str}`  ",
        f"**Total Execution Time:** `{duration:.1f} seconds`  ",
        f"**Total Ingested to Catalog:** `{len(new_papers)} papers`  ",
        f"**Full-Text Ingested into Chroma:** `{full_text_count} papers`  ",
        f"**Pre-existing (Deduplicated):** `{skipped_existing} papers`  ",
        "",
        "## 1. Topics Processed",
        "",
    ]
    for t in topics:
        lines.append(f"- **{t}**")

    lines.extend([
        "",
        "## 2. Ingested Literature Callset",
        "",
        "| arXiv ID | Title | Year | Citations | Topic / Category |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ])

    for p in sorted(new_papers, key=lambda x: -(x.get("citation_count") or 0))[:50]:
        aid = p.get("arxiv_id")
        title = (p.get("title") or "")[:50].replace("|", "-")
        year = p.get("year") or "—"
        cites = f"{p.get('citation_count', 0):,}"
        topic = p.get("search_topic") or p.get("categories") or "—"
        lines.append(
            f"| [{aid}](https://arxiv.org/abs/{aid}) | {title}... | {year} | {cites} | {topic} |"
        )

    if len(new_papers) > 50:
        lines.append(f"\n*... and {len(new_papers) - 50} more papers saved in `data/catalog.sqlite3`.*")

    lines.extend([
        "",
        "---",
        "### Next Recommended Actions",
        "1. **Launch Discovery Web Studio:** `python3 -m src.cli serve` or `make serve`",
        "2. **Ask Questions Grounded in Ingested Works:** `python3 -m src.cli ask \"Your technical question\"`",
        "3. **Compute Batch Novelty Subspaces:** `python3 scripts/batch_novelty_library.py --limit 50`",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Topic Bulk Ingestion Pipeline for ArXiv"
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        help="List of topic query phrases (e.g. 'mechanistic interpretability' 'diffusion')",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        default=None,
        help="Path to text file containing topic phrases (one per line)",
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=20,
        help="Maximum papers to retrieve per topic (default: 20)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated arXiv categories (e.g. 'cs.LG,cs.AI,stat.ML')",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Minimum publication year (e.g. 2021)",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Filter candidate papers with at least this many citations",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Automatically download PDFs/HTML, chunk, and embed into Chroma vector DB",
    )
    parser.add_argument(
        "--enrich-citations",
        action="store_true",
        default=True,
        help="Enrich new papers with citation counts from Semantic Scholar",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Rebuild corpus embeddings index after ingestion",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_MD,
        help="Path to output Markdown report",
    )

    args = parser.parse_args()
    topics = parse_topic_list(args.topics, args.topics_file)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None

    run_bulk_ingest(
        topics=topics,
        max_per_topic=args.max_per_topic,
        categories=cats,
        start_year=args.start_year,
        min_citations=args.min_citations,
        full_text=args.full_text,
        enrich_citations=args.enrich_citations,
        update_index=args.update_index,
        report_md_path=args.output_report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
