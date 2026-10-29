#!/usr/bin/env python3
"""
Enrich ArXiv corpus with citation counts from Semantic Scholar Batch API.

Features:
- Fast batch lookup (up to 100 papers per S2 POST request).
- Polite rate-limit handling with exponential backoff on HTTP 429.
- Atomic corpus write (.tmp + rename) with automatic pre-enrichment backup.
- In-place Chroma metadata update without touching vectors or documents.
- Safe resume capability: skips already enriched papers by default.

Usage:
    # Dry run on 20 papers
    python scripts/enrich_citations.py --limit 20 --dry-run

    # Enrich first 200 un-enriched papers and sync to Chroma
    python scripts/enrich_citations.py --limit 200

    # Enrich all un-enriched papers
    python scripts/enrich_citations.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.core.ids import normalise_arxiv_id

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus.json"
DEFAULT_CHROMA = PROJECT_ROOT / "data" / "chroma_corpus"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


def clean_arxiv_id(aid: str | None) -> str:
    """Normalize arXiv ID for S2 lookup delegating to canonical src.core.ids."""
    res = normalise_arxiv_id(aid)
    return res if res else ""


def s2_request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    return headers


def fetch_s2_batch(
    arxiv_ids: list[str],
    max_retries: int = 4,
) -> list[dict[str, Any] | None]:
    """
    Fetch paper metadata for a batch of arXiv IDs from Semantic Scholar.
    Returns list of paper dictionaries or None corresponding to the requested IDs.
    """
    if not arxiv_ids:
        return []

    headers = s2_request_headers()
    keys = [f"ARXIV:{aid}" for aid in arxiv_ids]
    params = {"fields": "paperId,externalIds,citationCount"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                S2_BATCH_URL,
                params=params,
                json={"ids": keys},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                return [None] * len(arxiv_ids)
            elif resp.status_code == 429:
                wait_sec = (2 ** attempt) * 4.0
                print(
                    f"\n  [S2 429 Rate Limit] Backing off for {wait_sec:.1f}s (attempt {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_sec)
                continue
            else:
                print(
                    f"\n  [S2 Error] HTTP {resp.status_code}: {resp.text[:200]}",
                    file=sys.stderr,
                    flush=True,
                )
                return [None] * len(arxiv_ids)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n  [S2 Network Error] {e}", file=sys.stderr, flush=True)
                return [None] * len(arxiv_ids)
            time.sleep(2.0 * (attempt + 1))

    return [None] * len(arxiv_ids)


def sync_chroma_citations(
    chroma_dir: Path,
    updates: dict[str, int],
) -> int:
    """
    Update citation_count metadata in Chroma collection 'arxiv_corpus' in-place.
    Does not re-embed or re-insert vectors/documents.
    """
    if not updates or not chroma_dir.exists():
        return 0

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection("arxiv_corpus")
    except Exception as e:
        print(f"  [Chroma Warning] Could not open Chroma collection: {e}", file=sys.stderr)
        return 0

    target_ids = list(updates.keys())
    batch_size = 500
    synced = 0

    for i in range(0, len(target_ids), batch_size):
        sub_ids = target_ids[i : i + batch_size]
        try:
            got = coll.get(ids=sub_ids, include=["metadatas"])
            found_ids = got.get("ids") or []
            found_metas = got.get("metadatas") or []
            if not found_ids:
                continue

            updated_ids = []
            updated_metas = []
            for paper_id, meta in zip(found_ids, found_metas):
                if paper_id in updates:
                    new_meta = dict(meta or {})
                    new_meta["citation_count"] = str(updates[paper_id])
                    updated_ids.append(paper_id)
                    updated_metas.append(new_meta)

            if updated_ids:
                coll.update(ids=updated_ids, metadatas=updated_metas)
                synced += len(updated_ids)
        except Exception as e:
            print(f"  [Chroma Warning] Update failed for sub-batch: {e}", file=sys.stderr)

    return synced


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch enrich ArXiv corpus with Semantic Scholar citation counts."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to corpus.json",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA,
        help="Path to chroma_corpus index directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum papers to enrich (default: 200). 0 means all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enrich all eligible papers without limit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich papers even if citation_count is already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate API calls and display statistics without saving.",
    )
    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="Skip updating Chroma metadata in-place.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of IDs per S2 batch request (default: 50, max: 100).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Polite pause between API batches in seconds (default: 0.5).",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus file not found at {args.corpus}", file=sys.stderr)
        return 1

    print(f"Loading corpus from {args.corpus}...")
    try:
        corpus: list[dict[str, Any]] = json.loads(args.corpus.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading {args.corpus}: {e}", file=sys.stderr)
        return 1

    total_corpus = len(corpus)
    already_with_cites = sum(
        1 for p in corpus if p.get("citation_count") is not None and p.get("citation_count") != ""
    )
    print(
        f"Corpus loaded: {total_corpus:,} papers ({already_with_cites:,} already have citation_count)."
    )

    # Filter target papers
    targets: list[tuple[int, str, dict[str, Any]]] = []
    for idx, paper in enumerate(corpus):
        cites = paper.get("citation_count")
        if not args.force and cites is not None and cites != "":
            continue
        aid = clean_arxiv_id(paper.get("arxiv_id"))
        if aid:
            targets.append((idx, aid, paper))

    if not targets:
        print("No papers require citation enrichment. All set!")
        return 0

    max_enrich = len(targets) if (args.all or args.limit <= 0) else min(args.limit, len(targets))
    selected_targets = targets[:max_enrich]
    print(f"Targeting {len(selected_targets):,} papers for citation enrichment.")

    batch_size = max(1, min(100, args.batch_size))
    batches = [
        selected_targets[i : i + batch_size]
        for i in range(0, len(selected_targets), batch_size)
    ]

    newly_enriched = 0
    not_found = 0
    total_citations_added = 0
    chroma_updates: dict[str, int] = {}

    print(
        f"Querying Semantic Scholar in {len(batches)} batches of up to {batch_size} IDs..."
    )

    for b_idx, batch in enumerate(batches, start=1):
        batch_aids = [aid for _, aid, _ in batch]
        results = fetch_s2_batch(batch_aids)

        batch_hits = 0
        for (_, aid, paper), s2_node in zip(batch, results):
            if not s2_node:
                not_found += 1
                continue

            c_count = s2_node.get("citationCount")
            if c_count is not None:
                try:
                    c_int = int(c_count)
                except (ValueError, TypeError):
                    c_int = 0
                paper["citation_count"] = c_int
                # Record for Chroma sync (use exact arxiv_id from corpus)
                orig_id = str(paper.get("arxiv_id") or aid)
                chroma_updates[orig_id] = c_int
                chroma_updates[aid] = c_int  # also map clean aid if different
                total_citations_added += c_int
                newly_enriched += 1
                batch_hits += 1
            else:
                not_found += 1

        print(
            f"  [Batch {b_idx:02d}/{len(batches):02d}] {batch_hits}/{len(batch)} papers matched. "
            f"(Cumulative enriched: {newly_enriched:,})",
            flush=True,
        )

        if b_idx < len(batches) and args.delay > 0:
            time.sleep(args.delay)

    print("\n" + "=" * 50)
    print("Enrichment Run Summary:")
    print(f"  Papers processed:        {len(selected_targets):,}")
    print(f"  Successfully enriched:   {newly_enriched:,}")
    print(f"  Not found / no count:    {not_found:,}")
    print(f"  Total citations counted: {total_citations_added:,}")
    print("=" * 50)

    if args.dry_run:
        print("[Dry Run] Skipped writing to disk and Chroma.")
        return 0

    if newly_enriched > 0:
        # Step 1: Backup original corpus if backup does not already exist
        bak_path = args.corpus.with_name(f"{args.corpus.name}.bak_before_enrich")
        if not bak_path.exists():
            print(f"Creating safety backup at {bak_path}...")
            shutil.copy2(args.corpus, bak_path)

        # Step 2: Atomic write of updated corpus
        tmp_path = args.corpus.with_name(f"{args.corpus.name}.tmp_{int(time.time())}")
        print(f"Saving updated corpus atomically to {args.corpus}...")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, args.corpus)
        print("  ✓ Corpus updated successfully.")

        # Step 3: In-place Chroma metadata update
        if not args.no_chroma and args.chroma_dir.exists():
            print(f"Syncing citations to Chroma at {args.chroma_dir}...")
            synced_count = sync_chroma_citations(args.chroma_dir, chroma_updates)
            print(f"  ✓ Synced citation metadata for {synced_count:,} records in Chroma.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
