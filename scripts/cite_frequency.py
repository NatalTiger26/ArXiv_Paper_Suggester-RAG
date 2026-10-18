#!/usr/bin/env python3
"""
Citation-frequency analysis over RAG-ingested papers.

For each paper already in the full-text RAG index, fetch its reference list
(Semantic Scholar primary, OpenAlex backup — same path as novelty/refs.py),
cache edges in data/novelty/novelty_cache.sqlite3, then rank references by
how many *ingested* papers cite them.

Usage (project root):
  uv run python scripts/cite_frequency.py
  uv run python scripts/cite_frequency.py --min-count 5 --top 50
  uv run python scripts/cite_frequency.py --limit 100          # first 100 ingested only
  uv run python scripts/cite_frequency.py --sleep 0.4          # gentler on S2
  uv run python scripts/cite_frequency.py --from-cache-only    # no network

Outputs:
  results/cite_analysis/ref_frequency.csv
  results/cite_analysis/ref_frequency.json
  results/cite_analysis/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "cite_analysis"


def _strip(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def list_ingested() -> list[str]:
    from rag.embed_store import list_ingested_papers

    papers = list_ingested_papers()
    ids = []
    for p in papers:
        aid = _strip(str(p.get("paper_id") or p.get("arxiv_id") or ""))
        if aid:
            ids.append(aid)
    # stable unique
    return sorted(set(ids))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference frequency among RAG-ingested papers")
    ap.add_argument("--min-count", type=int, default=3, help="Keep refs cited by ≥ this many ingested papers")
    ap.add_argument("--top", type=int, default=40, help="Print top-N to console / summary")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N ingested ids (0=all)")
    ap.add_argument("--ref-limit", type=int, default=80, help="Max references pulled per paper")
    ap.add_argument("--sleep", type=float, default=0.35, help="Pause between API calls (seconds)")
    ap.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Do not call APIs; only aggregate edges already in novelty SQLite cache",
    )
    ap.add_argument(
        "--also-library",
        action="store_true",
        help="Also mark whether the cited paper is in data/corpus.json",
    )
    args = ap.parse_args()

    ingested = list_ingested()
    if not ingested:
        print("No RAG-ingested papers found. Run ingest / reingest first.")
        sys.exit(1)
    if args.limit > 0:
        ingested = ingested[: args.limit]
    print(f"[cite] RAG ingested papers to scan: {len(ingested)}")

    library_ids: set[str] = set()
    if args.also_library:
        corpus_path = ROOT / "data" / "corpus.json"
        if corpus_path.exists():
            raw = json.loads(corpus_path.read_text(encoding="utf-8"))
            papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
            for p in papers:
                aid = _strip(str(p.get("arxiv_id") or p.get("id") or ""))
                if aid:
                    library_ids.add(aid)
            print(f"[cite] library ids loaded: {len(library_ids)}")

    from novelty.cache import connect, get_ref_ids, set_ref_edges, upsert_paper
    from novelty.ids import canonical_id, strip_version
    from novelty.refs import resolve_focal_and_refs

    conn = connect()
    cite_count: Counter[str] = Counter()
    cite_by: dict[str, set[str]] = defaultdict(set)
    meta: dict[str, dict] = {}
    scanned = 0
    from_cache = 0
    fetched = 0
    empty = 0
    errors = 0

    t0 = time.time()
    for i, aid in enumerate(ingested, 1):
        focal_cid = canonical_id(arxiv_id=strip_version(aid))
        if not focal_cid:
            continue

        ref_ids = get_ref_ids(conn, focal_cid)
        used_cache = bool(ref_ids)

        if not ref_ids and not args.from_cache_only:
            try:
                focal, refs, note = resolve_focal_and_refs(aid, ref_limit=args.ref_limit)
                if focal:
                    upsert_paper(conn, focal)
                ref_cids = []
                for r in refs or []:
                    upsert_paper(conn, r)
                    cid = r.get("canonical_id")
                    if cid:
                        ref_cids.append(cid)
                        meta[cid] = {
                            "title": r.get("title"),
                            "arxiv_id": r.get("arxiv_id"),
                            "year": r.get("year"),
                            "s2_id": r.get("s2_id"),
                            "doi": r.get("doi"),
                        }
                set_ref_edges(conn, focal_cid, ref_cids)
                ref_ids = ref_cids
                fetched += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[cite] error {aid}: {e}", flush=True)
                continue
        elif used_cache:
            from_cache += 1
        else:
            empty += 1
            continue

        if not ref_ids:
            empty += 1
            continue

        scanned += 1
        for cid in ref_ids:
            cite_count[cid] += 1
            cite_by[cid].add(aid)

        if i % 25 == 0 or i == len(ingested):
            elapsed = time.time() - t0
            print(
                f"[cite] {i}/{len(ingested)}  scanned_with_refs={scanned}  "
                f"cache={from_cache} api={fetched} empty={empty} err={errors}  "
                f"{elapsed/60:.1f}m",
                flush=True,
            )

    # Enrich meta from papers table when only cache was used
    try:
        rows = conn.execute(
            "SELECT canonical_id, title, arxiv_id, year, s2_id, doi FROM papers"
        ).fetchall()
        for row in rows:
            cid = row["canonical_id"]
            if cid not in meta:
                meta[cid] = {
                    "title": row["title"],
                    "arxiv_id": row["arxiv_id"],
                    "year": row["year"],
                    "s2_id": row["s2_id"],
                    "doi": row["doi"],
                }
    except Exception:
        pass

    # Rank
    ranked = []
    for cid, cnt in cite_count.most_common():
        if cnt < args.min_count:
            continue
        m = meta.get(cid) or {}
        arxiv = m.get("arxiv_id")
        ranked.append(
            {
                "canonical_id": cid,
                "count": cnt,
                "fraction_of_scanned": round(cnt / max(1, scanned), 4),
                "title": m.get("title") or "",
                "arxiv_id": arxiv or "",
                "year": m.get("year"),
                "doi": m.get("doi") or "",
                "in_library": bool(arxiv and arxiv in library_ids) if library_ids else None,
                "example_citers": sorted(cite_by[cid])[:8],
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ref_frequency.csv"
    json_path = OUT_DIR / "ref_frequency.json"
    md_path = OUT_DIR / "summary.md"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fields = [
            "count",
            "fraction_of_scanned",
            "title",
            "arxiv_id",
            "year",
            "doi",
            "canonical_id",
            "in_library",
            "example_citers",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in ranked:
            r = dict(row)
            r["example_citers"] = ";".join(r.get("example_citers") or [])
            w.writerow(r)

    payload = {
        "n_ingested_input": len(ingested),
        "n_scanned_with_refs": scanned,
        "from_cache": from_cache,
        "fetched_api": fetched,
        "empty": empty,
        "errors": errors,
        "min_count": args.min_count,
        "n_refs_above_cutoff": len(ranked),
        "refs": ranked,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Reference frequency (RAG-ingested papers)",
        "",
        f"- Ingested papers scanned (with ≥1 ref): **{scanned}** / {len(ingested)}",
        f"- API fetches: **{fetched}** · cache hits: **{from_cache}** · empty/missing: **{empty}**",
        f"- Cutoff: cited by ≥ **{args.min_count}** ingested papers",
        f"- Refs above cutoff: **{len(ranked)}**",
        "",
        f"## Top {min(args.top, len(ranked))}",
        "",
        "| # | Count | Title | arXiv | Year |",
        "|---|------:|-------|-------|------|",
    ]
    for i, row in enumerate(ranked[: args.top], 1):
        title = (row.get("title") or "")[:80].replace("|", "/")
        lines.append(
            f"| {i} | {row['count']} | {title} | {row.get('arxiv_id') or '—'} | {row.get('year') or '—'} |"
        )
    lines.append("")
    lines.append(f"Full table: `{csv_path.relative_to(ROOT)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"[cite] wrote {csv_path}")
    print(f"[cite] wrote {json_path}")
    print(f"[cite] wrote {md_path}")
    print()
    print(f"Top {min(args.top, len(ranked))} (min_count={args.min_count}):")
    for i, row in enumerate(ranked[: args.top], 1):
        print(
            f"  {i:3d}. n={row['count']:4d}  "
            f"{(row.get('arxiv_id') or '—'):12s}  "
            f"{(row.get('title') or '')[:70]}"
        )


if __name__ == "__main__":
    main()
