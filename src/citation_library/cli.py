"""
CLI: in-library citation graph.

Examples:
  # 1) Collect refs for every corpus paper (resumable, long-running)
  uv run python -m src.citation_library.cli collect --source library --sleep 0.35

  # 2) Same, but only RAG full-text papers
  uv run python -m src.citation_library.cli collect --source rag

  # 3) Aggregate + export ranked library / ingest candidates
  uv run python -m src.citation_library.cli report --min-count 3 --top 40

  # 4) Collect then report
  uv run python -m src.citation_library.cli run --source library --min-count 3

  # 5) Status of the graph DB
  uv run python -m src.citation_library.cli status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def cmd_collect(args: argparse.Namespace) -> None:
    from citation_library.collect import collect

    collect(
        source=args.source,
        ids_file=Path(args.ids_file) if args.ids_file else None,
        limit=args.limit,
        ref_limit=args.ref_limit,
        sleep=args.sleep,
        force=args.force,
        max_errors=args.max_errors,
        enrich_abstracts=bool(getattr(args, "enrich_abstracts", False)),
    )


def cmd_report(args: argparse.Namespace) -> None:
    from citation_library.report import build_ranking, write_outputs

    # Fill blank titles for highly cited works (one batch API call)
    try:
        from citation_library.enrich import enrich_missing_titles
        enrich_missing_titles(limit=getattr(args, "enrich_limit", 250) or 250)
    except Exception as e:
        print(f"[cite] title enrich skipped: {e}", flush=True)

    ranked = build_ranking(min_count=args.min_count)
    paths = write_outputs(ranked, min_count=args.min_count, top=args.top)
    print()
    print(f"Top {min(args.top, len(ranked))} by in-library citation count:")
    for i, row in enumerate(ranked[: args.top], 1):
        flag = " [in lib]" if row.get("in_library") else ""
        aid = (row.get("arxiv_id") or "").strip()
        title = (row.get("title") or "").strip()
        if not title:
            title = (row.get("canonical_id") or "")[:40]
        label = aid if aid else (row.get("s2_id") or "—")[:12]
        print(
            f"  {i:3d}. n={row['library_cite_count']:4d}  "
            f"{label:12s}  "
            f"{title[:65]}{flag}"
        )
    print()
    print(f"Candidate ids for ingest: {paths['candidate_ids']}")


def cmd_run(args: argparse.Namespace) -> None:
    cmd_collect(args)
    cmd_report(args)


def cmd_status(_args: argparse.Namespace) -> None:
    from citation_library.store import connect, stats

    conn = connect()
    st = stats(conn)
    conn.close()
    print("[cite] graph DB stats:")
    for k, v in st.items():
        print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="citation_library",
        description="Build an in-library citation graph and rank important references.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_collect_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--source",
            choices=["library", "rag", "ids"],
            default="library",
            help="library=corpus.json (default) · rag=full-text only · ids=file",
        )
        p.add_argument("--ids-file", type=str, default=None, help="One arXiv id per line")
        p.add_argument("--limit", type=int, default=0, help="Only first N papers (0=all)")
        p.add_argument("--ref-limit", type=int, default=100, help="Max refs per paper")
        p.add_argument("--sleep", type=float, default=0.35, help="Seconds between API calls")
        p.add_argument("--force", action="store_true", help="Re-fetch even if status=ok")
        p.add_argument("--max-errors", type=int, default=80, help="Stop after this many errors")
        p.add_argument(
            "--enrich-abstracts",
            action="store_true",
            help="Also fill ref abstracts (slow; only needed for novelty scoring, not cite counts)",
        )

    p_col = sub.add_parser("collect", help="Fetch reference lists (resumable)")
    add_collect_flags(p_col)
    p_col.set_defaults(func=cmd_collect)

    p_rep = sub.add_parser("report", help="Aggregate counts + export files")
    p_rep.add_argument("--min-count", type=int, default=3)
    p_rep.add_argument("--top", type=int, default=40)
    p_rep.add_argument("--enrich-limit", type=int, default=250, help="Batch-fill titles for top cited works missing metadata")
    p_rep.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="collect then report")
    add_collect_flags(p_run)
    p_run.add_argument("--min-count", type=int, default=3)
    p_run.add_argument("--top", type=int, default=40)
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("status", help="Show graph DB stats")
    p_st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
