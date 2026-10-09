#!/usr/bin/env python3
"""
CLI for the RAG extension.

  python -m src.rag.cli ingest 2501.16496 1806.07572
  python -m src.rag.cli ingest --from-ranking --top 8
  python -m src.rag.cli ingest --from-recommend "mechanistic interpretability" -k 4
  python -m src.rag.cli ask "What is the neural tangent kernel?"
  python -m src.rag.cli ask "…" --export
  python -m src.rag.cli eval
  python -m src.rag.cli eval --retrieval-only
  python -m src.rag.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.rag.ingest import ingest_from_personal_ranking, ingest_papers

    ids = list(args.ids or [])

    if args.from_ranking:
        results = ingest_from_personal_ranking(
            top_n=args.top,
            min_score=args.min_score,
            force_download=args.force,
        )
    else:
        if args.from_recommend:
            try:
                from ranking import score_and_rank
            except ImportError:
                try:
                    from src.ranking import score_and_rank
                except ImportError as e:
                    print(
                        f"Could not import ranking ({e}).\n"
                        "  --from-recommend needs the recommender + data/corpus.json.\n"
                        "  Use explicit IDs or --from-ranking instead.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            results_rec = score_and_rank(query=args.from_recommend, k=args.k)
            for r in results_rec:
                aid = r.get("arxiv_id") or r.get("paper_id")
                if aid:
                    ids.append(str(aid))
            print(f"[ingest] collected {len(ids)} ids from recommendations")

        if not ids:
            print("No paper IDs provided.", file=sys.stderr)
            sys.exit(1)
        results = ingest_papers(ids, force_download=args.force)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDone. {ok}/{len(results)} papers ingested successfully.")
    for r in results:
        if r.get("ok"):
            title = (r.get("title") or "")[:60]
            print(f"  ✓ {r['arxiv_id']}  ({r['chunks']} chunks)  {title}")
        else:
            print(f"  ✗ {r['arxiv_id']}  — {r.get('error')}")


def cmd_ask(args: argparse.Namespace) -> None:
    from src.rag.rag_pipeline import ask, export_markdown, format_answer

    result = ask(
        args.question,
        top_k=args.k,
        paper_ids=args.paper_ids or None,
        return_chunks=args.verbose,
        use_cache=not args.no_cache,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_answer(result))
        if args.verbose and result.get("chunks"):
            print("\n── Retrieved chunks ──")
            for c in result["chunks"]:
                print(f"  [{c['score']}] {c['title'][:50]} § {c['section']}")

    if args.export:
        path = export_markdown(result, args.question)
        print(f"\nExported → {path}")


def cmd_status(args: argparse.Namespace) -> None:
    from src.rag.embed_store import index_stats

    stats = index_stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Chunks : {stats['chunk_count']}")
        print(f"Papers : {stats['paper_count']}")
        print(f"Model  : {stats['embedding_model']}")
        print(f"Store  : {stats['persist_dir']}")
        print("\nIngested papers:")
        for p in stats.get("papers") or []:
            print(f"  • {p['paper_id']}  {p.get('title', '')[:70]}")
        if not stats.get("papers"):
            print("  (none yet — run `ingest` first)")


def cmd_papers(args: argparse.Namespace) -> None:
    from src.rag.embed_store import list_ingested_papers

    papers = list_ingested_papers()
    if args.json:
        print(json.dumps(papers, indent=2, ensure_ascii=False))
    else:
        if not papers:
            print("No papers ingested yet.")
            return
        for p in papers:
            print(f"{p['paper_id']}\t{p.get('title', '')}")


def cmd_eval(args: argparse.Namespace) -> None:
    from src.rag.evaluate import print_summary, run_eval

    report = run_eval(top_k=args.k, skip_generation=args.retrieval_only)
    print_summary(report)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.rag.cli",
        description="RAG extension for ArXiv Paper Suggester",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Download, convert, chunk and embed papers")
    ing.add_argument("ids", nargs="*", help="arXiv IDs")
    ing.add_argument("--from-recommend", metavar="QUERY", help="Ingest top-k recommendations")
    ing.add_argument("--from-ranking", action="store_true", help="Ingest from personal_importance.csv")
    ing.add_argument("--top", type=int, default=8, help="Top-N from personal ranking")
    ing.add_argument("--min-score", type=float, default=4.0, help="Min personal score")
    ing.add_argument("-k", type=int, default=5, help="Recommendations to pull")
    ing.add_argument("--force", action="store_true", help="Re-download / re-convert")
    ing.set_defaults(func=cmd_ingest)

    ask_p = sub.add_parser("ask", help="Ask a question over ingested papers")
    ask_p.add_argument("question")
    ask_p.add_argument("-k", type=int, default=6)
    ask_p.add_argument("--paper-ids", nargs="*")
    ask_p.add_argument("--json", action="store_true")
    ask_p.add_argument("-v", "--verbose", action="store_true")
    ask_p.add_argument("--no-cache", action="store_true")
    ask_p.add_argument("--export", action="store_true", help="Write Markdown answer to results/rag_answers/")
    ask_p.set_defaults(func=cmd_ask)

    st = sub.add_parser("status", help="Show index statistics")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    pp = sub.add_parser("papers", help="List ingested papers")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_papers)

    ev = sub.add_parser("eval", help="Run RAG evaluation suite")
    ev.add_argument("-k", type=int, default=6)
    ev.add_argument("--retrieval-only", action="store_true", help="Skip Gemini (no API cost)")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
