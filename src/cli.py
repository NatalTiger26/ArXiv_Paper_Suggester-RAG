#!/usr/bin/env python3
"""
Unified CLI Manager for ArXiv Paper Suggester.

Usage:
  python -m src.cli search "mechanistic interpretability"
  python -m src.cli rank "neural tangent kernel" -k 5
  python -m src.cli ingest 1806.07572
  python -m src.cli ask "What is the main contribution of paper 1806.07572?"
  python -m src.cli novelty 1806.07572
  python -m src.cli enrich --limit 50
  python -m src.cli health
  python -m src.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_search(args: argparse.Namespace) -> int:
    from src.catalog import search_catalog

    results = search_catalog(args.query, limit=args.limit)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    print(f"\nCatalog Search: '{args.query}' ({len(results)} matches)\n" + "-" * 70)
    for i, r in enumerate(results, 1):
        cites = f"{r['citation_count']} cites" if r.get("citation_count") is not None else "no cites"
        print(f"{i:2}. [{r.get('arxiv_id') or r.get('paper_id')}] ({r.get('year') or '—'}, {cites})")
        print(f"    {r.get('title')}")
    print("-" * 70)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    from src.ranking import parse_weights, score_and_rank, print_ranked_results, DEFAULT_WEIGHTS

    weights = parse_weights(args.weights) if args.weights else DEFAULT_WEIGHTS
    results = score_and_rank(
        query=args.query,
        k=args.k,
        weights=weights,
        temperature=args.temperature,
        half_life=args.half_life,
        use_llm=args.llm,
    )
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    print_ranked_results(args.query, results)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from src.rag.ingest import ingest_paper
    from src.rag.download import normalise_arxiv_id

    aid = normalise_arxiv_id(args.arxiv_id) or args.arxiv_id
    print(f"[cli] Ingesting paper {aid} into Chroma RAG...")
    res = ingest_paper(aid)
    print(f"[cli] Done: {res}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from src.rag.rag_pipeline import ask

    res = ask(
        question=args.query,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        return_chunks=True,
    )
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 70)
    print(f"Question: {args.query}")
    print(f"Coverage: {res.get('coverage')} | Cache Hit: {res.get('cache_hit')}")
    print("=" * 70)
    print(f"\n{res.get('answer')}\n")
    refs = res.get("references") or []
    if refs:
        print("References:")
        for i, ref in enumerate(refs, 1):
            print(f"  [{i}] {ref.get('paper_id')} § {ref.get('section', 'body')}")
    print("=" * 70)
    return 0


def cmd_novelty(args: argparse.Namespace) -> int:
    from src.novelty.score import score_paper
    from src.rag.download import normalise_arxiv_id

    aid = normalise_arxiv_id(args.arxiv_id) or args.arxiv_id
    profile = score_paper(aid)
    if args.json:
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 70)
    print(f"Novelty Subspace Analysis: {aid}")
    print("=" * 70)
    print(f"Max Cosine (m):              {profile.get('m', 0.0):.4f}")
    print(f"Synergy Blend (Delta):       {profile.get('Delta', 0.0):.4f}")
    print(f"Cone Reconstruction (R2_+):  {profile.get('R2_plus', 0.0):.4f}")
    print(f"Novelty Score:               {profile.get('novelty_score', 0.0):.4f}")
    attr = profile.get("attribution") or []
    if attr:
        print("\nDominant NNLS Attributions:")
        for a in attr[:5]:
            print(f"  - {a['paper_id']}: lambda = {a['weight']:.4f}")
    print("=" * 70)
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "enrich_citation_graph.py")]
    return subprocess.run(cmd).returncode


def cmd_health(args: argparse.Namespace) -> int:
    from src.catalog import catalog_stats
    from src.telemetry import get_telemetry_summary

    print("\n" + "=" * 70)
    print("System Health & Status")
    print("=" * 70)
    cat = catalog_stats()
    print(f"SQLite Catalog: {cat.get('total_papers', 0)} papers (FTS5 enabled: {cat.get('fts_enabled')})")

    try:
        from src.rag.embed_store import get_collection
        coll = get_collection(create=False)
        print(f"Chroma RAG Collection: {coll.count()} chunks")
    except Exception as e:
        print(f"Chroma RAG Collection: error ({e})")

    tel = get_telemetry_summary()
    print(f"Gemini Telemetry: {tel.get('total_calls', 0)} API calls recorded (${tel.get('total_cost_usd', 0.0):.4f} USD)")
    print("=" * 70)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from src.api.server import run
    run(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_batch_novelty(args: argparse.Namespace) -> int:
    from scripts.batch_novelty_library import run_batch_novelty
    run_batch_novelty(
        limit=args.limit,
        category=args.category,
        min_citations=args.min_citations,
        resume=not args.no_resume,
    )
    return 0


def cmd_bulk_ingest(args: argparse.Namespace) -> int:
    from scripts.bulk_topic_ingest import parse_topic_list, run_bulk_ingest
    topics = parse_topic_list(args.topics, args.topics_file)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    run_bulk_ingest(
        topics=topics,
        max_per_topic=args.max_per_topic,
        categories=cats,
        start_year=args.start_year,
        min_citations=args.min_citations,
        full_text=args.full_text,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Unified CLI Manager for ArXiv Paper Suggester",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Fast FTS5 BM25 catalog search")
    p_search.add_argument("query", help="Keyword or title to search")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")
    p_search.add_argument("--json", action="store_true", help="Output JSON")
    p_search.set_defaults(func=cmd_search)

    # rank
    p_rank = sub.add_parser("rank", help="Multi-signal paper scoring and ranking")
    p_rank.add_argument("query", help="Research topic or abstract snippet")
    p_rank.add_argument("-k", type=int, default=10, help="Top k results")
    p_rank.add_argument("--weights", type=str, default=None, help="Weights (emb,rec,cit,auth[,vel])")
    p_rank.add_argument("--half-life", type=float, default=4.0, help="Recency half-life in years")
    p_rank.add_argument("--temperature", type=float, default=0.0, help="MMR diversity temperature")
    p_rank.add_argument("--llm", action="store_true", help="Enable optional LLM judge")
    p_rank.add_argument("--json", action="store_true", help="Output JSON")
    p_rank.set_defaults(func=cmd_rank)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Selectively download, chunk, and embed an arXiv paper")
    p_ingest.add_argument("arxiv_id", help="ArXiv ID (e.g. 1806.07572)")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question with citation-grounded RAG")
    p_ask.add_argument("query", help="Technical question")
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    p_ask.add_argument("--no-cache", action="store_true", help="Disable semantic cache")
    p_ask.add_argument("--json", action="store_true", help="Output JSON")
    p_ask.set_defaults(func=cmd_ask)

    # novelty
    p_nov = sub.add_parser("novelty", help="Compute 2D cone novelty and NNLS attribution")
    p_nov.add_argument("arxiv_id", help="ArXiv ID (e.g. 1806.07572)")
    p_nov.add_argument("--json", action="store_true", help="Output JSON")
    p_nov.set_defaults(func=cmd_novelty)

    # enrich
    p_enrich = sub.add_parser("enrich", help="Enrich citation graph with global counts")
    p_enrich.add_argument("--limit", type=int, default=50, help="Number of hubs to enrich via S2")
    p_enrich.set_defaults(func=cmd_enrich)

    # health
    p_health = sub.add_parser("health", help="Check system health, index status, and telemetry")
    p_health.set_defaults(func=cmd_health)

    # serve
    p_serve = sub.add_parser("serve", help="Launch modern Web Studio and REST API")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    # batch-novelty
    p_bnov = sub.add_parser("batch-novelty", help="Batch compute 2D novelty geometry across catalog")
    p_bnov.add_argument("--limit", type=int, default=50, help="Number of papers to evaluate")
    p_bnov.add_argument("--category", type=str, default=None, help="Filter by category")
    p_bnov.add_argument("--min-citations", type=int, default=0, help="Min citations")
    p_bnov.add_argument("--no-resume", action="store_true", help="Recalculate already scored papers")
    p_bnov.set_defaults(func=cmd_batch_novelty)

    # bulk-ingest
    p_bulk = sub.add_parser("bulk-ingest", help="Autonomous multi-topic bulk paper ingestion")
    p_bulk.add_argument("--topics", nargs="*", help="Topic query phrases")
    p_bulk.add_argument("--topics-file", type=Path, default=None, help="File containing topics")
    p_bulk.add_argument("--max-per-topic", type=int, default=20, help="Max papers per topic")
    p_bulk.add_argument("--categories", type=str, default=None, help="Comma-separated categories")
    p_bulk.add_argument("--start-year", type=int, default=None, help="Minimum year")
    p_bulk.add_argument("--min-citations", type=int, default=0, help="Min citations")
    p_bulk.add_argument("--full-text", action="store_true", help="Download PDFs and ingest into Chroma RAG")
    p_bulk.set_defaults(func=cmd_bulk_ingest)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
