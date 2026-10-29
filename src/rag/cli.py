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
        results = ingest_papers(
            ids,
            force_download=args.force,
            prefer_html=not getattr(args, "force_pdf", False),
            force_pdf=getattr(args, "force_pdf", False),
        )

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDone. {ok}/{len(results)} papers ingested successfully.")
    for r in results:
        if r.get("ok"):
            title = (r.get("title") or "")[:60]
            fmt = r.get("source_format", "pdf")
            print(f"  ✓ {r['arxiv_id']}  ({r['chunks']} chunks, format={fmt})  {title}")
        else:
            print(f"  ✗ {r['arxiv_id']}  : {r.get('error')}")


def cmd_purge_pdfs(args: argparse.Namespace) -> None:
    from .config import PAPERS_DIR
    from .download import normalise_arxiv_id
    from .embed_store import list_ingested_papers
    from .pdf_to_md import markdown_path_for

    if not PAPERS_DIR.exists():
        print("No papers directory found at data/papers/.")
        return

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/papers/.")
        return

    print(f"[purge-pdfs] Checking {len(pdf_files)} local PDFs against RAG index…", flush=True)
    rag_ids = {str(p.get("paper_id") or p.get("arxiv_id") or "").strip() for p in list_ingested_papers()}

    safe_to_purge = []
    skipped = 0
    total_bytes = 0

    for pdf in pdf_files:
        raw_name = pdf.stem.replace("_", "/")
        aid = normalise_arxiv_id(raw_name)
        if not aid:
            skipped += 1
            continue
        md_file = markdown_path_for(aid)
        if md_file.exists() and md_file.stat().st_size > 200:
            if aid in rag_ids:
                size = pdf.stat().st_size
                safe_to_purge.append((pdf, size, aid))
                total_bytes += size
            else:
                skipped += 1
        else:
            skipped += 1

    mb_freed = total_bytes / (1024 * 1024)
    gb_freed = total_bytes / (1024 * 1024 * 1024)
    size_str = f"{gb_freed:.2f} GB" if gb_freed >= 1.0 else f"{mb_freed:.1f} MB"

    if args.dry_run:
        print(f"[dry-run] Found {len(safe_to_purge)} PDFs safely purgeable ({size_str} reclaimable).")
        print(f"[dry-run] Skipped {skipped} PDFs (unconverted or not in RAG index).")
        print("Run without --dry-run to permanently remove these redundant PDFs.")
        return

    purged = 0
    for pdf, _, _ in safe_to_purge:
        try:
            pdf.unlink()
            purged += 1
        except Exception:
            pass

    print(f"Purged {purged} redundant PDFs. Reclaimed {size_str} of disk space.")
    print("Full text and embeddings remain fully preserved in Markdown and ChromaDB.")


def cmd_purge_stale(args: argparse.Namespace) -> None:
    """Scans and safely purges abandoned crash dumps and legacy snapshots."""
    from src.settings import DATA_DIR, PROJECT_ROOT

    stale_dirs: list[Path] = []
    stale_files: list[Path] = []

    # 1. Scanned patterns in data/
    for p in DATA_DIR.glob("chroma_rag.hung_*"):
        if p.is_dir():
            stale_dirs.append(p)
    for p in DATA_DIR.glob("chroma_rag.bak_*"):
        if p.is_dir():
            stale_dirs.append(p)
    for p in DATA_DIR.glob("rag_ingest_progress.bak_*"):
        if p.is_dir():
            stale_dirs.append(p)
    for p in DATA_DIR.glob("corpus.json.bak_*"):
        if p.is_file():
            stale_files.append(p)

    # 2. Abandoned snapshots in backups/
    backups_dir = PROJECT_ROOT / "backups"
    if backups_dir.exists():
        for p in backups_dir.glob("snapshot_*"):
            if p.is_dir():
                stale_dirs.append(p)

    # 3. Scratch files in root
    term_file = PROJECT_ROOT / "terminal_commands.txt"
    if term_file.exists():
        stale_files.append(term_file)

    state_dir = PROJECT_ROOT / "state"
    if state_dir.exists() and not any(state_dir.iterdir()):
        stale_dirs.append(state_dir)

    total_bytes = 0
    for d in stale_dirs:
        for f in d.rglob("*"):
            if f.is_file() and not f.is_symlink():
                try:
                    total_bytes += f.stat().st_size
                except Exception:
                    pass
    for f in stale_files:
        if f.is_file():
            try:
                total_bytes += f.stat().st_size
            except Exception:
                pass

    size_str = (
        f"{total_bytes / (1024 ** 3):.2f} GB"
        if total_bytes >= 1024 ** 3
        else f"{total_bytes / (1024 ** 2):.1f} MB"
    )

    if not stale_dirs and not stale_files:
        print("No stale crash dumps or legacy snapshots found. Workspace is clean!")
        return

    print(f"Found {len(stale_dirs)} stale director{'ies' if len(stale_dirs) != 1 else 'y'} and {len(stale_files)} stale file(s):")
    for d in stale_dirs:
        print(f"  • [dir]  {d.relative_to(PROJECT_ROOT)}")
    for f in stale_files:
        print(f"  • [file] {f.relative_to(PROJECT_ROOT)}")
    print(f"Total reclaimable space: {size_str}")

    if args.dry_run:
        print("\n[dry-run] No items deleted. Run without --dry-run (or with --yes) to purge.")
        return

    if not args.yes:
        confirm = input("\nProceed with deletion of stale items? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    import shutil

    for d in stale_dirs:
        try:
            shutil.rmtree(d)
            print(f"  ✓ Deleted directory {d.name}")
        except Exception as e:
            print(f"  ✗ Failed to remove {d}: {e}")
    for f in stale_files:
        try:
            f.unlink()
            print(f"  ✓ Deleted file {f.name}")
        except Exception as e:
            print(f"  ✗ Failed to remove {f}: {e}")

    print(f"\nPurge complete. Successfully reclaimed {size_str} of disk space.")


def cmd_ask(args: argparse.Namespace) -> None:
    from src.rag.rag_pipeline import ask, export_markdown, format_answer

    is_hybrid = not getattr(args, "dense_only", False)
    result = ask(
        args.question,
        top_k=args.k,
        paper_ids=args.paper_ids or None,
        return_chunks=args.verbose,
        use_cache=not args.no_cache,
        hybrid=is_hybrid,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_answer(result))
        if args.verbose:
            print(f"\n── Retrieval Mode: {'Hybrid BM25 + Dense (RRF)' if is_hybrid else 'Dense Only'} ──")
            if result.get("chunks"):
                for c in result["chunks"]:
                    print(f"  [{c['score']}] {c['title'][:50]} § {c['section']}")

    if args.export:
        path = export_markdown(result, args.question)
        print(f"\nExported → {path}")



def cmd_reset_index(args):
    from .embed_store import reset_rag_collection
    reset_rag_collection()
    print("RAG index cleared. Re-ingest papers with: python -m src.rag.cli ingest --force …")

def cmd_status(args: argparse.Namespace) -> None:
    from src.rag.embed_store import index_stats, list_ingested_papers

    stats = index_stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Chunks : {stats['chunk_count']}")
        print(f"Papers : {stats['paper_count']}")
        print(f"Model  : {stats['embedding_model']}")
        print(f"Store  : {stats['persist_dir']}")
        print("\nIngested papers:")
        papers = stats.get("papers") or []
        if not papers and stats.get("paper_count", 0) > 0:
            all_p = list_ingested_papers()
            for p in all_p[:10]:
                print(f"  • {p['paper_id']}  {p.get('title', '')[:70]}")
            if len(all_p) > 10:
                print(f"  ... and {len(all_p) - 10} more (run `python -m src.rag.cli papers` to view all)")
        elif papers:
            for p in papers[:10]:
                print(f"  • {p['paper_id']}  {p.get('title', '')[:70]}")
            if len(papers) > 10:
                print(f"  ... and {len(papers) - 10} more")
        else:
            print("  (none yet, run `ingest` first)")


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

    is_hybrid = not getattr(args, "dense_only", False)
    report = run_eval(top_k=args.k, skip_generation=args.retrieval_only, hybrid=is_hybrid)
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
    ing.add_argument("--force-pdf", action="store_true", help="Force PDF download even if arXiv HTML is available")
    ing.set_defaults(func=cmd_ingest)

    ask_p = sub.add_parser("ask", help="Ask a question over ingested papers (default: Hybrid BM25 + Dense)")
    ask_p.add_argument("question")
    ask_p.add_argument("-k", type=int, default=6)
    ask_p.add_argument("--paper-ids", nargs="*")
    ask_p.add_argument("--json", action="store_true")
    ask_p.add_argument("-v", "--verbose", action="store_true")
    ask_p.add_argument("--no-cache", action="store_true")
    ask_p.add_argument("--dense-only", action="store_true", help="Disable hybrid BM25 lexical ranking (use dense cosine only)")
    ask_p.add_argument("--hybrid", action="store_true", help="Deprecated alias, hybrid is now enabled by default")
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
    ev.add_argument("--dense-only", action="store_true", help="Disable hybrid BM25 lexical ranking (evaluate dense only)")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    p_purge = sub.add_parser(
        "purge-pdfs",
        help="Purge local raw PDFs whose full text is already in Markdown and ChromaDB",
    )
    p_purge.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files and space would be freed without deleting",
    )
    p_purge.set_defaults(func=cmd_purge_pdfs)

    p_stale = sub.add_parser(
        "purge-stale",
        help="Scan and safely remove abandoned crash dumps (.hung) and legacy snapshot directories",
    )
    p_stale.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report reclaimable space without deleting",
    )
    p_stale.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Execute deletion without interactive confirmation",
    )
    p_stale.set_defaults(func=cmd_purge_stale)

    p_reset = sub.add_parser(
        "reset-index",
        help="Delete RAG Chroma collection (after embedder / EMBEDDING_DIM change)",
    )
    p_reset.set_defaults(func=cmd_reset_index)

    hl = sub.add_parser("health", help="Display full system health and resource metrics")
    hl.set_defaults(func=lambda args: __import__("scripts.system_health", fromlist=["print_health"]).print_health())

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
