#!/usr/bin/env python3
"""
Cold-start corpus builder.

Build or extend data/corpus.json from:
  - explicit arXiv IDs
  - keyword / category search via the arXiv API

Examples:
  python src/corpus_builder.py from-ids 1806.07572 2501.16496
  python src/corpus_builder.py search "neural tangent kernel" -n 20
  python src/corpus_builder.py search "mechanistic interpretability" --category cs.LG -n 15
  python src/corpus_builder.py status
  python src/corpus_builder.py rebuild-index
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ids import normalise_arxiv_id

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# Be polite to arXiv API
REQUEST_DELAY_S = 0.5


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return re.sub(r"\s+", " ", el.text).strip()


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    full_id = _text(entry.find(f"{ATOM}id"))
    # http://arxiv.org/abs/1806.07572v1
    aid = normalise_arxiv_id(full_id) or ""
    title = _text(entry.find(f"{ATOM}title"))
    summary = _text(entry.find(f"{ATOM}summary"))
    published = _text(entry.find(f"{ATOM}published"))
    authors = [
        _text(a.find(f"{ATOM}name"))
        for a in entry.findall(f"{ATOM}author")
        if _text(a.find(f"{ATOM}name"))
    ]
    cats = [
        c.get("term", "")
        for c in entry.findall(f"{ARXIV_NS}primary_category")
        + entry.findall(f"{ATOM}category")
        if c.get("term")
    ]
    # dedupe categories preserving order
    seen = set()
    categories = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            categories.append(c)

    doi = _text(entry.find(f"{ARXIV_NS}doi")) or None
    return {
        "arxiv_id": aid,
        "title": title,
        "abstract": summary,
        "authors": authors,
        "published_date": published[:10] if published else "",
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "doi": doi,
        "citation_count": None,  # filled later if enrichment available
        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf" if aid else None,
        "arxiv_url": f"https://arxiv.org/abs/{aid}" if aid else None,
    }


def fetch_by_ids(ids: list[str]) -> list[dict[str, Any]]:
    """Fetch metadata for explicit arXiv IDs (batched)."""
    clean = []
    for raw in ids:
        aid = normalise_arxiv_id(raw)
        if aid:
            clean.append(aid)
    if not clean:
        return []

    out: list[dict[str, Any]] = []
    # arXiv allows id_list with commas
    batch_size = 20
    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        url = f"{ARXIV_API}?id_list={','.join(batch)}&max_results={len(batch)}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for entry in root.findall(f"{ATOM}entry"):
            paper = _parse_entry(entry)
            if paper.get("arxiv_id"):
                out.append(paper)
        time.sleep(REQUEST_DELAY_S)
    return out



_ARXIV_STOP = frozenset({
    "a", "an", "the", "of", "for", "in", "on", "to", "and", "or", "via", "with",
    "from", "by", "as", "at", "into", "over", "under", "between", "than", "that",
    "this", "these", "those", "is", "are", "be", "can", "its", "their",
})


def _arxiv_text_clause(query: str) -> str:
    """Build an arXiv API text clause that is not limited to exact full phrases.

    Multi-word Scholar-style strings rarely match as a single ti/abs phrase.
    We AND significant terms (all:field) and also try a short phrase match.
    """
    q = " ".join(query.strip().split())
    if not q:
        return ""
    # Keep hyphenated tokens (Score-CAM, vision-language) intact
    raw_tokens = q.replace(",", " ").split()
    tokens = [t for t in raw_tokens if t.lower() not in _ARXIV_STOP and len(t) > 1]
    if not tokens:
        tokens = raw_tokens
    if len(tokens) == 1:
        return f"all:{tokens[0]}"
    # Cap width so the API query stays usable
    tokens = tokens[:6]
    and_all = " AND ".join(f"all:{t}" for t in tokens)
    if len(tokens) <= 4:
        phrase = " ".join(tokens)
        return f'((ti:"{phrase}" OR abs:"{phrase}") OR ({and_all}))'
    return f"({and_all})"


def search_arxiv(
    query: str,
    max_results: int = 20,
    category: str | None = None,
    sort_by: str = "relevance",
) -> list[dict[str, Any]]:
    """
    Search arXiv.
    sort_by: relevance | lastUpdatedDate | submittedDate
    """
    parts = []
    if category:
        parts.append(f"cat:{category}")
    clause = _arxiv_text_clause(query)
    if clause:
        parts.append(clause)
    if not parts:
        raise ValueError("Provide a query and/or category")

    search_query = " AND ".join(parts)
    sort_map = {
        "relevance": "relevance",
        "updated": "lastUpdatedDate",
        "submitted": "submittedDate",
        "lastUpdatedDate": "lastUpdatedDate",
        "submittedDate": "submittedDate",
    }
    sortBy = sort_map.get(sort_by, "relevance")

    url = (
        f"{ARXIV_API}?search_query={quote_plus(search_query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy={sortBy}&sortOrder=descending"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        paper = _parse_entry(entry)
        if paper.get("arxiv_id"):
            papers.append(paper)
    return papers


def load_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CORPUS_PATH
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def save_corpus(papers: list[dict[str, Any]], path: Path = CORPUS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def merge_papers(
    existing: list[dict[str, Any]],
    new_papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict] = {}
    for p in existing:
        aid = normalise_arxiv_id(str(p.get("arxiv_id") or "")) or str(p.get("doi") or "")
        if aid:
            by_id[aid] = p
    for p in new_papers:
        aid = normalise_arxiv_id(str(p.get("arxiv_id") or ""))
        if not aid:
            continue
        if aid in by_id:
            # fill missing fields only
            for k, v in p.items():
                if v and not by_id[aid].get(k):
                    by_id[aid][k] = v
        else:
            by_id[aid] = p
    return list(by_id.values())


def rebuild_index() -> None:
    """Rebuild corpus Chroma index from corpus."""
    # Import here so CLI help works without heavy deps loaded always
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from corpus_index import build
    except ImportError:
        from src.corpus_index import build

    class Args:
        try:
            from settings import EMBEDDING_MODEL
        except ImportError:
            from src.settings import EMBEDDING_MODEL
        model = EMBEDDING_MODEL
        batch_size = 32

    build(Args())


def corpus_status(path: Path = CORPUS_PATH) -> dict[str, Any]:
    papers = load_corpus(path)
    chroma = DATA_DIR / "chroma_corpus"
    return {
        "corpus_path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
        "n_papers": len(papers),
        "chroma_exists": chroma.exists(),
        "sample_ids": [p.get("arxiv_id") for p in papers[:8]],
    }


def cmd_from_ids(args: argparse.Namespace) -> None:
    papers = fetch_by_ids(args.ids)
    print(f"Fetched {len(papers)} papers from arXiv")
    existing = [] if args.replace else load_corpus()
    merged = merge_papers(existing, papers)
    path = save_corpus(merged)
    print(f"Corpus now has {len(merged)} papers → {path}")
    if args.build_index:
        print("Rebuilding embedding index…")
        rebuild_index()


def cmd_search(args: argparse.Namespace) -> None:
    papers = search_arxiv(
        query=args.query or "",
        max_results=args.n,
        category=args.category,
        sort_by=args.sort,
    )
    clause = _arxiv_text_clause(args.query or "")
    cat = f" cat:{args.category}" if args.category else ""
    print(f"Query clause:{cat} {clause}".rstrip())
    print(f"Found {len(papers)} papers")
    for p in papers[:10]:
        print(f"  {p.get('arxiv_id')}\t{(p.get('title') or '')[:70]}")
    if len(papers) > 10:
        print(f"  … and {len(papers) - 10} more")

    if args.dry_run:
        return

    existing = [] if args.replace else load_corpus()
    merged = merge_papers(existing, papers)
    path = save_corpus(merged)
    print(f"Corpus now has {len(merged)} papers → {path}")
    if args.build_index:
        print("Rebuilding embedding index…")
        rebuild_index()


def cmd_status(_: argparse.Namespace) -> None:
    s = corpus_status()
    print(json.dumps(s, indent=2))


def cmd_rebuild(_: argparse.Namespace) -> None:
    if not CORPUS_PATH.exists():
        print(f"No corpus at {CORPUS_PATH}. Add papers first.", file=sys.stderr)
        sys.exit(1)
    rebuild_index()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build / extend the paper corpus from arXiv")
    sub = p.add_subparsers(dest="cmd", required=True)

    ids = sub.add_parser("from-ids", help="Add papers by arXiv ID")
    ids.add_argument("ids", nargs="+")
    ids.add_argument("--replace", action="store_true", help="Replace corpus instead of merge")
    ids.add_argument("--build-index", action="store_true", help="Rebuild Chroma after save")
    ids.set_defaults(func=cmd_from_ids)

    s = sub.add_parser("search", help="Search arXiv by keywords")
    s.add_argument("query", nargs="?", default="", help="Search keywords")
    s.add_argument("-n", type=int, default=20, help="Max results")
    s.add_argument("--category", default=None, help="e.g. cs.LG, stat.ML")
    s.add_argument("--sort", default="relevance", choices=["relevance", "updated", "submitted"])
    s.add_argument("--replace", action="store_true")
    s.add_argument("--build-index", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="Print results only, do not save")
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("status", help="Show corpus status")
    st.set_defaults(func=cmd_status)

    rb = sub.add_parser("rebuild-index", help="Rebuild corpus Chroma from corpus JSON")
    rb.set_defaults(func=cmd_rebuild)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
