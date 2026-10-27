"""
REST API and Web Studio Server for ArXiv Paper Suggester.

Powered by Starlette and Uvicorn.
Endpoints:
  GET  /api/health            - System health, dual index counts, telemetry overview
  GET  /api/search?q=...      - Sub-2ms SQLite FTS5 catalog search
  POST /api/recommend         - Multi-signal ranking (embedding, recency, citations, author, velocity)
  POST /api/ask               - Citation-grounded RAG ask with section priors
  GET  /api/ask/stream?q=...  - Server-Sent Events (SSE) streaming answers and source passages
  POST /api/ingest            - Fast-skip selective paper ingestion
  GET  /api/novelty?arxiv_id= - 2D cone subspace geometry and attribution metrics
  GET  /api/hubs              - Top citation landmark hubs
  GET  /api/telemetry         - Detailed Gemini API usage and latency metrics
  POST /api/telemetry/reset   - Reset recorded telemetry
  GET  /*                     - Static Modern Web Studio
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import PROJECT_ROOT
from src.catalog import search_catalog, get_paper, catalog_stats
from src.telemetry import get_telemetry_summary, reset_telemetry


def sanitize_json_obj(obj: Any) -> Any:
    """Recursively convert NaNs, Infs, and NumPy types into valid JSON-compliant structures."""
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_obj(v) for v in obj]
    if hasattr(obj, "tolist"):
        return sanitize_json_obj(obj.tolist())
    if hasattr(obj, "item"):
        return sanitize_json_obj(obj.item())
    return str(obj)


# ── Health & Diagnostics ───────────────────────────────────────────────────────


async def api_health(request: Request) -> JSONResponse:
    """System health check and index metrics."""
    stats: dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "catalog": catalog_stats(),
    }

    # Citation graph
    graph_db = PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3"
    if graph_db.exists():
        try:
            conn = sqlite3.connect(graph_db)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM papers")
            n_papers = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0")
            n_with_cites = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM edges")
            n_edges = cur.fetchone()[0]
            conn.close()
            stats["citation_graph"] = {
                "total_papers": n_papers,
                "papers_with_citations": n_with_cites,
                "edges": n_edges,
            }
        except Exception as e:
            stats["citation_graph"] = {"error": str(e)}
    else:
        stats["citation_graph"] = {"status": "missing"}

    # Chroma collection stats
    try:
        from src.rag.embed_store import get_collection
        coll = get_collection(create=False)
        stats["chroma_rag"] = {
            "total_chunks": coll.count(),
        }
    except Exception as e:
        stats["chroma_rag"] = {"error": str(e)}

    stats["telemetry"] = get_telemetry_summary()
    return JSONResponse(stats)


# ── Search Catalog (FTS5) ──────────────────────────────────────────────────────


async def api_search(request: Request) -> JSONResponse:
    """Sub-millisecond FTS5 BM25 catalog search."""
    q = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", 20))
    if not q:
        return JSONResponse({"results": [], "count": 0, "query": ""})

    t0 = time.perf_counter()
    results = search_catalog(q, limit=limit)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return JSONResponse({
        "results": results,
        "count": len(results),
        "latency_ms": elapsed_ms,
        "query": q,
    })


# ── Multi-Signal Recommendation ───────────────────────────────────────────────


async def api_recommend(request: Request) -> JSONResponse:
    """Score and rank papers with 5 signals: embedding, recency, citation, author, velocity."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    query = data.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Field 'query' is required."}, status_code=400)

    k = int(data.get("k", 10))
    weights = data.get("weights")
    half_life = float(data.get("half_life", 4.0))
    temperature = float(data.get("temperature", 0.0))
    use_llm = bool(data.get("use_llm", False))

    t0 = time.perf_counter()
    from src.ranking import score_and_rank
    from src.rag.download import normalise_arxiv_id
    from src.catalog import get_paper, connect, upsert_paper_record
    from src.corpus_builder import fetch_by_ids

    # Check if query is directly an arXiv ID or URL
    direct_match = None
    direct_aid = normalise_arxiv_id(query)
    rank_query = query
    if direct_aid:
        p = get_paper(direct_aid)
        if not p:
            try:
                fetched = fetch_by_ids([direct_aid])
                if fetched:
                    item = fetched[0]
                    conn = connect()
                    pub_yr = None
                    try:
                        pub = item.get("published_date") or ""
                        pub_yr = int(str(pub)[:4]) if str(pub)[:4].isdigit() else None
                    except Exception:
                        pass
                    cats = item.get("categories") or ([item.get("primary_category")] if item.get("primary_category") else [])
                    upsert_paper_record(
                        conn,
                        arxiv_id=direct_aid,
                        title=item.get("title") or direct_aid,
                        abstract=item.get("abstract") or "",
                        year=pub_yr,
                        categories=cats,
                        doi=item.get("doi"),
                    )
                    conn.close()
                    p = get_paper(direct_aid)
            except Exception:
                pass
        if p:
            direct_match = {
                "arxiv_id": direct_aid,
                "paper_id": direct_aid,
                "title": p.get("title") or direct_aid,
                "abstract": p.get("abstract") or "",
                "year": p.get("year"),
                "citation_count": p.get("citation_count"),
                "categories": p.get("categories"),
                "final_score": 1.0,
                "is_direct_match": True,
                "signals": {"embedding": 1.0, "recency": 1.0, "citation": 1.0, "velocity": 1.0},
            }
            rank_query = p.get("title") or query

    try:
        ranked = score_and_rank(
            query=rank_query,
            k=k,
            weights=weights,
            half_life=half_life,
            temperature=temperature,
            use_llm=use_llm,
        )
        if direct_match:
            ranked = [direct_match] + [r for r in ranked if (r.get("arxiv_id") or r.get("paper_id")) != direct_aid]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return JSONResponse({
            "query": query,
            "results": ranked,
            "count": len(ranked),
            "latency_ms": elapsed_ms,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Citation-Grounded RAG Ask ─────────────────────────────────────────────────


async def api_ask(request: Request) -> JSONResponse:
    """Citation-grounded RAG ask with section priors and deduplication."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    query = data.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Field 'query' is required."}, status_code=400)

    top_k = int(data.get("top_k", 5))
    use_cache = bool(data.get("use_cache", True))
    paper_ids = data.get("paper_ids")

    t0 = time.perf_counter()
    from src.rag.rag_pipeline import ask

    try:
        res = ask(
            question=query,
            top_k=top_k,
            paper_ids=paper_ids,
            return_chunks=True,
            use_cache=use_cache,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        res["latency_ms"] = elapsed_ms
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_ask_stream(request: Request) -> Response:
    """
    Server-Sent Events (SSE) streaming endpoint.
    Sends retrieved source passages immediately, then streams the answer.
    """
    q = request.query_params.get("q", "").strip()
    top_k = int(request.query_params.get("top_k", 5))
    use_cache = request.query_params.get("use_cache", "true").lower() in ("true", "1")

    if not q:
        return JSONResponse({"error": "Missing 'q' query parameter."}, status_code=400)

    async def event_generator() -> AsyncGenerator[str, None]:
        from src.rag.rag_pipeline import ask

        loop = asyncio.get_running_loop()
        # Run retrieval and generation in executor to not block event loop
        t0 = time.perf_counter()
        res = await loop.run_in_executor(
            None,
            lambda: ask(
                question=q,
                top_k=top_k,
                return_chunks=True,
                use_cache=use_cache,
            ),
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 1. Yield retrieved source passages immediately
        sources_payload = {
            "coverage": res.get("coverage"),
            "references": res.get("references") or [],
            "chunks": res.get("chunks") or [],
            "cache_hit": res.get("cache_hit", False),
            "latency_ms": elapsed_ms,
        }
        yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

        # 2. Stream answer text
        full_answer = res.get("answer") or ""
        # Simulate word/token delivery for smooth UI rendering
        words = full_answer.split(" ")
        for i in range(0, len(words), 3):
            token_slice = " ".join(words[i : i + 3])
            if i > 0:
                token_slice = " " + token_slice
            yield f"event: token\ndata: {json.dumps({'token': token_slice})}\n\n"
            await asyncio.sleep(0.015)

        # 3. Final done event
        yield f"event: done\ndata: {json.dumps({'done': True, 'coverage': res.get('coverage')})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Paper Ingest ──────────────────────────────────────────────────────────────


async def api_ingest(request: Request) -> JSONResponse:
    """Selective paper ingest with fast-skip if already indexed."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    aid = data.get("arxiv_id", "").strip()
    if not aid:
        return JSONResponse({"error": "Field 'arxiv_id' is required."}, status_code=400)

    from src.rag.download import normalise_arxiv_id
    from src.rag.embed_store import _progress_done_ids, get_collection
    from src.rag.ingest import ingest_paper

    clean_id = normalise_arxiv_id(aid) or aid

    # Fast-skip check (<5ms)
    done_ids = set(_progress_done_ids())
    if clean_id in done_ids:
        return JSONResponse({
            "status": "already_indexed",
            "arxiv_id": clean_id,
            "message": f"Paper {clean_id} is already indexed in Chroma RAG.",
            "skipped": True,
        })

    coll = get_collection(create=False)
    existing = coll.get(where={"paper_id": clean_id}, limit=1)
    if existing and existing.get("ids"):
        return JSONResponse({
            "status": "already_indexed",
            "arxiv_id": clean_id,
            "message": f"Paper {clean_id} already has indexed chunks in Chroma RAG.",
            "skipped": True,
        })

    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(None, lambda: ingest_paper(clean_id))

        # Ensure paper record is in catalog.sqlite3 as well
        from src.catalog import connect, get_paper, upsert_paper_record
        from src.corpus_builder import fetch_by_ids
        if not get_paper(clean_id):
            try:
                fetched = fetch_by_ids([clean_id])
                if fetched:
                    p = fetched[0]
                    conn = connect()
                    year = int(str(p.get("published_date", ""))[:4]) if str(p.get("published_date", ""))[:4].isdigit() else None
                    cats = p.get("categories") or ([p.get("primary_category")] if p.get("primary_category") else [])
                    upsert_paper_record(
                        conn,
                        arxiv_id=clean_id,
                        title=p.get("title") or clean_id,
                        abstract=p.get("abstract") or "",
                        year=year,
                        categories=cats,
                        doi=p.get("doi"),
                    )
                    conn.close()
            except Exception:
                pass

        return JSONResponse({
            "status": "ingested",
            "arxiv_id": clean_id,
            "result": res,
            "skipped": False,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Live arXiv Search & Direct Import ──────────────────────────────────────────


async def api_arxiv_search(request: Request) -> JSONResponse:
    """Search live arXiv across any discipline (astro-ph, physics, cs, math, q-bio)."""
    q = request.query_params.get("q", "").strip()
    cat = request.query_params.get("category", "").strip() or None
    limit = int(request.query_params.get("limit", 15))
    if not q:
        return JSONResponse({"results": [], "count": 0, "query": ""})

    from src.corpus_builder import search_arxiv
    from src.catalog import clean_arxiv_id, get_paper
    from src.rag.embed_store import _progress_done_ids, get_collection

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        raw_papers = await loop.run_in_executor(
            None,
            lambda: search_arxiv(query=q, max_results=limit, category=cat, sort_by="relevance"),
        )
    except Exception as e:
        return JSONResponse({"error": str(e), "results": []}, status_code=500)

    done_ids = set(_progress_done_ids())
    coll = get_collection(create=False)

    results = []
    for p in raw_papers:
        aid = p.get("arxiv_id") or ""
        cleaned = clean_arxiv_id(aid)
        in_catalog = bool(get_paper(cleaned)) if cleaned else False
        in_rag = (cleaned in done_ids)
        if not in_rag and cleaned:
            try:
                has_coll = coll.get(where={"paper_id": cleaned}, limit=1)
                in_rag = bool(has_coll and has_coll.get("ids"))
            except Exception:
                pass

        results.append({
            "arxiv_id": cleaned or aid,
            "title": p.get("title") or "Untitled",
            "abstract": p.get("summary") or p.get("abstract") or "",
            "authors": p.get("authors") or [],
            "year": int(str(p.get("published", ""))[:4]) if str(p.get("published", ""))[:4].isdigit() else None,
            "categories": p.get("categories") or ([p.get("primary_category")] if p.get("primary_category") else []),
            "pdf_url": p.get("pdf_url"),
            "arxiv_url": p.get("arxiv_url"),
            "in_catalog": in_catalog,
            "in_rag": in_rag,
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse({
        "results": results,
        "count": len(results),
        "latency_ms": elapsed_ms,
        "query": q,
    })


async def api_arxiv_import(request: Request) -> JSONResponse:
    """Import a specific paper into the catalog and optionally into Chroma RAG."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    raw_id = data.get("arxiv_id", "").strip()
    ingest_rag = bool(data.get("ingest_rag", True))

    if not raw_id:
        return JSONResponse({"error": "Field 'arxiv_id' is required."}, status_code=400)

    from src.corpus_builder import fetch_by_ids, normalise_arxiv_id
    from src.catalog import clean_arxiv_id, connect, get_paper, upsert_paper_record
    from src.rag.ingest import ingest_paper

    aid = normalise_arxiv_id(raw_id) or clean_arxiv_id(raw_id) or raw_id

    # 1. Fetch metadata from arXiv if not already in catalog
    existing = get_paper(aid)
    if not existing:
        fetched = fetch_by_ids([aid])
        if fetched:
            p = fetched[0]
            conn = connect()
            year = None
            try:
                pub = p.get("published_date") or ""
                year = int(str(pub)[:4]) if str(pub)[:4].isdigit() else None
            except Exception:
                pass
            cats = p.get("categories") or ([p.get("primary_category")] if p.get("primary_category") else [])
            upsert_paper_record(
                conn,
                arxiv_id=aid,
                title=p.get("title") or aid,
                abstract=p.get("abstract") or "",
                year=year,
                citation_count=None,
                categories=cats,
                doi=p.get("doi"),
            )
            conn.close()
            existing = get_paper(aid)

    rag_result = None
    if ingest_rag:
        loop = asyncio.get_running_loop()
        try:
            rag_result = await loop.run_in_executor(None, lambda: ingest_paper(aid))
        except Exception as e:
            rag_result = {"ok": False, "error": str(e)}

    return JSONResponse({
        "ok": True,
        "arxiv_id": aid,
        "paper": existing or {"arxiv_id": aid, "title": aid},
        "rag": rag_result,
    })


# ── Novelty & 2D Subspace Geometry ────────────────────────────────────────────


async def api_novelty(request: Request) -> JSONResponse:
    """Compute novelty profile and 2D cone subspace coordinates."""
    aid = request.query_params.get("arxiv_id", "").strip()
    if not aid:
        return JSONResponse({"ok": False, "error": "Missing 'arxiv_id' query parameter."}, status_code=400)

    from src.rag.download import normalise_arxiv_id
    clean_id = normalise_arxiv_id(aid) or aid

    from src.novelty.score import score_paper

    loop = asyncio.get_running_loop()
    try:
        profile = await loop.run_in_executor(None, lambda: score_paper(clean_id, allow_partial=True))
        return JSONResponse(sanitize_json_obj(profile))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "message": str(e)}, status_code=200)


# ── Landmark Citation Hubs ────────────────────────────────────────────────────


async def api_hubs(request: Request) -> JSONResponse:
    """Return top landmark citation hubs from graph.sqlite3."""
    graph_db = PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3"
    if not graph_db.exists():
        return JSONResponse({"hubs": [], "count": 0})

    conn = sqlite3.connect(graph_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT canonical_id, arxiv_id, title, year, citation_count
            FROM papers
            WHERE citation_count IS NOT NULL AND citation_count > 0
            ORDER BY citation_count DESC
            LIMIT 50
            """
        ).fetchall()
        hubs = [
            {
                "canonical_id": r["canonical_id"],
                "arxiv_id": r["arxiv_id"] or "",
                "title": r["title"] or "Untitled",
                "year": str(r["year"]) if r["year"] else "—",
                "citation_count": r["citation_count"] or 0,
            }
            for r in rows
        ]
        return JSONResponse({"hubs": hubs, "count": len(hubs)})
    finally:
        conn.close()


# ── Telemetry ─────────────────────────────────────────────────────────────────


async def api_telemetry(request: Request) -> JSONResponse:
    """Return live telemetry summary and recent Gemini calls."""
    summary = get_telemetry_summary()
    return JSONResponse(summary)


async def api_telemetry_reset(request: Request) -> JSONResponse:
    """Reset telemetry history."""
    reset_telemetry()
    return JSONResponse({"status": "reset", "telemetry": get_telemetry_summary()})


# ── App Construction ──────────────────────────────────────────────────────────


def create_app() -> Starlette:
    """Build the Starlette application with CORS and routes."""
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/api/health", api_health, methods=["GET"]),
        Route("/api/search", api_search, methods=["GET"]),
        Route("/api/recommend", api_recommend, methods=["POST"]),
        Route("/api/ask", api_ask, methods=["POST"]),
        Route("/api/ask/stream", api_ask_stream, methods=["GET"]),
        Route("/api/ingest", api_ingest, methods=["POST"]),
        Route("/api/arxiv/search", api_arxiv_search, methods=["GET"]),
        Route("/api/arxiv/import", api_arxiv_import, methods=["POST"]),
        Route("/api/novelty", api_novelty, methods=["GET"]),
        Route("/api/hubs", api_hubs, methods=["GET"]),
        Route("/api/telemetry", api_telemetry, methods=["GET"]),
        Route("/api/telemetry/reset", api_telemetry_reset, methods=["POST"]),
    ]

    web_dir = PROJECT_ROOT / "web"
    if web_dir.exists():
        routes.append(Mount("/", app=StaticFiles(directory=str(web_dir), html=True), name="web"))

    return Starlette(debug=False, routes=routes, middleware=middleware)


app = create_app()


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run API server with Uvicorn."""
    import uvicorn
    print(f"\n🚀 Starting ArXiv Scientific Discovery Server on http://{host}:{port}\n")
    uvicorn.run("src.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
