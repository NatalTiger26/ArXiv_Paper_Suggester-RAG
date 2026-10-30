"""
High-Performance SQLite FTS5 Paper Catalog Bridge.

Provides:
- $O(1)$ indexed paper lookup by arXiv ID.
- Sub-millisecond BM25 full-text keyword search across titles and abstracts (via SQLite FTS5).
- Method / acronym entity extraction from natural language queries for RAG routing.
- Instant $O(1)$ citation updates without re-serializing 7.6MB JSON.
- Seamless bidirectional sync with data/corpus.json for 100% backward compatibility.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "catalog.sqlite3"
DEFAULT_CORPUS_JSON = PROJECT_ROOT / "data" / "corpus.json"


_CATALOG_INITIALIZED: set[str] = set()


def clean_arxiv_id(aid: str | None) -> str:
    """Return normalized arXiv ID delegating to canonical src.core.ids."""
    if not aid:
        return ""
    from src.core.ids import normalise_arxiv_id
    res = normalise_arxiv_id(aid)
    return res if res else ""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Connect to catalog database with memory mapping and one-time schema initialization."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    from src.core.db import apply_performance_pragmas
    apply_performance_pragmas(conn)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    p_str = str(path.resolve())
    if p_str not in _CATALOG_INITIALIZED:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    year INTEGER,
                    citation_count INTEGER,
                    categories TEXT,
                    doi TEXT,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_cites ON papers(citation_count)"
            )

            # SQLite FTS5 external content table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    arxiv_id,
                    title,
                    abstract,
                    categories,
                    content='papers',
                    content_rowid='rowid'
                )
                """
            )

            # Triggers to keep FTS index synchronized with papers table
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract, categories)
                    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract, new.categories);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract, categories)
                    VALUES('delete', old.rowid, old.arxiv_id, old.title, old.abstract, old.categories);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract, categories)
                    VALUES('delete', old.rowid, old.arxiv_id, old.title, old.abstract, old.categories);
                    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract, categories)
                    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract, new.categories);
                END
                """
            )
        _CATALOG_INITIALIZED.add(p_str)

    return conn


def get_paper(arxiv_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """O(1) paper lookup by arXiv ID."""
    aid = clean_arxiv_id(arxiv_id)
    if not aid:
        return None

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        cur = conn.execute(
            """
            SELECT arxiv_id, title, abstract, year, citation_count, categories, doi, updated_at
            FROM papers
            WHERE arxiv_id = ?
            """,
            (aid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cats = [c.strip() for c in (row["categories"] or "").split(",") if c.strip()]
        return {
            "arxiv_id": row["arxiv_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "year": row["year"],
            "citation_count": row["citation_count"],
            "categories": cats,
            "doi": row["doi"],
            "updated_at": row["updated_at"],
        }
    finally:
        if close_conn:
            conn.close()


def search_catalog(
    query: str,
    limit: int = 20,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Sub-millisecond BM25 keyword search across titles and abstracts."""
    q = re.sub(r'[^\w\s\-]', ' ', query).strip()
    if not q:
        return []

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        # Match FTS terms
        terms = [t for t in q.split() if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"*' for t in terms)

        cur = conn.execute(
            """
            SELECT p.arxiv_id, p.title, p.abstract, p.year, p.citation_count, p.categories,
                   bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) AS rank_score
            FROM papers_fts f
            JOIN papers p ON p.rowid = f.rowid
            WHERE papers_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        )
        results = []
        for r in cur.fetchall():
            cats = [c.strip() for c in (r["categories"] or "").split(",") if c.strip()]
            results.append({
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "abstract": r["abstract"],
                "year": r["year"],
                "citation_count": r["citation_count"],
                "categories": cats,
                "score": float(r["rank_score"]),
            })
        return results
    except Exception as e:
        return []
    finally:
        if close_conn:
            conn.close()


def resolve_entities_in_query(
    query: str,
    limit: int = 3,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """
    Extract method, acronym, or paper title entities (e.g. 'Grad-CAM', 'NTK', 'SHAP', 'LoRA')
    from a question and match them against papers_fts to return candidate arXiv IDs.
    """
    # Exclude common query words
    stop = {
        "what", "how", "why", "when", "where", "who", "which", "is", "are", "the",
        "of", "in", "and", "or", "to", "for", "with", "a", "an", "paper", "papers",
        "method", "approach", "model", "explain", "main", "contribution", "summary",
        "does", "do", "used", "can", "using", "work", "novel", "result", "results"
    }
    tokens = [w for w in re.findall(r"\b[A-Za-z0-9_.\-]+\b", query) if len(w) > 2]
    keywords = [t for t in tokens if t.lower() not in stop]
    if not keywords:
        return []

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        # Search exact phrases or terms in titles specifically
        exact_terms = " OR ".join(f'title:"{kw}"' for kw in keywords[:4])
        cur = conn.execute(
            """
            SELECT p.arxiv_id, bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) as rank
            FROM papers_fts f
            JOIN papers p ON p.rowid = f.rowid
            WHERE papers_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (exact_terms, limit),
        )
        return [r["arxiv_id"] for r in cur.fetchall() if r["arxiv_id"]]
    except Exception:
        return []
    finally:
        if close_conn:
            conn.close()


def update_citation(arxiv_id: str, count: int, conn: sqlite3.Connection | None = None) -> None:
    """Instant O(1) update of citation count."""
    aid = clean_arxiv_id(arxiv_id)
    if not aid:
        return

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        with conn:
            conn.execute(
                "UPDATE papers SET citation_count = ?, updated_at = ? WHERE arxiv_id = ?",
                (int(count), time.time(), aid),
            )
    finally:
        if close_conn:
            conn.close()


def backfill_catalog_citations(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Backfill missing citation counts from local cache and graph databases."""
    from scripts.sync_citation_counts import run_sync
    return run_sync()


def upsert_paper_record(
    conn: sqlite3.Connection,
    arxiv_id: str,
    title: str,
    abstract: str,
    year: int | None = None,
    citation_count: int | None = None,
    categories: str | list[str] | None = None,
    doi: str | None = None,
) -> None:
    """Insert or update a single paper record in SQLite catalog."""
    aid = clean_arxiv_id(arxiv_id)
    if not aid:
        return
    now = time.time()
    cats_str = ",".join(categories) if isinstance(categories, list) else str(categories or "")
    with conn:
        conn.execute(
            """
            INSERT INTO papers (arxiv_id, title, abstract, year, citation_count, categories, doi, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                year = COALESCE(excluded.year, papers.year),
                citation_count = COALESCE(excluded.citation_count, papers.citation_count),
                categories = excluded.categories,
                doi = excluded.doi,
                updated_at = excluded.updated_at
            """,
            (aid, title, abstract, year, citation_count, cats_str, doi or "", now),
        )


def sync_from_corpus_json(
    corpus_path: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Populate or update SQLite catalog from data/corpus.json."""
    c_path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS_JSON
    if not c_path.exists():
        return 0

    try:
        data = json.loads(c_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[catalog] Error reading {c_path}: {e}")
        return 0

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    count = 0
    now = time.time()
    try:
        with conn:
            for item in data:
                aid = clean_arxiv_id(item.get("arxiv_id") or item.get("id"))
                if not aid:
                    continue
                title = str(item.get("title") or "").strip()
                abstract = str(item.get("abstract") or "").strip()
                year = None
                try:
                    pub = item.get("published_date") or item.get("year") or ""
                    year = int(str(pub)[:4]) if str(pub)[:4].isdigit() else None
                except Exception:
                    pass

                c_cnt = None
                try:
                    raw_c = item.get("citation_count")
                    c_cnt = int(raw_c) if raw_c is not None and str(raw_c).isdigit() else None
                except Exception:
                    pass

                cats = item.get("categories") or []
                cats_str = ",".join(cats) if isinstance(cats, list) else str(cats)
                doi = str(item.get("doi") or "")

                conn.execute(
                    """
                    INSERT INTO papers (arxiv_id, title, abstract, year, citation_count, categories, doi, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        year = COALESCE(excluded.year, papers.year),
                        citation_count = COALESCE(excluded.citation_count, papers.citation_count),
                        categories = excluded.categories,
                        doi = excluded.doi,
                        updated_at = excluded.updated_at
                    """,
                    (aid, title, abstract, year, c_cnt, cats_str, doi, now),
                )
                count += 1
        return count
    finally:
        if close_conn:
            conn.close()


def catalog_stats(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Return summary statistics of the catalog database."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM papers")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0")
        with_cites = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM papers_fts")
        fts_cnt = cur.fetchone()[0]
        return {
            "total_papers": total,
            "papers_with_citations": with_cites,
            "fts_indexed": fts_cnt,
            "fts_enabled": True,
        }
    finally:
        if close_conn:
            conn.close()


def sync_to_corpus_json(
    corpus_path: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Export catalog to data/corpus.json atomically maintaining 100% backward compatibility."""
    c_path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS_JSON

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    try:
        cur = conn.execute(
            """
            SELECT arxiv_id, title, abstract, year, citation_count, categories, doi
            FROM papers
            ORDER BY arxiv_id ASC
            """
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            cats = [c.strip() for c in (r["categories"] or "").split(",") if c.strip()]
            out.append({
                "arxiv_id": r["arxiv_id"],
                "title": r["title"] or "",
                "abstract": r["abstract"] or "",
                "published_date": f"{r['year']}-01-01" if r["year"] else "",
                "citation_count": r["citation_count"],
                "categories": cats,
                "doi": r["doi"] or "",
            })

        tmp_path = c_path.with_name(f"{c_path.name}.tmp_{int(time.time())}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, c_path)
        return len(out)
    finally:
        if close_conn:
            conn.close()


def get_catalog_stats(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Summary statistics for the library catalog and RAG index."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    md_dir = PROJECT_ROOT / "data" / "markdown"
    full_text_count = len(list(md_dir.glob("*.md"))) if md_dir.exists() else 0

    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_papers,
                MIN(year) as min_year,
                MAX(year) as max_year,
                AVG(CASE WHEN citation_count > 0 THEN citation_count ELSE NULL END) as avg_cites,
                MAX(citation_count) as max_cites
            FROM papers
            """
        ).fetchone()

        total = row["total_papers"] if row else 0
        min_y = row["min_year"] if row and row["min_year"] else 2017
        max_y = row["max_year"] if row and row["max_year"] else 2026
        avg_c = round(float(row["avg_cites"] or 0), 1) if row else 0.0

        # Top categories
        cat_rows = conn.execute(
            "SELECT categories FROM papers WHERE categories IS NOT NULL AND length(categories) > 0"
        ).fetchall()
        from collections import Counter
        cat_counts = Counter()
        for cr in cat_rows:
            for c in (cr["categories"] or "").split(","):
                c_clean = c.strip()
                if c_clean:
                    cat_counts[c_clean] += 1

        return {
            "total_papers": total,
            "full_text_count": full_text_count,
            "min_year": min_y,
            "max_year": max_y,
            "avg_citations": avg_c,
            "top_categories": [c for c, _ in cat_counts.most_common(12)],
        }
    finally:
        if close_conn:
            conn.close()


def browse_catalog(
    limit: int = 50,
    offset: int = 0,
    query: str | None = None,
    category: str | None = None,
    min_year: int | None = None,
    min_citations: int | None = None,
    full_text_only: bool = False,
    sort_by: str = "citations_desc",
    conn: sqlite3.Connection | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Paginated catalog browser with multi-field filtering and full-text detection.
    Returns (papers, total_matching_count).
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    md_dir = PROJECT_ROOT / "data" / "markdown"
    q_clean = re.sub(r'[^\w\s\-]', ' ', query or "").strip()

    try:
        where_clauses = []
        params: list[Any] = []

        if category and category.strip():
            where_clauses.append("p.categories LIKE ?")
            params.append(f"%{category.strip()}%")

        if min_year is not None and min_year > 1900:
            where_clauses.append("p.year >= ?")
            params.append(int(min_year))

        if min_citations is not None and min_citations > 0:
            where_clauses.append("p.citation_count >= ?")
            params.append(int(min_citations))

        # Query handling (FTS vs standard)
        use_fts = bool(q_clean)
        terms = [t for t in q_clean.split() if len(t) > 1] if use_fts else []

        if use_fts and terms:
            fts_match = " OR ".join(f'"{t}"*' for t in terms)
            where_clauses.append("papers_fts MATCH ?")
            params.append(fts_match)

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            # Count total
            count_sql = f"""
                SELECT COUNT(*) as cnt
                FROM papers_fts f
                JOIN papers p ON p.rowid = f.rowid
                {where_str}
            """
            total_matching = conn.execute(count_sql, params).fetchone()["cnt"]

            # Order by
            if sort_by == "citations_desc":
                order_clause = "ORDER BY p.citation_count DESC, bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) ASC"
            elif sort_by == "year_desc":
                order_clause = "ORDER BY p.year DESC, bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) ASC"
            elif sort_by == "title_asc":
                order_clause = "ORDER BY p.title ASC"
            else:
                order_clause = "ORDER BY bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) ASC, p.citation_count DESC"

            data_sql = f"""
                SELECT p.arxiv_id, p.title, p.abstract, p.year, p.citation_count, p.categories, p.doi,
                       bm25(papers_fts, 5.0, 10.0, 2.0, 1.0) as rank_score
                FROM papers_fts f
                JOIN papers p ON p.rowid = f.rowid
                {where_str}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, params + [limit, offset]).fetchall()
        else:
            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            count_sql = f"SELECT COUNT(*) as cnt FROM papers p {where_str}"
            total_matching = conn.execute(count_sql, params).fetchone()["cnt"]

            if sort_by == "year_desc":
                order_clause = "ORDER BY p.year DESC NULLS LAST, p.citation_count DESC"
            elif sort_by == "title_asc":
                order_clause = "ORDER BY p.title ASC"
            else:
                order_clause = "ORDER BY p.citation_count DESC NULLS LAST, p.year DESC NULLS LAST"

            data_sql = f"""
                SELECT p.arxiv_id, p.title, p.abstract, p.year, p.citation_count, p.categories, p.doi
                FROM papers p
                {where_str}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

        papers = []
        for r in rows:
            aid = r["arxiv_id"]
            cats = [c.strip() for c in (r["categories"] or "").split(",") if c.strip()]
            has_ft = (md_dir / f"{aid}.md").exists() if md_dir.exists() else False

            if full_text_only and not has_ft:
                continue

            papers.append({
                "arxiv_id": aid,
                "title": r["title"] or aid,
                "abstract": r["abstract"] or "",
                "year": r["year"],
                "citation_count": r["citation_count"] or 0,
                "categories": cats,
                "doi": r["doi"],
                "has_full_text": has_ft,
            })

        return papers, total_matching
    finally:
        if close_conn:
            conn.close()
