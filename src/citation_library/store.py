"""SQLite store for citation edges and paper metadata."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from settings import PROJECT_ROOT
except ImportError:
    from src.settings import PROJECT_ROOT

DEFAULT_DB = PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            canonical_id TEXT PRIMARY KEY,
            arxiv_id     TEXT,
            s2_id        TEXT,
            doi          TEXT,
            title        TEXT,
            year         INTEGER,
            citation_count INTEGER,
            abstract     TEXT,
            source       TEXT,
            updated_at   REAL
        );
        CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);

        CREATE TABLE IF NOT EXISTS edges (
            citer_id   TEXT NOT NULL,
            cited_id   TEXT NOT NULL,
            PRIMARY KEY (citer_id, cited_id)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_cited ON edges(cited_id);
        CREATE INDEX IF NOT EXISTS idx_edges_citer ON edges(citer_id);

        CREATE TABLE IF NOT EXISTS fetch_status (
            arxiv_id   TEXT PRIMARY KEY,
            canonical_id TEXT,
            status     TEXT NOT NULL,  -- ok | empty | error
            n_refs     INTEGER DEFAULT 0,
            source     TEXT,
            error      TEXT,
            updated_at REAL
        );
        """
    )
    conn.commit()


def upsert_paper(conn: sqlite3.Connection, rec: dict[str, Any]) -> None:
    cid = rec.get("canonical_id")
    if not cid:
        return
    # Empty strings must not overwrite existing good metadata via COALESCE
    title = (rec.get("title") or "").strip() or None
    arxiv = (rec.get("arxiv_id") or "").strip() or None
    doi = (rec.get("doi") or "").strip() or None
    s2 = (rec.get("s2_id") or "").strip() or None
    abstract = (rec.get("abstract") or "").strip() or None
    conn.execute(
        """
        INSERT INTO papers (
            canonical_id, arxiv_id, s2_id, doi, title, year,
            citation_count, abstract, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_id) DO UPDATE SET
            arxiv_id = COALESCE(excluded.arxiv_id, papers.arxiv_id),
            s2_id = COALESCE(excluded.s2_id, papers.s2_id),
            doi = COALESCE(excluded.doi, papers.doi),
            title = COALESCE(excluded.title, papers.title),
            year = COALESCE(excluded.year, papers.year),
            citation_count = COALESCE(excluded.citation_count, papers.citation_count),
            abstract = COALESCE(excluded.abstract, papers.abstract),
            source = COALESCE(excluded.source, papers.source),
            updated_at = excluded.updated_at
        """,
        (
            cid,
            arxiv,
            s2,
            doi,
            title,
            rec.get("year"),
            rec.get("citation_count"),
            abstract,
            rec.get("source") or rec.get("source_api"),
            time.time(),
        ),
    )


def replace_edges(
    conn: sqlite3.Connection, citer_id: str, cited_ids: Iterable[str]
) -> int:
    conn.execute("DELETE FROM edges WHERE citer_id = ?", (citer_id,))
    rows = [(citer_id, c) for c in cited_ids if c and c != citer_id]
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO edges (citer_id, cited_id) VALUES (?, ?)",
            rows,
        )
    return len(rows)


def set_fetch_status(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    status: str,
    canonical_id: str | None = None,
    n_refs: int = 0,
    source: str | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO fetch_status (
            arxiv_id, canonical_id, status, n_refs, source, error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            canonical_id = excluded.canonical_id,
            status = excluded.status,
            n_refs = excluded.n_refs,
            source = excluded.source,
            error = excluded.error,
            updated_at = excluded.updated_at
        """,
        (arxiv_id, canonical_id, status, n_refs, source, error, time.time()),
    )


def get_fetch_status(conn: sqlite3.Connection, arxiv_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM fetch_status WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    return dict(row) if row else None


def already_ok(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    row = get_fetch_status(conn, arxiv_id)
    return bool(row and row.get("status") == "ok")


def citation_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """cited_id → how many library citers."""
    return conn.execute(
        """
        SELECT
            e.cited_id AS canonical_id,
            COUNT(DISTINCT e.citer_id) AS library_cite_count,
            p.title,
            p.arxiv_id,
            p.year,
            p.doi,
            p.s2_id,
            p.citation_count AS global_citation_count
        FROM edges e
        LEFT JOIN papers p ON p.canonical_id = e.cited_id
        GROUP BY e.cited_id
        ORDER BY library_cite_count DESC, global_citation_count DESC
        """
    ).fetchall()


def example_citers(conn: sqlite3.Connection, cited_id: str, limit: int = 8) -> list[str]:
    cids = [cited_id]
    cid_str = str(cited_id).strip()
    if not cid_str.startswith("arxiv:") and not cid_str.startswith("s2:") and not cid_str.startswith("doi:"):
        cids.append(f"arxiv:{cid_str}")
    elif cid_str.startswith("arxiv:"):
        cids.append(cid_str[6:].strip())
    placeholders = ",".join(["?"] * len(cids))
    rows = conn.execute(
        f"""
        SELECT DISTINCT citer_id FROM edges
        WHERE cited_id IN ({placeholders})
        LIMIT ?
        """,
        (*cids, limit),
    ).fetchall()
    out = []
    for r in rows:
        cid = r["citer_id"] if isinstance(r, (dict, sqlite3.Row)) else r[0]
        # prefer arxiv part
        if cid.startswith("arxiv:"):
            out.append(cid.split(":", 1)[1])
        else:
            out.append(cid)
    return out


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_ok = conn.execute(
        "SELECT COUNT(*) FROM fetch_status WHERE status = 'ok'"
    ).fetchone()[0]
    n_empty = conn.execute(
        "SELECT COUNT(*) FROM fetch_status WHERE status = 'empty'"
    ).fetchone()[0]
    n_err = conn.execute(
        "SELECT COUNT(*) FROM fetch_status WHERE status = 'error'"
    ).fetchone()[0]
    return {
        "papers_cached": n_papers,
        "edges": n_edges,
        "fetch_ok": n_ok,
        "fetch_empty": n_empty,
        "fetch_error": n_err,
    }


def papers_missing_title(conn: sqlite3.Connection, *, limit: int = 200) -> list[sqlite3.Row]:
    """Cited works that rank highly but lack title (for batch enrich)."""
    return conn.execute(
        """
        SELECT e.cited_id AS canonical_id, p.s2_id, p.arxiv_id, p.title,
               COUNT(DISTINCT e.citer_id) AS library_cite_count
        FROM edges e
        LEFT JOIN papers p ON p.canonical_id = e.cited_id
        GROUP BY e.cited_id
        HAVING (p.title IS NULL OR TRIM(p.title) = '')
        ORDER BY library_cite_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
