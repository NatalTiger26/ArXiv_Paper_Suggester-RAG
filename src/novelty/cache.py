"""SQLite store: paper abstracts+embeddings, reference edges, score profiles.

Does NOT touch Chroma recommend/RAG indexes.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from settings import DATA_DIR, PROJECT_ROOT
except ImportError:
    from src.settings import DATA_DIR, PROJECT_ROOT

NOVELTY_DIR = DATA_DIR / "novelty"
DB_PATH = NOVELTY_DIR / "novelty_cache.sqlite3"


def _ensure_dir() -> None:
    NOVELTY_DIR.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    _ensure_dir()
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            canonical_id TEXT PRIMARY KEY,
            arxiv_id TEXT,
            s2_id TEXT,
            openalex_id TEXT,
            doi TEXT,
            title TEXT,
            abstract TEXT,
            year INTEGER,
            categories TEXT,
            source_api TEXT,
            fetch_date REAL,
            times_cited_by_scored INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            canonical_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            embedded_at REAL NOT NULL,
            PRIMARY KEY (canonical_id, model, dim),
            FOREIGN KEY (canonical_id) REFERENCES papers(canonical_id)
        );

        CREATE TABLE IF NOT EXISTS ref_edges (
            focal_id TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            position INTEGER,
            source_api TEXT,
            fetch_date REAL,
            PRIMARY KEY (focal_id, ref_id)
        );

        CREATE TABLE IF NOT EXISTS scores (
            focal_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            profile_json TEXT NOT NULL,
            cohort_note TEXT,
            computed_at REAL NOT NULL,
            PRIMARY KEY (focal_id, model, dim)
        );

        CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
        CREATE INDEX IF NOT EXISTS idx_edges_focal ON ref_edges(focal_id);
        CREATE INDEX IF NOT EXISTS idx_edges_ref ON ref_edges(ref_id);
        """
    )
    conn.commit()


def pack_vector(vec: list[float] | Any) -> bytes:
    import numpy as np

    arr = np.asarray(vec, dtype="float32").ravel()
    return arr.tobytes()


def unpack_vector(blob: bytes) -> Any:
    import numpy as np

    return np.frombuffer(blob, dtype="float32").copy()


def upsert_paper(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    title: str | None = None,
    abstract: str | None = None,
    year: int | None = None,
    categories: list[str] | str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    doi: str | None = None,
    source_api: str | None = None,
) -> None:
    cats = categories
    if isinstance(cats, list):
        cats = json.dumps(cats)
    conn.execute(
        """
        INSERT INTO papers (
            canonical_id, arxiv_id, s2_id, openalex_id, doi,
            title, abstract, year, categories, source_api, fetch_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_id) DO UPDATE SET
            arxiv_id=COALESCE(excluded.arxiv_id, papers.arxiv_id),
            s2_id=COALESCE(excluded.s2_id, papers.s2_id),
            openalex_id=COALESCE(excluded.openalex_id, papers.openalex_id),
            doi=COALESCE(excluded.doi, papers.doi),
            title=COALESCE(excluded.title, papers.title),
            abstract=COALESCE(excluded.abstract, papers.abstract),
            year=COALESCE(excluded.year, papers.year),
            categories=COALESCE(excluded.categories, papers.categories),
            source_api=COALESCE(excluded.source_api, papers.source_api),
            fetch_date=excluded.fetch_date
        """,
        (
            canonical_id,
            arxiv_id,
            s2_id,
            openalex_id,
            doi,
            title,
            abstract,
            year,
            cats,
            source_api,
            time.time(),
        ),
    )
    conn.commit()


def get_paper(conn: sqlite3.Connection, canonical_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM papers WHERE canonical_id=?", (canonical_id,)
    ).fetchone()
    return dict(row) if row else None


def put_embedding(
    conn: sqlite3.Connection,
    canonical_id: str,
    model: str,
    dim: int,
    vector,
) -> None:
    conn.execute(
        """
        INSERT INTO embeddings (canonical_id, model, dim, vector, embedded_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(canonical_id, model, dim) DO UPDATE SET
            vector=excluded.vector,
            embedded_at=excluded.embedded_at
        """,
        (canonical_id, model, int(dim), pack_vector(vector), time.time()),
    )
    conn.commit()


def get_embedding(
    conn: sqlite3.Connection, canonical_id: str, model: str, dim: int
):
    row = conn.execute(
        """
        SELECT vector FROM embeddings
        WHERE canonical_id=? AND model=? AND dim=?
        """,
        (canonical_id, model, int(dim)),
    ).fetchone()
    if not row:
        return None
    return unpack_vector(row["vector"])


def set_ref_edges(
    conn: sqlite3.Connection,
    focal_id: str,
    ref_ids: list[str],
    source_api: str,
) -> None:
    conn.execute("DELETE FROM ref_edges WHERE focal_id=?", (focal_id,))
    now = time.time()
    conn.executemany(
        """
        INSERT INTO ref_edges (focal_id, ref_id, position, source_api, fetch_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(focal_id, rid, i, source_api, now) for i, rid in enumerate(ref_ids)],
    )
    # bump times_cited_by_scored for refs
    for rid in ref_ids:
        conn.execute(
            """
            UPDATE papers SET times_cited_by_scored = times_cited_by_scored + 1
            WHERE canonical_id=?
            """,
            (rid,),
        )
    conn.commit()


def get_ref_ids(conn: sqlite3.Connection, focal_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT ref_id FROM ref_edges
        WHERE focal_id=?
        ORDER BY position ASC
        """,
        (focal_id,),
    ).fetchall()
    return [r["ref_id"] for r in rows]


def save_score(
    conn: sqlite3.Connection,
    focal_id: str,
    model: str,
    dim: int,
    profile: dict[str, Any],
    cohort_note: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO scores (focal_id, model, dim, profile_json, cohort_note, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(focal_id, model, dim) DO UPDATE SET
            profile_json=excluded.profile_json,
            cohort_note=excluded.cohort_note,
            computed_at=excluded.computed_at
        """,
        (
            focal_id,
            model,
            int(dim),
            json.dumps(profile, ensure_ascii=False),
            cohort_note,
            time.time(),
        ),
    )
    conn.commit()


def get_score(
    conn: sqlite3.Connection, focal_id: str, model: str, dim: int
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT profile_json, cohort_note, computed_at FROM scores
        WHERE focal_id=? AND model=? AND dim=?
        """,
        (focal_id, model, int(dim)),
    ).fetchone()
    if not row:
        return None
    return {
        "profile": json.loads(row["profile_json"]),
        "cohort_note": row["cohort_note"],
        "computed_at": row["computed_at"],
    }


def all_focal_ids_with_refs(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT focal_id FROM ref_edges ORDER BY focal_id"
    ).fetchall()
    return [r["focal_id"] for r in rows]


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    def cnt(table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])

    return {
        "papers": cnt("papers"),
        "embeddings": cnt("embeddings"),
        "ref_edges": cnt("ref_edges"),
        "scores": cnt("scores"),
    }
