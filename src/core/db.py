"""Centralized SQLite database connection management, WAL handling, and pooling."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Global cache of open persistent connections per path
_CONNECTION_POOL: dict[str, sqlite3.Connection] = {}


def get_db_connection(
    path: str | Path,
    *,
    timeout: float = 15.0,
    wal_mode: bool = True,
    row_factory: Any = sqlite3.Row,
) -> sqlite3.Connection:
    """Open or reuse a pooled SQLite connection with busy timeout and WAL pragmas."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = str(p)

    conn = _CONNECTION_POOL.get(key)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            _CONNECTION_POOL.pop(key, None)

    conn = sqlite3.connect(str(p), timeout=timeout)
    if row_factory:
        conn.row_factory = row_factory

    if wal_mode:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA cache_size = -64000")
    _CONNECTION_POOL[key] = conn
    return conn


def apply_performance_pragmas(conn: sqlite3.Connection) -> None:
    """Apply memory mapping and cache size pragmas to any open connection."""
    try:
        conn.execute("PRAGMA mmap_size = 268435456")
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA busy_timeout = 5000")
    except Exception:
        pass


def checkpoint_db(path: str | Path, mode: str = "TRUNCATE") -> None:
    """Execute WAL checkpoint to truncate or passive-flush the journal."""
    p = Path(path).resolve()
    if not p.exists():
        return
    try:
        conn = sqlite3.connect(str(p), timeout=5.0)
        conn.execute(f"PRAGMA wal_checkpoint({mode})")
        conn.close()
    except Exception:
        pass


def close_all_connections() -> None:
    """Close all open pooled connections and flush checkpoints."""
    for key, conn in list(_CONNECTION_POOL.items()):
        try:
            conn.close()
        except Exception:
            pass
    _CONNECTION_POOL.clear()
