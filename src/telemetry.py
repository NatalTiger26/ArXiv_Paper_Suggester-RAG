"""
Slim SQLite-backed Telemetry Engine for Gemini API calls.

Tracks:
- Total calls & call types (rag_ask, novelty_score, eval, etc.)
- Models used
- Token usage (prompt, candidates, total)
- Latency (ms, p50, p95)
- Estimated cost in $USD based on published Google Gemini pricing
- Cache hits / skips

Database is stored at data/telemetry.sqlite3.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "telemetry.sqlite3"

# Pricing per million tokens (USD)
# Flash tier (Gemini 1.5 Flash, 2.0 Flash, 2.5 Flash)
PRICE_FLASH_INPUT_PER_M = 0.075
PRICE_FLASH_OUTPUT_PER_M = 0.30

# Pro tier (Gemini 1.5 Pro, 2.5 Pro)
PRICE_PRO_INPUT_PER_M = 1.25
PRICE_PRO_OUTPUT_PER_M = 5.00


def calculate_cost_usd(model: str, prompt_tokens: int, candidate_tokens: int) -> float:
    """Calculate approximate USD cost for token usage."""
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(candidate_tokens or 0))

    model_lower = (model or "").lower()
    if "pro" in model_lower:
        input_rate = PRICE_PRO_INPUT_PER_M / 1_000_000.0
        output_rate = PRICE_PRO_OUTPUT_PER_M / 1_000_000.0
    else:
        # Default to Flash rates
        input_rate = PRICE_FLASH_INPUT_PER_M / 1_000_000.0
        output_rate = PRICE_FLASH_OUTPUT_PER_M / 1_000_000.0

    cost = (p_tok * input_rate) + (c_tok * output_rate)
    return round(cost, 6)


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open SQLite connection and ensure schema is initialized."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                call_type TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                candidate_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                cache_hit INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_events(timestamp)"
        )
    return conn


def record_call(
    call_type: str,
    model: str,
    prompt_tokens: int = 0,
    candidate_tokens: int = 0,
    latency_ms: float = 0.0,
    cache_hit: bool = False,
    db_path: Path | str | None = None,
) -> int:
    """Record an API event. Returns inserted row ID."""
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(candidate_tokens or 0))
    tot_tok = p_tok + c_tok
    cost = 0.0 if cache_hit else calculate_cost_usd(model, p_tok, c_tok)
    lat = max(0.0, float(latency_ms or 0.0))
    hit = 1 if cache_hit else 0
    now_ts = datetime.now(timezone.utc).isoformat()

    try:
        conn = get_connection(db_path)
        with conn:
            cur = conn.execute(
                """
                INSERT INTO telemetry_events (
                    timestamp, call_type, model, prompt_tokens, candidate_tokens,
                    total_tokens, cost_usd, latency_ms, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_ts,
                    str(call_type or "unknown"),
                    str(model or "unknown"),
                    p_tok,
                    c_tok,
                    tot_tok,
                    cost,
                    round(lat, 2),
                    hit,
                ),
            )
            return cur.lastrowid or 0
    except Exception as e:
        print(f"[telemetry] Warning: failed to record event: {e}")
        return 0


def get_telemetry_summary(db_path: Path | str | None = None) -> dict[str, Any]:
    """Calculate aggregated metrics and recent events."""
    try:
        conn = get_connection(db_path)
        with conn:
            cur = conn.execute(
                """
                SELECT 
                    COUNT(*) AS total_calls,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(candidate_tokens), 0) AS total_candidate_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                    COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                    COALESCE(SUM(cache_hit), 0) AS cache_hits
                FROM telemetry_events
                """
            )
            row = cur.fetchone()

            # Latency percentiles
            lats_cur = conn.execute(
                "SELECT latency_ms FROM telemetry_events ORDER BY latency_ms ASC"
            )
            latencies = [r[0] for r in lats_cur.fetchall()]

            # Recent 10 calls
            rec_cur = conn.execute(
                """
                SELECT timestamp, call_type, model, total_tokens, cost_usd, latency_ms, cache_hit
                FROM telemetry_events
                ORDER BY id DESC LIMIT 10
                """
            )
            recent = [dict(r) for r in rec_cur.fetchall()]

        total_calls = row["total_calls"] or 0
        hits = row["cache_hits"] or 0
        hit_rate = round((hits / total_calls * 100.0), 1) if total_calls > 0 else 0.0

        p50 = 0.0
        p95 = 0.0
        if latencies:
            p50 = round(latencies[int(len(latencies) * 0.50)], 1)
            p95 = round(latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)], 1)

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": row["total_prompt_tokens"],
            "total_candidate_tokens": row["total_candidate_tokens"],
            "total_tokens": row["total_tokens"],
            "total_cost_usd": round(row["total_cost_usd"], 5),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "cache_hits": hits,
            "cache_hit_rate_pct": hit_rate,
            "recent_calls": recent,
        }
    except Exception as e:
        return {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_candidate_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "cache_hits": 0,
            "cache_hit_rate_pct": 0.0,
            "recent_calls": [],
            "error": str(e),
        }


def reset_telemetry(db_path: Path | str | None = None) -> None:
    """Clear all recorded telemetry rows and vacuum database."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM telemetry_events")
    conn.isolation_level = None
    try:
        conn.execute("VACUUM")
    except Exception:
        pass
    finally:
        conn.close()
