"""Null calibration with matched n, overlap exclusion, and minimum pool size."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .cache import connect, get_embedding, get_paper, get_ref_ids
from .metrics import compute_profile

MIN_NULL_POOL = 10
DEFAULT_N_NULL = 30


def _focal_meta(conn, focal_id: str) -> dict[str, Any]:
    p = get_paper(conn, focal_id) or {}
    cats = p.get("categories") or "[]"
    if isinstance(cats, str):
        try:
            import json

            cats = json.loads(cats)
        except Exception:
            cats = [cats] if cats else []
    return {
        "year": p.get("year"),
        "categories": set(str(c) for c in (cats or [])),
    }


def _cohort_score(meta_f: dict, meta_o: dict) -> int:
    """Higher = better null donor (year proximity + category overlap)."""
    score = 0
    yf, yo = meta_f.get("year"), meta_o.get("year")
    if yf and yo:
        score += max(0, 5 - abs(int(yf) - int(yo)))
    inter = meta_f.get("categories", set()) & meta_o.get("categories", set())
    score += 2 * len(inter)
    return score


def shuffled_reference_percentiles(
    focal_id: str,
    x: np.ndarray,
    model: str,
    dim: int,
    profile: dict[str, Any],
    *,
    n_null: int = DEFAULT_N_NULL,
    seed: int = 0,
    center: np.ndarray | None = None,
    min_pool: int = MIN_NULL_POOL,
) -> dict[str, Any]:
    """
    Score x against other focals' reference *sets*.

    - Excludes any ref id that appears in the focal's own ref list (overlap).
    - Subsamples each null ref set to the same n as the focal profile.
    - Prefers same year/category donors when available.
    - Refuses percentiles if usable null pool < min_pool.
    """
    conn = connect()
    focal_refs = set(get_ref_ids(conn, focal_id))
    n_target = int(profile.get("n_refs_used") or 0)
    if n_target < 2:
        return {"ok": False, "reason": "focal_too_few_refs", "pool": 0}

    meta_f = _focal_meta(conn, focal_id)
    candidates: list[tuple[int, str]] = []
    for other in _all_focals(conn):
        if other == focal_id:
            continue
        refs = get_ref_ids(conn, other)
        # drop overlapping ids with focal bibliography
        refs = [r for r in refs if r not in focal_refs]
        if len(refs) < n_target:
            continue
        meta_o = _focal_meta(conn, other)
        candidates.append((_cohort_score(meta_f, meta_o), other))

    candidates.sort(key=lambda t: -t[0])
    # Stratified null candidate selection: prioritize donors with category match
    stratified_candidates = [cid for score, cid in candidates if score >= 2]
    is_stratified = len(stratified_candidates) >= min_pool
    if is_stratified:
        pool_ids = stratified_candidates
    else:
        pool_ids = [cid for _, cid in candidates]
    if len(pool_ids) < min_pool:
        return {
            "ok": False,
            "reason": "null_pool_too_small",
            "pool": len(pool_ids),
            "min_pool": min_pool,
            "message": (
                f"Need ≥{min_pool} other scored focals with enough non-overlapping "
                f"refs, have {len(pool_ids)}. Score more papers, then re-run null."
            ),
        }

    rng = random.Random(seed)
    # take up to n_null from best cohort matches, shuffle among top band
    top = pool_ids[: max(n_null * 2, min_pool)]
    if len(top) > n_null:
        sample = rng.sample(top, n_null)
    else:
        sample = top

    keys = ["m", "a", "R2_plus", "Delta", "D_median"]
    null_vals: dict[str, list[float]] = {k: [] for k in keys}

    for other in sample:
        refs = [r for r in get_ref_ids(conn, other) if r not in focal_refs]
        if len(refs) < n_target:
            continue
        # deterministic subsample to matched n
        rng_o = random.Random(f"{seed}:{other}:{n_target}")
        pick = refs if len(refs) == n_target else rng_o.sample(refs, n_target)
        cols = []
        used = []
        for rid in pick:
            v = get_embedding(conn, rid, model, dim)
            if v is not None:
                arr = np.asarray(v, dtype=np.float64)
                if arr.shape[0] == x.shape[0]:
                    cols.append(arr)
                    used.append(rid)
        if len(cols) < n_target:
            continue
        Y = np.column_stack(cols)
        p = compute_profile(x, Y, ref_ids=used, center=center)
        for k in keys:
            val = p.get(k)
            if val is not None and val == val:
                null_vals[k].append(float(val))

    usable = max((len(null_vals[k]) for k in keys), default=0)
    if usable < min_pool:
        return {
            "ok": False,
            "reason": "null_usable_too_small",
            "pool": len(pool_ids),
            "usable": usable,
            "min_pool": min_pool,
        }

    out: dict[str, Any] = {
        "ok": True,
        "pool_candidates": len(pool_ids),
        "stratified_category": is_stratified,
        "n_null_requested": n_null,
        "n_null_used": usable,
        "matched_n": n_target,
        "percentiles": {},
    }
    for k in keys:
        arr = null_vals[k]
        obs = profile.get(k)
        if obs is None or obs != obs or len(arr) < min_pool:
            out["percentiles"][k] = None
            continue
        a = np.asarray(arr, dtype=np.float64)
        out["percentiles"][k] = float(np.mean(a < float(obs)))
        out[f"null_mean_{k}"] = float(np.mean(a))
        out[f"null_std_{k}"] = float(np.std(a))
    return out


def _all_focals(conn) -> list[str]:
    rows = conn.execute("SELECT DISTINCT focal_id FROM ref_edges").fetchall()
    return [r["focal_id"] if hasattr(r, "keys") else r[0] for r in rows]


def covariate_notes(n_refs: int, year: int | None, abstract_len: int) -> dict[str, Any]:
    return {
        "n_refs": n_refs,
        "log_n_refs": float(np.log1p(n_refs)),
        "year": year,
        "abstract_len": abstract_len,
    }


def estimate_cohort_mean(
    conn,
    model: str,
    dim: int,
    *,
    max_vectors: int = 2000,
) -> np.ndarray | None:
    """Mean of cached embedding vectors for (model, dim): used for centering."""
    rows = conn.execute(
        """
        SELECT vector FROM embeddings
        WHERE model=? AND dim=?
        LIMIT ?
        """,
        (model, int(dim), int(max_vectors)),
    ).fetchall()
    if len(rows) < 20:
        return None
    mats = [np.frombuffer(r["vector"], dtype=np.float32).astype(np.float64) for r in rows]
    # equalize length
    d0 = mats[0].shape[0]
    mats = [m for m in mats if m.shape[0] == d0]
    if len(mats) < 20:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0)
