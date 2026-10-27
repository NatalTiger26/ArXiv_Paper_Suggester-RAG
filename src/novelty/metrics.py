"""Reference-conditional embedding geometry metrics.

Framing: geometry vs cited works in a fixed embedding space — not idea-level novelty.

Pipeline (per score call):
  1) Unit-normalize inputs
  2) Optional cohort centering (subtract mean, re-unit)
  3) Compute m, a, c, dispersion, cone R²_+, Δ, attribution
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    if n < 1e-12 or not np.isfinite(n):
        return np.zeros_like(v)
    return v / n


def prepare_matrix(
    x: np.ndarray,
    Y: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build unit columns; optionally subtract the same cohort mean from x and each y,
    then re-normalize (approximate whitening-free centering).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[0] != x.shape[0] and Y.shape[1] == x.shape[0]:
        Y = Y.T

    if center is not None:
        c = np.asarray(center, dtype=np.float64).ravel()
        if c.shape[0] == x.shape[0] and np.all(np.isfinite(c)):
            x = x - c
            Y = Y - c.reshape(-1, 1)

    x = _unit(x)
    cols = []
    for j in range(Y.shape[1]):
        u = _unit(Y[:, j])
        if float(np.linalg.norm(u)) > 0.5:
            cols.append(u)
    Y_out = np.column_stack(cols) if cols else np.zeros((x.shape[0], 0), dtype=np.float64)
    return x, Y_out


def pairwise_dispersion(
    Y: np.ndarray, quantiles: tuple[float, ...] = (0.5, 0.9)
) -> dict[str, float]:
    """Dispersion among reference vectors (pairwise 1 - cosine)."""
    n = Y.shape[1]
    if n < 2:
        out = {f"D_q{int(q * 100)}": float("nan") for q in quantiles}
        out.update({"D_median": float("nan"), "n_pairs": 0})
        return out
    dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1.0 - float(np.dot(Y[:, i], Y[:, j])))
    arr = np.asarray(dists, dtype=np.float64)
    out: dict[str, float] = {
        "D_median": float(np.median(arr)),
        "n_pairs": float(arr.size),
    }
    for q in quantiles:
        out[f"D_q{int(q * 100)}"] = float(np.quantile(arr, q))
    return out


def nnls_cone_r2(x: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Non-negative least squares projection onto cone {Y λ : λ ≥ 0}.
    For unit x: R2_+ = ||P_K x||^2  (Moreau: residual energy = 1 - R2).
    """
    n = Y.shape[1]
    if n == 0 or float(np.linalg.norm(x)) < 1e-12:
        return 0.0, np.zeros(n, dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(Y))):
        return 0.0, np.zeros(n, dtype=np.float64)
    try:
        from scipy.optimize import nnls

        lam, _ = nnls(Y, x)
    except Exception:
        try:
            lam, *_ = np.linalg.lstsq(Y, x, rcond=None)
            lam = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
        except Exception:
            return 0.0, np.zeros(n, dtype=np.float64)
    lam = np.nan_to_num(np.asarray(lam, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    proj = Y @ lam
    r2 = float(np.dot(proj, proj))
    if not np.isfinite(r2):
        r2 = 0.0
    return max(0.0, min(1.0, r2)), lam


def effective_dimension(vectors: np.ndarray) -> float | None:
    """Participation ratio d_eff = (tr Σ)^2 / tr(Σ²) of row-stacked vectors."""
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        return None
    X = X - X.mean(axis=0, keepdims=True)
    n, d = X.shape
    try:
        if n <= d:
            G = (X @ X.T) / max(n - 1, 1)
            ev = np.linalg.eigvalsh(G)
        else:
            C = (X.T @ X) / max(n - 1, 1)
            ev = np.linalg.eigvalsh(C)
        ev = np.clip(ev, 0.0, None)
        s1 = float(ev.sum())
        s2 = float((ev ** 2).sum())
        if s2 < 1e-18 or np.isnan(s1) or np.isnan(s2):
            return None
        res = float((s1 * s1) / s2)
        return None if (np.isnan(res) or np.isinf(res)) else res
    except Exception:
        return None


def chance_r2_ols_mean(n_refs: int, d_eff: float) -> float | None:
    """Isotropic OLS mean R² ≈ n / d_eff (upper bound motivation for cone)."""
    if not d_eff or d_eff != d_eff or d_eff <= 0 or n_refs <= 0:
        return None
    return float(min(1.0, n_refs / d_eff))


def compute_profile(
    x: np.ndarray,
    Y: np.ndarray,
    *,
    ref_ids: list[str] | None = None,
    center: np.ndarray | None = None,
    top_k_attr: int = 8,
) -> dict[str, Any]:
    """
    Core profile after optional centering:

      m      = max_i cos(x, y_i)
      a      = cos(x, y_bar/||y_bar||)
      c      = ||y_bar||²
      D_*    = pairwise dispersion among refs
      R2_+   = cone explained energy (NNLS, λ≥0)
      Delta  = R2_+ - max(m, 0)²
      attr   = normalized λ
    """
    x_u, Y_u = prepare_matrix(x, Y, center=center)
    d, n = int(Y_u.shape[0]), int(Y_u.shape[1])
    out: dict[str, Any] = {
        "n_refs_used": n,
        "dim": d,
        "centered": center is not None,
    }
    if n == 0:
        out.update(
            {
                "m": None,
                "a": None,
                "c": None,
                "R2_plus": None,
                "m_plus_sq": None,
                "Delta": None,
                "attribution": [],
                "dispersion": {},
                "error": "no_reference_vectors",
            }
        )
        return out

    k = Y_u.T @ x_u
    m = float(np.max(k)) if k.size > 0 else 0.0
    if not np.isfinite(m):
        m = 0.0
    m_idx = int(np.argmax(k)) if k.size > 0 else 0
    m_plus = max(m, 0.0)
    m_plus_sq = m_plus ** 2

    ybar = Y_u.mean(axis=1) if Y_u.shape[1] > 0 else np.zeros_like(x_u)
    c = float(np.dot(ybar, ybar))
    if not np.isfinite(c):
        c = 0.0
    a = float(np.dot(x_u, ybar) / np.sqrt(c)) if c > 1e-12 else 0.0
    if not np.isfinite(a):
        a = 0.0

    disp = pairwise_dispersion(Y_u)
    r2, lam = nnls_cone_r2(x_u, Y_u)
    if not np.isfinite(r2):
        r2 = 0.0
    delta = float(r2 - m_plus_sq)
    if not np.isfinite(delta) or (-1e-9 < delta < 0):
        delta = 0.0

    s = float(lam.sum())
    atr = (lam / s) if s > 1e-12 else np.zeros_like(lam)

    order = np.argsort(-atr)
    attribution = []
    for j in order[:top_k_attr]:
        if atr[j] <= 1e-8:
            continue
        rid = ref_ids[j] if ref_ids and j < len(ref_ids) else str(int(j))
        attribution.append(
            {"ref_id": rid, "weight": float(atr[j]), "cos": float(k[j])}
        )

    # local d_eff on reference vectors (rows)
    d_eff = effective_dimension(Y_u.T)
    out.update(
        {
            "m": m,
            "m_ref_id": (
                ref_ids[m_idx] if ref_ids and m_idx < len(ref_ids) else m_idx
            ),
            "a": a,
            "c": c,
            "R2_plus": r2,
            "m_plus_sq": m_plus_sq,
            "Delta": delta,
            "dispersion": disp,
            "D_median": disp.get("D_median"),
            "D_q90": disp.get("D_q90"),
            "attribution": attribution,
            "lambda_sum": s,
            "raw_lambda": lam.tolist() if hasattr(lam, "tolist") else list(lam),
            "d_eff_refs": d_eff,
            "chance_R2_ols_mean": chance_r2_ols_mean(n, d_eff),
        }
    )
    return out
