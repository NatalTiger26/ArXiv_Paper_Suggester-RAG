"""
2D Geometric Subspace & Cone Projection Visualizer.

Projects high-dimensional focal paper vector x and reference vectors Y onto
an orthonormal 2D plane:
  e1 = dominant parent reference (y_m*)
  e2 = orthogonal blend direction in the reference cone (unit(x_hat - (x_hat · e1) e1))

In this plane:
  - Dominant parent sits at (1, 0) on the unit circle.
  - Subspace projection x_hat sits at (m, √Δ), with radial distance √(R²₊).
  - Out-of-cone orthogonal residual is √(1 - R²₊) relative to unit circle ||x|| = 1.
  - Cited references project to (y_j · e1, y_j · e2) with NNLS attribution weights.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np


def project_subspace_2d(
    x: np.ndarray,
    Y: np.ndarray,
    ref_ids: list[str],
    m: float,
    r2_plus: float,
    delta: float,
    lam: np.ndarray,
    metadata_lookup: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute 2D coordinates for focal paper, dominant parent, and references."""
    x_u = np.asarray(x, dtype=np.float64).ravel()
    norm_x = np.linalg.norm(x_u)
    if norm_x > 1e-12:
        x_u = x_u / norm_x

    Y_u = np.asarray(Y, dtype=np.float64)
    if Y_u.ndim == 1:
        Y_u = Y_u.reshape(-1, 1)
    if Y_u.shape[0] != x_u.shape[0] and Y_u.shape[1] == x_u.shape[0]:
        Y_u = Y_u.T

    n_refs = Y_u.shape[1]
    if n_refs == 0:
        return {"ok": False, "error": "no_references"}

    # Normalize reference columns
    cols = []
    for j in range(n_refs):
        col = Y_u[:, j]
        n_col = np.linalg.norm(col)
        cols.append(col / n_col if n_col > 1e-12 else col)
    Y_u = np.column_stack(cols)

    # Closest single parent
    dots = Y_u.T @ x_u
    m_idx = int(np.argmax(dots))
    e1 = Y_u[:, m_idx].copy()
    e1_norm = np.linalg.norm(e1)
    if e1_norm > 1e-12:
        e1 = e1 / e1_norm

    # Reconstructed cone vector x_hat
    lam_arr = np.asarray(lam, dtype=np.float64).ravel()
    if len(lam_arr) == n_refs:
        x_hat = Y_u @ lam_arr
    else:
        x_hat = e1 * m

    # Orthogonal blend direction e2
    proj_e1 = float(np.dot(x_hat, e1))
    v2 = x_hat - proj_e1 * e1
    v2_norm = float(np.linalg.norm(v2))
    if v2_norm > 1e-9:
        e2 = v2 / v2_norm
    else:
        # Fallback to random orthogonal vector to e1
        rng = np.random.RandomState(42)
        rand_v = rng.randn(len(e1))
        v2 = rand_v - np.dot(rand_v, e1) * e1
        e2 = v2 / max(np.linalg.norm(v2), 1e-12)

    # Coordinates
    focal_x = float(m)
    focal_y = float(np.sqrt(max(0.0, delta)))
    parent_id = ref_ids[m_idx] if ref_ids and m_idx < len(ref_ids) else f"ref_{m_idx}"

    sum_lam = float(lam_arr.sum()) if len(lam_arr) == n_refs else 1.0

    points = []
    # 1. Focal paper
    points.append({
        "label": "Focal Paper",
        "id": "focal",
        "role": "Focal Paper",
        "x": round(focal_x, 4),
        "y": round(focal_y, 4),
        "weight": 1.0,
        "share_pct": 100.0,
        "title": "Current Paper",
    })

    # 2. Dominant parent
    parent_meta = metadata_lookup(parent_id) if metadata_lookup else {}
    points.append({
        "label": f"Parent: {parent_id}",
        "id": parent_id,
        "role": "Dominant Parent",
        "x": 1.0,
        "y": 0.0,
        "weight": round(float(lam_arr[m_idx]) if len(lam_arr) > m_idx else 1.0, 4),
        "share_pct": round(float(lam_arr[m_idx] / max(sum_lam, 1e-12) * 100.0) if len(lam_arr) > m_idx else 100.0, 1),
        "title": str(parent_meta.get("title") or parent_id)[:100],
    })

    # 3. Subspace projection point (x_hat)
    points.append({
        "label": "Subspace Reconstruction (x̂)",
        "id": "reconstruction",
        "role": "Cone Projection (x̂)",
        "x": round(focal_x, 4),
        "y": round(focal_y, 4),
        "weight": round(float(np.sqrt(max(0.0, r2_plus))), 4),
        "share_pct": round(float(r2_plus * 100.0), 1),
        "title": "Combined Prior Art Blend",
    })

    # 4. Other reference papers
    for j in range(n_refs):
        if j == m_idx:
            continue
        rid = ref_ids[j] if ref_ids and j < len(ref_ids) else f"ref_{j}"
        px = float(np.dot(Y_u[:, j], e1))
        py = float(np.dot(Y_u[:, j], e2))
        w = float(lam_arr[j]) if len(lam_arr) > j else 0.0
        sh = float(w / max(sum_lam, 1e-12) * 100.0)
        rmeta = metadata_lookup(rid) if metadata_lookup else {}
        is_contributor = w > 1e-4
        points.append({
            "label": str(rmeta.get("paper_id") or rid),
            "id": rid,
            "role": "Foundational Parent" if is_contributor else "Cited Reference",
            "is_contributor": is_contributor,
            "x": round(px, 4),
            "y": round(py, 4),
            "weight": round(w, 4),
            "share_pct": round(sh, 1),
            "title": str(rmeta.get("title") or rid)[:120],
            "url": rmeta.get("url"),
            "provider": rmeta.get("provider"),
        })

    residual_energy = max(0.0, 1.0 - r2_plus)
    residual_distance = float(np.sqrt(residual_energy))

    return {
        "ok": True,
        "points": points,
        "metrics": {
            "m": round(m, 4),
            "R2_plus": round(r2_plus, 4),
            "Delta": round(delta, 4),
            "residual_distance": round(residual_distance, 4),
            "dominant_parent_id": parent_id,
        },
    }


def render_altair_subspace_chart(projection_data: dict[str, Any]):
    """Build an interactive Altair chart for Streamlit."""
    import pandas as pd
    try:
        import altair as alt
    except ImportError:
        return None

    if not projection_data or not projection_data.get("ok"):
        return None

    df = pd.DataFrame(projection_data["points"])

    color_scale = alt.Scale(
        domain=["Focal Paper", "Dominant Parent", "Cone Projection (x̂)", "Cited Reference"],
        range=["#9333ea", "#f59e0b", "#10b981", "#3b82f6"],
    )

    shape_scale = alt.Scale(
        domain=["Focal Paper", "Dominant Parent", "Cone Projection (x̂)", "Cited Reference"],
        range=["diamond", "star", "cross", "circle"],
    )

    base = alt.Chart(df).encode(
        x=alt.X("x:Q", title="Dominant Parent Axis (e₁ · y_parent)", scale=alt.Scale(domain=[-0.2, 1.2])),
        y=alt.Y("y:Q", title="Orthogonal Blend Axis (e₂ · √Δ)", scale=alt.Scale(domain=[-0.2, 1.2])),
        color=alt.Color("role:N", scale=color_scale, title="Element"),
        shape=alt.Shape("role:N", scale=shape_scale, title="Element"),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("title:N", title="Paper Title"),
            alt.Tooltip("role:N", title="Role"),
            alt.Tooltip("weight:Q", title="NNLS Weight", format=".3f"),
            alt.Tooltip("share_pct:Q", title="Attribution Share", format=".1f"),
            alt.Tooltip("x:Q", title="Coordinate X", format=".3f"),
            alt.Tooltip("y:Q", title="Coordinate Y", format=".3f"),
        ],
    )

    points = base.mark_point(size=120, filled=True, opacity=0.85)

    chart = (points).properties(
        width="container" if hasattr(alt, "properties") else 650,
        height=400,
        title="2D Reference Cone Geometry & Subspace Projection",
    ).interactive()

    return chart
