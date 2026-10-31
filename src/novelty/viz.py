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
            "dominant_parent_title": str(parent_meta.get("title") or parent_id)[:100],
        },
    }


def render_altair_subspace_chart(
    projection_data: dict[str, Any],
    *,
    show_cone: bool = True,
    show_labels: bool = True,
    show_zones: bool = True,
    width: int | str = "container",
    height: int = 480,
):
    """
    Build a publication-grade interactive 7-layer Altair chart for Streamlit.
    Visualizes the 2D reference cone geometry, unit horizon arc, and novelty residual.
    """
    import pandas as pd
    try:
        import altair as alt
    except ImportError:
        return None

    if not projection_data or not projection_data.get("ok"):
        return None

    pts = projection_data.get("points") or []
    metrics = projection_data.get("metrics") or {}
    if not pts:
        return None

    df = pd.DataFrame(pts)
    m_val = float(metrics.get("m") or 0.0)
    delta_val = float(metrics.get("Delta") or 0.0)
    r2_val = float(metrics.get("R2_plus") or 0.0)
    focal_y = float(np.sqrt(max(0.0, delta_val)))

    # Layer 1: Unit Sphere Information Horizon (||x|| = 1.0)
    theta = np.linspace(0, np.pi / 2, 70)
    arc_df = pd.DataFrame({"x": np.cos(theta), "y": np.sin(theta)})
    arc_layer = alt.Chart(arc_df).mark_line(
        color="#94a3b8", strokeDash=[5, 5], strokeWidth=2.0
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-0.05, 1.15]), title="Dominant Parent Alignment (e₁ · y_parent = m)"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-0.05, 1.15]), title="Orthogonal Literature Synergy (e₂ = √Δ)"),
    )

    layers = [arc_layer]

    # Layer 2: Vector Decomposition Rays
    if show_cone:
        rays = [
            {"x": 0.0, "y": 0.0, "x2": 1.0, "y2": 0.0, "role": "Dominant Parent Ray", "color": "#f59e0b"},
            {"x": 0.0, "y": 0.0, "x2": m_val, "y2": focal_y, "role": "Cone Projection Vector", "color": "#10b981"},
        ]
        norm_c = float(np.sqrt(m_val ** 2 + focal_y ** 2))
        if norm_c > 1e-4:
            u_x = m_val / norm_c
            u_y = focal_y / norm_c
            rays.append({"x": m_val, "y": focal_y, "x2": u_x, "y2": u_y, "role": "Novelty Residual Gap", "color": "#8b5cf6"})

        rays_df = pd.DataFrame(rays)
        rays_layer = alt.Chart(rays_df).mark_rule(strokeWidth=2.2).encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("color:N", scale=None)
        )
        layers.append(rays_layer)

    # Layer 3: Semantic Discovery Zones Annotations
    if show_zones:
        zones_df = pd.DataFrame([
            {"x": 0.86, "y": 0.06, "text": "Linear Extension Zone (High m, Low Δ)"},
            {"x": 0.50, "y": 0.88, "text": "High Synergy Blend Zone (High Δ)"},
            {"x": 0.20, "y": 0.15, "text": "Paradigm Shift / Outlier Zone (Low R²₊)"},
            {"x": 0.72, "y": 0.74, "text": "Unit Sphere Horizon (‖x‖ = 1.0)"},
        ])
        zones_layer = alt.Chart(zones_df).mark_text(
            color="#64748b", fontSize=10, fontStyle="italic", align="center"
        ).encode(x="x:Q", y="y:Q", text="text:N")
        layers.append(zones_layer)

    # Layer 4: Data Points with Distinct Roles, Shapes, and Colors
    color_scale = alt.Scale(
        domain=["Focal Paper", "Dominant Parent", "Cone Projection (x̂)", "Foundational Parent", "Cited Reference"],
        range=["#d946ef", "#f59e0b", "#06b6d4", "#10b981", "#64748b"],
    )
    shape_scale = alt.Scale(
        domain=["Focal Paper", "Dominant Parent", "Cone Projection (x̂)", "Foundational Parent", "Cited Reference"],
        range=["diamond", "star", "cross", "triangle-up", "circle"],
    )

    points_layer = alt.Chart(df).mark_point(filled=True, opacity=0.85).encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color("role:N", scale=color_scale, title="Subspace Element"),
        shape=alt.Shape("role:N", scale=shape_scale, title="Subspace Element"),
        size=alt.Size("weight:Q", scale=alt.Scale(range=[60, 320]), legend=None),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("title:N", title="Paper Title"),
            alt.Tooltip("role:N", title="Role"),
            alt.Tooltip("weight:Q", title="NNLS Weight", format=".3f"),
            alt.Tooltip("share_pct:Q", title="Attribution Share", format=".1f"),
            alt.Tooltip("x:Q", title="Coordinate X (m)", format=".3f"),
            alt.Tooltip("y:Q", title="Coordinate Y (√Δ)", format=".3f"),
        ],
    )
    layers.append(points_layer)

    # Layer 5: Direct On-Plot Text Callout Labels
    if show_labels:
        labels_list = [
            {"x": m_val, "y": focal_y + 0.05, "text": f"🎯 Focal Paper (m={m_val:.2f}, √Δ={focal_y:.2f})"},
            {"x": 1.0, "y": 0.05, "text": "⭐ Dominant Parent (1.0, 0.0)"},
        ]
        # Identify top contributing reference that is not dominant parent
        top_other = [p for p in pts if p.get("role") == "Foundational Parent" and p.get("weight", 0) > 0.05]
        if top_other:
            best_ref = max(top_other, key=lambda p: p.get("weight", 0))
            labels_list.append({
                "x": float(best_ref["x"]),
                "y": float(best_ref["y"]) + 0.05,
                "text": f"🌱 Key Parent: {str(best_ref.get('title') or best_ref.get('label'))[:25]}",
            })

        labels_df = pd.DataFrame(labels_list)
        labels_layer = alt.Chart(labels_df).mark_text(
            color="#0f172a", fontSize=11, fontWeight="bold", align="center"
        ).encode(x="x:Q", y="y:Q", text="text:N")
        layers.append(labels_layer)

    # Combine into LayerChart
    chart = alt.layer(*layers).properties(
        width=width if isinstance(width, int) else "container",
        height=height,
        title=alt.TitleParams(
            text="2D Reference Cone Geometry and Novelty Subspace",
            subtitle="e₁ = Dominant Parent Axis · e₂ = Orthogonal Blend Dimension · Radius = Reconstruction Energy",
            fontSize=15,
            subtitleFontSize=12,
        ),
    ).interactive()

    return chart


def render_altair_comparison_chart(
    proj_a: dict[str, Any],
    proj_b: dict[str, Any],
    title_a: str = "Paper A",
    title_b: str = "Paper B",
    *,
    height: int = 500,
):
    """
    Build a dual-paper comparative subspace chart in Altair.
    Overlays Paper A and Paper B on the unified 2D plane and renders the evolution trajectory shift arrow.
    """
    import pandas as pd
    try:
        import altair as alt
    except ImportError:
        return None

    if not proj_a or not proj_a.get("ok") or not proj_b or not proj_b.get("ok"):
        return None

    mA = float(proj_a.get("metrics", {}).get("m") or 0.0)
    deltaA = float(proj_a.get("metrics", {}).get("Delta") or 0.0)
    yA = float(np.sqrt(max(0.0, deltaA)))

    mB = float(proj_b.get("metrics", {}).get("m") or 0.0)
    deltaB = float(proj_b.get("metrics", {}).get("Delta") or 0.0)
    yB = float(np.sqrt(max(0.0, deltaB)))

    # 1. Unit sphere horizon
    theta = np.linspace(0, np.pi / 2, 70)
    arc_df = pd.DataFrame({"x": np.cos(theta), "y": np.sin(theta)})
    arc_layer = alt.Chart(arc_df).mark_line(
        color="#94a3b8", strokeDash=[5, 5], strokeWidth=2.0
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-0.05, 1.15]), title="Dominant Parent Alignment (e₁ = m)"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-0.05, 1.15]), title="Orthogonal Literature Synergy (e₂ = √Δ)"),
    )

    # 2. Evolution Trajectory Shift Arrow from Paper A to Paper B
    shift_df = pd.DataFrame([{
        "x": mA, "y": yA, "x2": mB, "y2": yB,
    }])
    shift_rule = alt.Chart(shift_df).mark_rule(
        color="#ec4899", strokeWidth=3.0, strokeDash=[4, 3]
    ).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")

    mid_x = (mA + mB) / 2.0
    mid_y = (yA + yB) / 2.0 + 0.04
    shift_text_df = pd.DataFrame([{
        "x": mid_x, "y": mid_y, "text": f"Evolution Shift (Δm={mB - mA:+.2f}, Δ√Δ={yB - yA:+.2f})"
    }])
    shift_label = alt.Chart(shift_text_df).mark_text(
        color="#db2777", fontSize=11, fontWeight="bold", align="center"
    ).encode(x="x:Q", y="y:Q", text="text:N")

    # 3. Dual Focal Points & Dominant Parent
    focal_points = [
        {
            "label": f"Paper A: {title_a[:35]}",
            "x": mA,
            "y": yA,
            "role": f"Paper A: {title_a[:25]}",
            "size": 320,
            "m": mA,
            "delta": deltaA,
        },
        {
            "label": f"Paper B: {title_b[:35]}",
            "x": mB,
            "y": yB,
            "role": f"Paper B: {title_b[:25]}",
            "size": 320,
            "m": mB,
            "delta": deltaB,
        },
        {
            "label": "Dominant Reference Baseline",
            "x": 1.0,
            "y": 0.0,
            "role": "Dominant Reference (1, 0)",
            "size": 250,
            "m": 1.0,
            "delta": 0.0,
        },
    ]
    focal_df = pd.DataFrame(focal_points)
    points_layer = alt.Chart(focal_df).mark_point(filled=True, opacity=0.9).encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color("role:N", scale=alt.Scale(
            domain=[f"Paper A: {title_a[:25]}", f"Paper B: {title_b[:25]}", "Dominant Reference (1, 0)"],
            range=["#06b6d4", "#d946ef", "#f59e0b"]
        ), title="Comparative Element"),
        shape=alt.Shape("role:N", scale=alt.Scale(
            domain=[f"Paper A: {title_a[:25]}", f"Paper B: {title_b[:25]}", "Dominant Reference (1, 0)"],
            range=["diamond", "diamond", "star"]
        ), title="Comparative Element"),
        size=alt.Size("size:Q", legend=None),
        tooltip=[
            alt.Tooltip("label:N", title="Paper"),
            alt.Tooltip("m:Q", title="Anchor Similarity (m)", format=".3f"),
            alt.Tooltip("delta:Q", title="Synergy Blend (Δ)", format=".3f"),
            alt.Tooltip("x:Q", title="Coord X", format=".3f"),
            alt.Tooltip("y:Q", title="Coord Y", format=".3f"),
        ],
    )

    # 4. Reference points from both papers (background)
    combined_refs = []
    for p in (proj_a.get("points") or []):
        if p.get("role") in ("Foundational Parent", "Cited Reference"):
            combined_refs.append({**p, "paper_origin": "Paper A Ref"})
    for p in (proj_b.get("points") or []):
        if p.get("role") in ("Foundational Parent", "Cited Reference"):
            combined_refs.append({**p, "paper_origin": "Paper B Ref"})

    layers = [arc_layer, shift_rule, shift_label]
    if combined_refs:
        crefs_df = pd.DataFrame(combined_refs)
        crefs_layer = alt.Chart(crefs_df).mark_point(
            color="#94a3b8", size=45, opacity=0.35, shape="circle"
        ).encode(
            x="x:Q", y="y:Q",
            tooltip=[
                alt.Tooltip("title:N", title="Cited Reference"),
                alt.Tooltip("paper_origin:N", title="Cited By"),
                alt.Tooltip("weight:Q", title="Weight", format=".3f"),
            ]
        )
        layers.append(crefs_layer)

    layers.append(points_layer)

    chart = alt.layer(*layers).properties(
        width="container",
        height=height,
        title=alt.TitleParams(
            text="Comparative Subspace Trajectory and Lineage Shift",
            subtitle=f"Comparing {title_a[:35]} vs {title_b[:35]} in 2D Reference Geometry",
            fontSize=15,
            subtitleFontSize=12,
        ),
    ).interactive()

    return chart


def render_ranking_breakdown_chart(
    ranked_papers: list[dict[str, Any]],
    *,
    top_k: int = 8,
    height: int = 280,
):
    """
    Build a stacked horizontal bar chart displaying constituent signal contributions
    for the top-k ranked papers in the recommendation engine.
    """
    import pandas as pd
    try:
        import altair as alt
    except ImportError:
        return None

    if not ranked_papers:
        return None

    signal_labels = {
        "embedding": "Semantic Match",
        "recency": "Recency Boost",
        "citation": "Citation Impact",
        "pagerank": "PageRank Authority",
        "author": "Author Overlap",
        "velocity": "Citation Velocity",
        "llm": "LLM Rerank",
    }
    signal_colors = {
        "Semantic Match": "#8b5cf6",
        "Recency Boost": "#06b6d4",
        "Citation Impact": "#f59e0b",
        "PageRank Authority": "#10b981",
        "Author Overlap": "#ec4899",
        "Citation Velocity": "#3b82f6",
        "LLM Rerank": "#f97316",
    }

    rows = []
    candidates = ranked_papers[:top_k]
    for idx, paper in enumerate(candidates, 1):
        aid = str(paper.get("arxiv_id") or paper.get("paper_id") or f"P{idx}")
        raw_title = str(paper.get("title") or aid).strip()
        short_title = (raw_title[:32] + "...") if len(raw_title) > 35 else raw_title
        label = f"{idx}. {aid}: {short_title}"

        contribution = paper.get("contribution") or {}
        signals = paper.get("signals") or {}

        found_contrib = False
        for sig_key, sig_name in signal_labels.items():
            if sig_key in contribution and not sig_key.startswith("weights"):
                val = float(contribution[sig_key] or 0.0)
                if val > 1e-4:
                    rows.append({
                        "Paper": label,
                        "Signal": sig_name,
                        "Contribution": round(val, 4),
                        "Order": idx,
                    })
                    found_contrib = True

        if not found_contrib and signals:
            for sig_key, sig_name in signal_labels.items():
                if sig_key in signals:
                    val = float(signals[sig_key] or 0.0)
                    if val > 1e-4:
                        rows.append({
                            "Paper": label,
                            "Signal": sig_name,
                            "Contribution": round(val, 4),
                            "Order": idx,
                        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    paper_order = list(dict.fromkeys(df["Paper"].tolist()))

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Paper:N", sort=paper_order, title="Ranked Paper"),
        x=alt.X("Contribution:Q", title="Signal Contribution to Composite Score"),
        color=alt.Color(
            "Signal:N",
            scale=alt.Scale(
                domain=list(signal_colors.keys()),
                range=list(signal_colors.values()),
            ),
            title="Ranking Signal",
        ),
        tooltip=[
            alt.Tooltip("Paper:N", title="Paper"),
            alt.Tooltip("Signal:N", title="Signal Component"),
            alt.Tooltip("Contribution:Q", title="Score Contribution", format=".4f"),
        ],
    ).properties(
        width="container",
        height=height,
        title=alt.TitleParams(
            text="Multi-Signal Score Decomposition Across Ranked Papers",
            subtitle="Constituent contributions of dense embedding, recency, citations, authority, and authors",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )

    return chart

