#!/usr/bin/env python3
"""
Unified web UI for ArXiv Paper Suggester.

  uv sync
  uv run python -m streamlit run src/app.py
"""

from __future__ import annotations

import re
import os
import sys
import time
import math
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="ArXiv Paper Suggester",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme state (default dark)
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "dark"


def _inject_theme_css(theme: str) -> None:
    """Dark works with config base=dark; light applies a full override layer."""
    if theme == "light":
        css = """
        <style>
        /* ===== LIGHT MODE (full override of dark base theme) ===== */
        :root {
          --aps-bg: #f3f4f8;
          --aps-surface: #ffffff;
          --aps-sidebar: #eceef5;
          --aps-text: #111318;
          --aps-muted: #5c6370;
          --aps-border: #cdd2dc;
          --aps-primary: #3b5bdb;
          --aps-primary-hover: #364fc7;
        }

        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        section.main,
        .main .block-container {
          background-color: var(--aps-bg) !important;
          color: var(--aps-text) !important;
        }
        [data-testid="stHeader"] {
          background: rgba(243,244,248,0.96) !important;
          border-bottom: 1px solid var(--aps-border);
        }

        /* Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
          background-color: var(--aps-sidebar) !important;
          border-right: 1px solid var(--aps-border);
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] .stMarkdown span,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaption"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
          color: var(--aps-text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaption"],
        [data-testid="stSidebar"] .stCaption {
          color: var(--aps-muted) !important;
        }

        /* Main text */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"] p,
        label,
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
          color: var(--aps-text) !important;
        }
        [data-testid="stMetricLabel"],
        .stCaption,
        [data-testid="stCaption"] {
          color: var(--aps-muted) !important;
        }

        /* Inputs */
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-baseweb="input"] {
          background-color: var(--aps-surface) !important;
          color: var(--aps-text) !important;
          border: 1px solid var(--aps-border) !important;
          caret-color: var(--aps-text) !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
          color: #8b919c !important;
          opacity: 1 !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
          color: var(--aps-muted) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
          color: var(--aps-text) !important;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
          background: var(--aps-surface) !important;
          border: 1px solid var(--aps-border) !important;
          border-radius: 0.5rem !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] svg,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span {
          color: var(--aps-text) !important;
        }

        /* Alerts keep readable text */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
          color: var(--aps-text) !important;
        }

        /* Code */
        code, pre, .stCode code {
          background-color: #e6e8f0 !important;
          color: #1e2430 !important;
        }

        /* ---- Buttons (fix theme-button glitches in light mode) ---- */
        /* Primary */
        button[data-testid="baseButton-primary"],
        button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
          background-color: var(--aps-primary) !important;
          background-image: none !important;
          color: #ffffff !important;
          border: 1px solid var(--aps-primary) !important;
        }
        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span,
        button[kind="primary"] p,
        button[kind="primary"] span {
          color: #ffffff !important;
        }
        button[data-testid="baseButton-primary"]:hover,
        button[kind="primary"]:hover {
          background-color: var(--aps-primary-hover) !important;
          border-color: var(--aps-primary-hover) !important;
          color: #ffffff !important;
        }

        /* Secondary / default */
        button[data-testid="baseButton-secondary"],
        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"] {
          background-color: #ffffff !important;
          background-image: none !important;
          color: var(--aps-text) !important;
          border: 1px solid var(--aps-border) !important;
        }
        button[data-testid="baseButton-secondary"] p,
        button[data-testid="baseButton-secondary"] span,
        button[kind="secondary"] p,
        button[kind="secondary"] span {
          color: var(--aps-text) !important;
        }
        button[data-testid="baseButton-secondary"]:hover,
        button[kind="secondary"]:hover {
          background-color: #eef0f6 !important;
          border-color: #b8bfcc !important;
          color: var(--aps-text) !important;
        }

        /* Download / other buttons */
        button[data-testid="baseButton-secondary"],
        div.stButton > button {
          border-radius: 0.5rem !important;
        }

        /* Theme picker itself: keep both readable */
        div[data-testid="stColumn"] button[data-testid="baseButton-primary"],
        div[data-testid="stColumn"] button[data-testid="baseButton-secondary"] {
          font-weight: 600 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] { color: var(--aps-text) !important; }

        /* Dividers */
        hr { border-color: var(--aps-border) !important; }

        /* Slider track (Quick demo top-N etc.) */
        div[data-baseweb="slider"] div[role="slider"] {
          background-color: var(--aps-primary) !important;
        }

        /* Segmented Controls (Fix black box in light mode) */
        [data-testid="stSegmentedControl"],
        div[data-testid="stSegmentedControl"] > div {
          background-color: #e5e8f0 !important;
          border-radius: 0.5rem !important;
          border: 1px solid var(--aps-border) !important;
        }
        [data-testid="stSegmentedControl"] button,
        [data-testid="stSegmentedControl"] div[role="radiogroup"] button,
        button[data-testid="stSegmentedControlButton"],
        [data-baseweb="button-group"] button {
          background-color: transparent !important;
          color: var(--aps-text) !important;
          border: none !important;
        }
        [data-testid="stSegmentedControl"] button p,
        [data-testid="stSegmentedControl"] button span,
        [data-baseweb="button-group"] button p,
        [data-baseweb="button-group"] button span {
          color: var(--aps-text) !important;
        }
        [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        button[data-testid="stSegmentedControlButton"][aria-checked="true"],
        [data-baseweb="button-group"] button[aria-checked="true"] {
          background-color: #ffffff !important;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        }
        [data-testid="stSegmentedControl"] button[aria-checked="true"] p,
        [data-testid="stSegmentedControl"] button[aria-checked="true"] span {
          color: var(--aps-primary) !important;
          font-weight: 600 !important;
        }

        /* Paper Library Cards (Light) */
        .paper-card-container {
          background-color: #ffffff !important;
          border: 1px solid #cdd2dc !important;
          border-radius: 8px !important;
          padding: 14px 16px !important;
          margin-bottom: 12px !important;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        }
        .paper-badge-arxiv {
          background-color: rgba(59, 91, 219, 0.1) !important;
          color: #3b5bdb !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-weight: 600 !important;
          font-size: 11px !important;
        }
        .paper-badge-ready {
          background-color: rgba(16, 185, 129, 0.12) !important;
          color: #059669 !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-weight: 600 !important;
          font-size: 11px !important;
        }
        .paper-badge-missing {
          background-color: rgba(100, 116, 139, 0.1) !important;
          color: #64748b !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-size: 11px !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        /* ===== DARK MODE (align with config.toml base=dark) ===== */
        :root {
          --aps-bg: #0e1117;
          --aps-surface: #1a1d27;
          --aps-sidebar: #141722;
          --aps-text: #e8eaed;
          --aps-muted: #a0a4b0;
          --aps-border: rgba(232,234,237,0.14);
          --aps-primary: #8b9cff;
        }
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        section.main,
        .main .block-container {
          background-color: var(--aps-bg) !important;
          color: var(--aps-text) !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
          background-color: var(--aps-sidebar) !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1,
        .stMarkdown h2, .stMarkdown h3, label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricValue"] {
          color: var(--aps-text) !important;
        }
        [data-testid="stMetricLabel"], .stCaption, [data-testid="stCaption"] {
          color: var(--aps-muted) !important;
        }
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        [data-baseweb="select"] > div {
          background-color: var(--aps-surface) !important;
          color: var(--aps-text) !important;
        }
        code, pre, .stCode code {
          background-color: #262a36 !important;
          color: #e8eaed !important;
        }
        div[data-testid="stExpander"] {
          border: 1px solid var(--aps-border) !important;
          border-radius: 0.5rem !important;
        }
        /* Leave Streamlit dark button defaults mostly alone — only normalize text */
        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span {
          color: #0e1117 !important;
        }
        button[data-testid="baseButton-secondary"] p,
        button[data-testid="baseButton-secondary"] span {
          color: var(--aps-text) !important;
        }

        /* Paper Library Cards (Dark) */
        .paper-card-container {
          background-color: #1a1d27 !important;
          border: 1px solid rgba(232,234,237,0.14) !important;
          border-radius: 8px !important;
          padding: 14px 16px !important;
          margin-bottom: 12px !important;
        }
        .paper-badge-arxiv {
          background-color: rgba(139, 156, 255, 0.15) !important;
          color: #8b9cff !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-weight: 600 !important;
          font-size: 11px !important;
        }
        .paper-badge-ready {
          background-color: rgba(52, 211, 153, 0.15) !important;
          color: #34d399 !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-weight: 600 !important;
          font-size: 11px !important;
        }
        .paper-badge-missing {
          background-color: rgba(148, 163, 184, 0.1) !important;
          color: #94a3b8 !important;
          padding: 2px 8px !important;
          border-radius: 4px !important;
          font-size: 11px !important;
        }
        </style>
        """

    css += """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .block-container { padding-top: 0.75rem; padding-bottom: 2.5rem; max-width: 1100px; }
    [data-testid="stSidebar"] { min-width: 280px; }
    button:focus-visible, a:focus-visible, input:focus-visible {
      outline: 2px solid #8b9cff !important;
      outline-offset: 2px !important;
    }
    .stMarkdown, .stCaption { line-height: 1.55; }
    @media (prefers-reduced-motion: reduce) {
      * { animation: none !important; transition: none !important; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



def _render_theme_toggle() -> None:
    """Theme switcher: native buttons only (always receives clicks)."""
    theme = st.session_state.get("ui_theme", "dark")
    _inject_theme_css(theme)

    head_l, head_r = st.columns([0.72, 0.28])
    with head_l:
        st.title("ArXiv Paper Suggester")
        st.caption("Find papers · read full text · ask with citations")
    with head_r:
        st.caption("Appearance")
        b_dark, b_light = st.columns(2)
        with b_dark:
            try:
                clicked_dark = st.button(
                    "🌙 Dark",
                    key="theme_btn_dark",
                    type="primary" if theme == "dark" else "secondary",
                    width="stretch",
                )
            except TypeError:
                clicked_dark = st.button(
                    "🌙 Dark",
                    key="theme_btn_dark",
                    type="primary" if theme == "dark" else "secondary",
                    use_container_width=True,
                )
        with b_light:
            try:
                clicked_light = st.button(
                    "☀️ Light",
                    key="theme_btn_light",
                    type="primary" if theme == "light" else "secondary",
                    width="stretch",
                )
            except TypeError:
                clicked_light = st.button(
                    "☀️ Light",
                    key="theme_btn_light",
                    type="primary" if theme == "light" else "secondary",
                    use_container_width=True,
                )
        if clicked_dark and theme != "dark":
            st.session_state["ui_theme"] = "dark"
            st.rerun()
        if clicked_light and theme != "light":
            st.session_state["ui_theme"] = "light"
            st.rerun()




WEIGHT_KEYS = ("embedding", "recency", "citation", "author")

# Explicit values (2 d.p.) — Balanced matches ranking.DEFAULT_WEIGHTS
WEIGHT_PRESETS = {
    "Balanced (default)": {
        "embedding": 0.50,
        "recency": 0.20,
        "citation": 0.25,
        "author": 0.05,
    },
    "Precision (embedding ↑)": {
        "embedding": 0.65,
        "recency": 0.10,
        "citation": 0.20,
        "author": 0.05,
    },
    "Prefer recent": {
        "embedding": 0.45,
        "recency": 0.35,
        "citation": 0.15,
        "author": 0.05,
    },
    "Citation heavy": {
        "embedding": 0.35,
        "recency": 0.10,
        "citation": 0.50,
        "author": 0.05,
    },
}

PRESET_HELP = {
    "Balanced (default)": "Mix of semantic match, recency, citations, and author signal.",
    "Precision (embedding ↑)": "Trust topic similarity more — when the library is already on-topic.",
    "Prefer recent": "Boost newer papers — fast-moving subfields.",
    "Citation heavy": "Prefer highly cited work — can favor older landmarks.",
    "Custom": "Your own weights. Ranking uses these values (re-normalized if needed).",
}


def _fmt_weights(w: dict | None) -> str:
    w = w or WEIGHT_PRESETS["Balanced (default)"]
    return (
        f"embedding={w['embedding']:.2f} · recency={w['recency']:.2f} · "
        f"citation={w['citation']:.2f} · author={w['author']:.2f}"
    )


def _match_preset(w: dict, tol: float = 0.011) -> str | None:
    """Return preset name if weights match within tol, else None."""
    for name, preset in WEIGHT_PRESETS.items():
        if not preset:
            continue
        if all(abs(float(w.get(k, 0)) - float(preset[k])) <= tol for k in WEIGHT_KEYS):
            return name
    return None


def _normalize_weights(w: dict) -> dict:
    s = sum(max(0.0, float(w.get(k, 0))) for k in WEIGHT_KEYS)
    if s <= 1e-9:
        return dict(WEIGHT_PRESETS["Balanced (default)"])
    return {k: round(max(0.0, float(w.get(k, 0))) / s, 4) for k in WEIGHT_KEYS}


# ── Activity log (visible when things feel stuck) ─────────────────────────


def _ensure_log():
    if "activity_log" not in st.session_state:
        st.session_state["activity_log"] = []


def log(msg: str) -> None:
    """Append a timestamped line to the in-app activity log."""
    _ensure_log()
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    st.session_state["activity_log"].append(line)
    # Keep last 80 lines
    st.session_state["activity_log"] = st.session_state["activity_log"][-80:]
    print(line, flush=True)  # also terminal / streamlit console



def int_with_input(
    label: str,
    min_v: int,
    max_v: int,
    default: int,
    *,
    key: str,
    help: str | None = None,
    slider_ratio: float = 0.72,
) -> int:
    """Integer control: slider + exact number field (kept in sync)."""
    if max_v < min_v:
        max_v = min_v
    default = int(max(min_v, min(max_v, default)))
    sk, nk = f"{key}__s", f"{key}__n"

    if key not in st.session_state:
        st.session_state[key] = default
    # Clamp stored value if limits changed (e.g. shorter rank list)
    st.session_state[key] = int(max(min_v, min(max_v, int(st.session_state[key]))))

    if sk not in st.session_state:
        st.session_state[sk] = int(st.session_state[key])
    if nk not in st.session_state:
        st.session_state[nk] = int(st.session_state[key])

    def _from_slider() -> None:
        val = int(st.session_state[sk])
        st.session_state[key] = val
        st.session_state[nk] = val

    def _from_num() -> None:
        val = int(st.session_state[nk])
        val = max(min_v, min(max_v, val))
        st.session_state[key] = val
        st.session_state[sk] = val
        st.session_state[nk] = val

    left, right = st.columns([slider_ratio, 1.0 - slider_ratio])
    with left:
        st.slider(
            label,
            min_value=min_v,
            max_value=max_v,
            key=sk,
            help=help,
            on_change=_from_slider,
        )
    with right:
        st.number_input(
            "Exact",
            min_value=min_v,
            max_value=max_v,
            step=1,
            key=nk,
            help=f"Type any integer from {min_v} to {max_v}",
            on_change=_from_num,
        )
    return int(st.session_state[key])


def float_with_input(
    label: str,
    min_v: float,
    max_v: float,
    default: float,
    step: float,
    *,
    key: str,
    help: str | None = None,
) -> float:
    """Float control: slider + exact number field (kept in sync)."""
    default = float(max(min_v, min(max_v, default)))
    sk, nk = f"{key}__s", f"{key}__n"

    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[key] = float(max(min_v, min(max_v, float(st.session_state[key]))))

    if sk not in st.session_state:
        st.session_state[sk] = float(st.session_state[key])
    if nk not in st.session_state:
        st.session_state[nk] = float(st.session_state[key])

    def _from_slider() -> None:
        val = float(st.session_state[sk])
        st.session_state[key] = val
        st.session_state[nk] = val

    def _from_num() -> None:
        val = float(st.session_state[nk])
        val = max(min_v, min(max_v, val))
        st.session_state[key] = val
        st.session_state[sk] = val
        st.session_state[nk] = val

    left, right = st.columns([0.72, 0.28])
    with left:
        st.slider(
            label,
            min_value=float(min_v),
            max_value=float(max_v),
            step=float(step),
            key=sk,
            help=help,
            on_change=_from_slider,
        )
    with right:
        st.number_input(
            "Exact",
            min_value=float(min_v),
            max_value=float(max_v),
            step=float(step),
            key=nk,
            help=f"Type any value from {min_v} to {max_v}",
            on_change=_from_num,
        )
    return float(st.session_state[key])



def growing_text_area(
    label: str,
    *,
    key: str,
    default: str = "",
    min_lines: int = 3,
    max_lines: int = 24,
    placeholder: str | None = None,
    help: str | None = None,
) -> str:
    """Multi-line text that grows with content (no silent clipping)."""
    if key not in st.session_state:
        st.session_state[key] = default
    text = str(st.session_state.get(key) or "")
    wrap = 0
    for line in (text.splitlines() or [""]):
        wrap += max(1, (len(line) + 88) // 89)
    nlines = max(min_lines, min(max_lines, wrap + 1))
    height = int(22 * nlines + 40)
    return st.text_area(
        label,
        height=height,
        key=key,
        placeholder=placeholder,
        help=help,
    )


def _release_runtime_memory() -> None:
    """Drop transient Python/GPU caches without unloading the embedder."""
    import gc

    gc.collect()
    try:
        import torch

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def set_results_panel(kind: str, payload: dict) -> None:
    """Replace the previous results panel (one active set at a time)."""
    st.session_state["results_panel"] = {
        "kind": kind,
        "payload": payload,
        "ts": time.time(),
    }


def render_results_panel() -> None:
    """Dedicated results area: replaces prior results; lists are collapsible."""
    panel = st.session_state.get("results_panel")
    if not panel:
        return
    kind = panel.get("kind")
    payload = panel.get("payload") or {}
    st.markdown("---")
    st.markdown("### Results")
    if kind == "rank":
        n = int(payload.get("n") or 0)
        k = int(payload.get("k") or n)
        st.caption(f"Ranked **{n}** papers · requested top **{k}**")
        rows = payload.get("rows") or []
        with st.expander(f"Paper list ({len(rows)})", expanded=True):
            for i, r in enumerate(rows, 1):
                pid = r.get("id") or ""
                title = (r.get("title") or "")[:120]
                score = r.get("score", "")
                p_c1, p_c2 = st.columns([5, 1])
                with p_c1:
                    if pid:
                        st.markdown(
                            f"**{i}. [{pid}](https://arxiv.org/abs/{pid})**: {title}"
                        )
                    else:
                        st.markdown(f"**{i}.** {title}")
                    st.caption(f"Score: {score}")
                with p_c2:
                    if pid:
                        if st.button("🔬 Inspect", key=f"inspect_subspace_{pid}_{i}", help="Inspect novelty subspace in Novelty tab"):
                            st.session_state["novelty_target_id"] = pid
                            st.session_state["trigger_nov_compute"] = True
                            st.info(f"Target set to `{pid}`. Switch to Novelty & Graph Lab tab to view its geometry.")
    elif kind == "ingest":
        results = payload.get("results") or []
        ok_n = sum(1 for r in results if r.get("ok"))
        st.caption(f"Ingest finished: **{ok_n}/{len(results)}** succeeded")
        with st.expander(f"Per-paper status ({len(results)})", expanded=True):
            rows = []
            for r in results:
                chunks = r.get("chunks")
                chunks_disp = str(chunks) if (r.get("ok") and chunks is not None) else "—"
                fmt = (r.get("source_format") or "pdf").upper() if r.get("ok") else "—"
                rows.append(
                    {
                        "Title / error": str(r.get("title") or r.get("error") or "")[:120],
                        "arXiv ID": str(r.get("arxiv_id") or ""),
                        "Format": fmt,
                        "Status": "ok" if r.get("ok") else "fail",
                        "Chunks": chunks_disp,
                    }
                )
            try:
                st.dataframe(rows, width="stretch", hide_index=True)
            except TypeError:
                st.dataframe(rows, use_container_width=True, hide_index=True)
        if ok_n:
            st.info("Open the **Ask** tab when ready.")
    elif kind == "library":
        st.caption(payload.get("summary") or "Library updated")
        detail = payload.get("detail")
        if detail:
            with st.expander("Details", expanded=False):
                st.write(detail)


def render_activity_log(expanded: bool = False) -> None:
    _ensure_log()
    with st.expander("Activity log (if something feels stuck, open this)", expanded=expanded):
        lines = st.session_state.get("activity_log") or []
        if not lines:
            st.caption("No activity yet.")
        else:
            st.code("\n".join(lines[-40:]), language=None)
        if st.button("Clear log", key="clear_activity_log"):
            st.session_state["activity_log"] = []
            st.rerun()


def friendly_error(exc: BaseException | str) -> str:
    msg = str(exc)
    low = msg.lower()
    log(f"ERROR: {msg[:300]}")
    if "dimension" in low and ("768" in msg or "512" in msg or "embedding" in low):
        return (
            "**Embedding dimension mismatch.** Vector store dim ≠ current `EMBEDDING_DIM`.\n\n"
            "```bash\n"
            "uv run python -m src.rag.cli reset-index\n"
            "uv run python -m src.rag.cli ingest --force <ids…>\n"
            "```"
        )
    if "api key" in low or "apikey" in low or "authentication" in low or "401" in low:
        return (
            "**API key missing or rejected.** Set `GEMINI_API_KEY` in `.env` and restart the app."
        )
    if "not found" in low and "model" in low:
        return (
            "**Model name not available.** Check `GEMINI_MODEL` in `.env`."
        )
    if "connection" in low or "timeout" in low or "network" in low:
        return f"**Network problem.**\n\n`{msg[:300]}`"
    if "collection" in low and "not" in low:
        return (
            "**Vector index missing.** Rebuild recommend index or ingest full-text papers first."
        )
    return f"**Something went wrong.**\n\n`{msg[:500]}`"


@st.cache_data(ttl=120, show_spinner=False)
def corpus_status_cached():
    """Library size + recommend index flag. Cheap; cached 2 min."""
    try:
        from corpus_builder import corpus_status

        return corpus_status()
    except Exception as e:
        return {
            "n_papers": 0,
            "chroma_exists": False,
            "error": str(e),
            "sample_ids": [],
        }


@st.cache_data(ttl=120, show_spinner=False)
def rag_stats_cached():
    """Counts only — does NOT scan all Chroma chunks (that froze the UI)."""
    try:
        from rag.embed_store import index_stats

        return index_stats(include_papers=False)
    except Exception:
        try:
            from src.rag.embed_store import index_stats

            return index_stats(include_papers=False)
        except Exception as e:
            return {
                "paper_count": 0,
                "chunk_count": 0,
                "papers": [],
                "error": str(e),
                "embedding_model": "—",
            }


@st.cache_data(ttl=300, show_spinner=False)
def rag_papers_catalog_cached():
    """Paper id+title list for filters. Prefer progress files; rare Chroma scan."""
    try:
        from rag.embed_store import list_ingested_papers

        return list_ingested_papers(force_chroma=False)
    except Exception:
        try:
            from src.rag.embed_store import list_ingested_papers

            return list_ingested_papers(force_chroma=False)
        except Exception:
            return []


def clear_data_caches():
    corpus_status_cached.clear()
    rag_stats_cached.clear()
    rag_papers_catalog_cached.clear()
    try:
        from rag.embed_store import invalidate_paper_list_cache

        invalidate_paper_list_cache()
    except Exception:
        try:
            from src.rag.embed_store import invalidate_paper_list_cache

            invalidate_paper_list_cache()
        except Exception:
            pass
    # Drop session catalog so next run refreshes
    if "papers_catalog" in st.session_state:
        del st.session_state["papers_catalog"]


def save_library(papers: list, replace: bool = False) -> int:
    from corpus_builder import load_corpus, save_corpus, merge_papers

    if replace:
        save_corpus(papers)
        return len(papers)
    merged = merge_papers(load_corpus(), papers)
    save_corpus(merged)
    return len(merged)


def system_health() -> dict:
    try:
        from settings import FALLBACK_LLM_API_KEY, EMBEDDING_MODEL
    except Exception:
        FALLBACK_LLM_API_KEY = os.getenv("FALLBACK_LLM_API_KEY") or os.getenv(
            "NVIDIA_API_KEY"
        )
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
    gemini = os.getenv("GEMINI_API_KEY", "")
    return {
        "gemini_key": bool(gemini and str(gemini).strip()),
        "fallback_key": bool(
            FALLBACK_LLM_API_KEY and str(FALLBACK_LLM_API_KEY).strip()
        ),
        "embedding_model": EMBEDDING_MODEL,
    }


def next_step_hint(cs: dict, rs: dict) -> tuple[str, str]:
    n = int(cs.get("n_papers") or 0)
    rag_n = int(rs.get("paper_count") or 0)
    if n == 0:
        return (
            "Start here: build a library",
            "Add papers by keywords or arXiv IDs on the **Library** tab, then rebuild the index.",
        )
    if not cs.get("chroma_exists"):
        return (
            "Library ready — build the recommend index",
            "Open **Library** → “Rebuild index only”.",
        )
    if rag_n == 0:
        return (
            "Recommend papers, then ingest for Q&A",
            "Use **Recommend** → **Ingest for Q&A**. Ask only sees full-text papers.",
        )
    return (
        "Ready to ask",
        "Use **Ask** for grounded answers with citations.",
    )


def push_recent_question(q: str) -> None:
    q = (q or "").strip()
    if not q:
        return
    hist = st.session_state.setdefault("recent_questions", [])
    hist = [x for x in hist if x != q]
    hist.insert(0, q)
    st.session_state["recent_questions"] = hist[:8]


def paper_id_from_row(r: dict) -> str:
    pid = r.get("arxiv_id") or r.get("paper_id") or r.get("id") or ""
    pid = str(pid).replace("arXiv:", "").replace("http://arxiv.org/abs/", "")
    pid = pid.replace("https://arxiv.org/abs/", "")
    if "v" in pid and pid.split("v")[0].replace(".", "").replace("/", "").isdigit() is False:
        # keep version strip for standard ids like 1806.07572v1
        pass
    if "v" in pid:
        base, _, ver = pid.rpartition("v")
        if ver.isdigit():
            pid = base
    return pid.strip()


def format_paper_label(pid: str, title: str | None = None, *, max_title: int = 80) -> str:
    """Human-readable option: 'Title (arxiv_id)' for multiselect / search."""
    pid = (pid or "").strip()
    t = (title or "").strip().replace("\n", " ")
    if len(t) > max_title:
        t = t[: max_title - 1].rstrip() + "…"
    if t and pid:
        return f"{t} ({pid})"
    if pid:
        return pid
    return t or "(unknown)"


def parse_paper_label(label: str) -> str:
    """Extract arXiv id from a format_paper_label string, or bare id."""
    import re

    s = (label or "").strip()
    if not s:
        return ""
    m = re.search(r"\(([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)\)\s*$", s)
    if m:
        return paper_id_from_row({"id": m.group(1)})
    m = re.search(r"\(([a-z\-]+/[0-9]{7}(?:v[0-9]+)?)\)\s*$", s, re.I)
    if m:
        return paper_id_from_row({"id": m.group(1)})
    # bare id
    return paper_id_from_row({"id": s})


def ingested_papers_catalog(rs: dict | None = None, *, force: bool = False) -> list[dict]:
    """Normalized list of {paper_id, title, label} for UI pickers.

    Uses session + cached progress-file catalog — not a full Chroma scan on each
    keystroke / click.
    """
    if not force and st.session_state.get("papers_catalog"):
        cached = st.session_state["papers_catalog"]
        # If cache has ids but every title empty and user searches titles, still OK for ids
        if cached:
            return cached

    raw = rag_papers_catalog_cached()
    if not raw and rs:
        raw = rs.get("papers") or []

    out: list[dict] = []
    seen: set[str] = set()
    for p in raw:
        pid = paper_id_from_row(p)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        title = (p.get("title") or "").strip()
        out.append(
            {
                "paper_id": pid,
                "title": title,
                "label": format_paper_label(pid, title),
            }
        )
    out.sort(key=lambda x: (x["title"] or x["paper_id"]).lower())
    st.session_state["papers_catalog"] = out
    return out


def paper_scope_picker(
    catalog: list[dict],
    *,
    state_key: str = "ask_scope_ids",
) -> list[str]:
    """
    Restrict Ask to selected papers:

      1. Type title or arXiv id -> Find
      2. Tick matches -> Add to selection
      3. Empty selection = search all papers

    Returns list of arXiv ids (empty = no restriction).
    """
    if not catalog:
        st.caption("No full-text papers loaded yet. Ingest from **Recommend** first.")
        return []

    id_to_row = {c["paper_id"]: c for c in catalog}
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    st.session_state[state_key] = [
        pid for pid in st.session_state[state_key] if pid in id_to_row
    ]

    n = len(catalog)
    st.markdown(
        f"Optional: limit answers to specific papers ({n} full-text available). "
        "**Empty selection = search all.**"
    )

    selected = list(st.session_state[state_key])
    if selected:
        st.markdown(f"**Selected for this question ({len(selected)})**")
        for pid in selected[:15]:
            row = id_to_row.get(pid) or {}
            title = (row.get("title") or pid)[:70]
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(pid)
            with c2:
                if st.button("Remove", key=f"{state_key}_rm_{pid}", help=f"Remove {pid}"):
                    st.session_state[state_key] = [
                        x for x in st.session_state[state_key] if x != pid
                    ]
                    st.rerun()
        if len(selected) > 15:
            st.caption(f"... and {len(selected) - 15} more")
        if st.button("Clear selection (use all papers)", key=f"{state_key}_clear"):
            st.session_state[state_key] = []
            st.rerun()
    else:
        st.info("No paper filter — Ask will search **all** ingested full-text papers.")

    st.markdown("**Find papers to add**")
    with st.form(key=f"{state_key}_find_form", clear_on_submit=False):
        q = st.text_input(
            "Search by title or arXiv id",
            placeholder="e.g. Grad-CAM, interpretability, 2501.16496",
            help="Matches titles and ids. Press Find, then add matches below.",
        )
        find_clicked = st.form_submit_button("Find")

    if find_clicked:
        st.session_state[f"{state_key}_last_q"] = (q or "").strip()

    last_q = (st.session_state.get(f"{state_key}_last_q") or "").strip()
    if last_q:
        q_norm = last_q.lower().strip()
        # also try as normalized arxiv id
        q_id = parse_paper_label(last_q) or q_norm
        q_id = q_id.lower().replace("arxiv:", "").strip()
        tokens = [t for t in q_norm.replace(",", " ").split() if t]

        def _row_hay(c: dict) -> str:
            return " ".join(
                [
                    str(c.get("title") or ""),
                    str(c.get("paper_id") or ""),
                    str(c.get("arxiv_id") or ""),
                    str(c.get("label") or ""),
                ]
            ).lower()

        matches: list[dict] = []
        # Pass 1: exact / prefix id match (highest priority)
        for c in catalog:
            pid = str(c.get("paper_id") or "").lower()
            if not pid:
                continue
            if q_id == pid or pid.startswith(q_id) or q_id in pid:
                if len(q_id) >= 4:  # avoid tiny accidental hits
                    matches.append(c)
            if len(matches) >= 40:
                break
        # Pass 2: token search on title+id if still empty or id query was short
        if not matches and tokens:
            for c in catalog:
                hay = _row_hay(c)
                if all(t in hay for t in tokens):
                    matches.append(c)
                if len(matches) >= 40:
                    break
        # Pass 3: single-substring on full hay
        if not matches and len(q_norm) >= 3:
            for c in catalog:
                if q_norm in _row_hay(c):
                    matches.append(c)
                if len(matches) >= 40:
                    break

        st.caption(f"Search **{last_q}** → {len(matches)} match(es) (max 40 shown)")
        if not matches:
            # Show a few catalog samples so user can see real ids/titles
            sample = catalog[:5]
            sample_txt = "; ".join(
                f"`{c.get('paper_id')}` {(c.get('title') or '')[:40]}" for c in sample
            )
            st.warning(
                "No matches. Try a shorter keyword, or an id exactly as in the library."
            )
            st.caption(f"Catalog sample ({len(catalog)} papers): {sample_txt}")
        else:
            match_labels = [c["label"] for c in matches]
            label_to_id = {c["label"]: c["paper_id"] for c in matches}
            picked = st.multiselect(
                "Matches — choose papers to add",
                options=match_labels,
                key=f"{state_key}_match_pick",
            )
            b1, b2 = st.columns(2)
            with b1:
                add = st.button(
                    "Add to selection",
                    type="primary",
                    key=f"{state_key}_add",
                    disabled=not picked,
                )
            with b2:
                add_all = st.button("Add all matches", key=f"{state_key}_add_all")
            if add and picked:
                cur = list(st.session_state[state_key])
                for lab in picked:
                    pid = label_to_id.get(lab) or parse_paper_label(lab)
                    if pid and pid not in cur:
                        cur.append(pid)
                st.session_state[state_key] = cur
                st.rerun()
            if add_all:
                cur = list(st.session_state[state_key])
                for c in matches:
                    if c["paper_id"] not in cur:
                        cur.append(c["paper_id"])
                st.session_state[state_key] = cur
                st.rerun()
    else:
        st.caption("Enter a title keyword or arXiv id, then press **Find**.")

    return list(st.session_state[state_key])


def paper_picker_multiselect(
    catalog: list[dict],
    *,
    key: str,
    label: str = "Restrict to papers",
    help_text: str | None = None,
    large_threshold: int = 120,
) -> list[str]:
    """Backward-compatible alias."""
    return paper_scope_picker(catalog, state_key=key or "ask_scope_ids")


def show_ingest_results(results: list[dict]) -> None:
    if not results:
        st.warning("No ingest results.")
        return
    ok_n = sum(1 for r in results if r.get("ok"))
    log(f"Ingest summary: {ok_n}/{len(results)} ok")
    set_results_panel("ingest", {"results": results})
    render_results_panel()


def answer_to_markdown(question: str, result: dict) -> str:
    lines = [f"# {question}", "", result.get("answer") or "", "", "## Sources", ""]
    for r in result.get("references") or []:
        lines.append(
            f"- [{r.get('n')}] {r.get('title') or ''} "
            f"(arXiv:{r.get('paper_id')}, § {r.get('section')}, score={r.get('score')})"
        )
        if r.get("snippet"):
            lines.append(f"  - {r['snippet'][:200]}")
    return "\n".join(lines)


def render_answer_block(
    question: str,
    result: dict,
    *,
    key_prefix: str = "ask",
    scope_ids: list[str] | None = None,
    scope_labels: list[str] | None = None,
    corpus_paper_count: int | None = None,
) -> None:
    if result.get("cache_hit"):
        st.info(
            f"Cached answer (similarity {result.get('cache_score')}). "
            "Turn off cache in Advanced for a fresh LLM call."
        )
    # Scope line: which papers were searched
    if scope_ids:
        n = len(scope_ids)
        st.caption(f"**Searched over** {n} selected paper{'s' if n != 1 else ''} (filter on).")
        with st.expander(f"Filter: {n} paper(s)", expanded=False):
            for lab in (scope_labels or scope_ids)[:30]:
                st.markdown(f"- {lab}")
            if n > 30:
                st.caption(f"… and {n - 30} more")
    elif corpus_paper_count is not None:
        st.caption(f"**Searched over** all **{corpus_paper_count}** ingested full-text papers.")

    cov = result.get("coverage") or "n/a"
    if cov == "high":
        st.success(f"Coverage: **{cov}**")
    elif cov == "partial":
        st.warning(f"Coverage: **{cov}** — answer may be incomplete.")
    else:
        st.error(f"Coverage: **{cov}** — library may not contain this topic.")
    st.caption(
        "Coverage is from retrieval scores, not a calibrated probability. "
        "`high` is better; `low`/`none` means the topic may be missing from full-text."
    )
    if cov in ("low", "none"):
        st.info(
            "**Next step:** clear the paper filter if you used one, or go to **Recommend** → "
            "rank your interest → **Ingest for Q&A** so the right full texts are available."
        )
    st.markdown("#### Answer")
    st.markdown(result.get("answer") or "_No answer_")
    refs = result.get("references") or []
    if refs:
        st.markdown("#### Sources")
        for r in refs:
            title = (r.get("title") or r.get("paper_id") or "")[:90]
            with st.expander(f"[{r['n']}] {title}  ·  score={r.get('score')}"):
                st.markdown(
                    f"**arXiv** `{r.get('paper_id')}` · **Section** {r.get('section')}"
                )
                st.caption(r.get("snippet") or "")
                pid = r.get("paper_id") or ""
                if pid:
                    st.link_button("Open on arXiv", f"https://arxiv.org/abs/{pid}")
                st.code(
                    f"[{r['n']}] {r.get('title')} (arXiv:{r.get('paper_id')})",
                    language=None,
                )
    md = answer_to_markdown(question, result)
    # Keys must be unique across ALL tabs (Streamlit runs every tab body each rerun).
    safe_q = abs(hash((key_prefix, question))) % 10_000_000
    st.download_button(
        "Download answer (Markdown)",
        data=md,
        file_name="answer.md",
        mime="text/markdown",
        key=f"dl_{key_prefix}_{safe_q}",
    )


def run_ingest_with_progress(
    ids: list[str],
    force: bool = False,
    *,
    low_memory: bool = False,
) -> list[dict]:
    ids = [str(i).strip() for i in ids if i and str(i).strip()]
    if not ids:
        return []
    log(f"Ingest start: {len(ids)} papers force={force} low_mem={low_memory} → {ids}")

    # Static guidance (educated choice of N)
    st.caption(
        "Typical cost **per paper** (first time, EmbeddingGemma on Apple GPU): "
        "download ~2–10s · convert/OCR ~10–60s · embed ~5–30s. "
        "Roughly **~30–90s/paper** on average; OCR-heavy PDFs can exceed **2–4 min**. "
        "Cached PDFs/Markdown are much faster. "
        f"Ballpark for **{len(ids)}** papers: "
        f"**{_fmt_duration(45 * len(ids))}–{_fmt_duration(90 * len(ids))}** "
        "(wide range — live ETA below adapts to this run)."
    )
    if len(ids) >= 10:
        st.warning(
            f"Ingesting **{len(ids)}** papers is RAM-heavy with EmbeddingGemma. "
            "Expect high memory use; enable **Low memory mode** if the system gets tight. "
            "MPS out-of-memory on a paper will auto-retry after a cache clear."
        )

    progress = st.progress(0, text="Preparing…")
    status = st.empty()
    eta_box = st.empty()
    results: list[dict] = []
    per_paper_times: list[float] = []

    try:
        try:
            from rag.embed_store import load_embedding_model
            from rag.ingest import ingest_paper
        except ImportError:
            from src.rag.embed_store import load_embedding_model
            from src.rag.ingest import ingest_paper

        status.markdown(
            "**Step 1/2 — Loading embedding model** (first load can take 1–2 min)…"
        )
        progress.progress(3, text="Loading embedder…")
        t0 = time.time()
        model = load_embedding_model()
        log(f"Embedder ready in {time.time() - t0:.1f}s")
        progress.progress(8, text="Embedder ready")

        if low_memory:
            # Prefer smaller encode batches for this process
            os.environ["RAG_EMBED_BATCH"] = "8"
        else:
            os.environ.pop("RAG_EMBED_BATCH", None)

        n = len(ids)
        for i, aid in enumerate(ids):
            elapsed = sum(per_paper_times)
            avg = (elapsed / len(per_paper_times)) if per_paper_times else 60.0
            remaining = avg * (n - i)
            eta_box.caption(
                f"Progress **{i}/{n}** papers · elapsed **{_fmt_duration(elapsed)}** · "
                f"ETA **~{_fmt_duration(remaining)}** "
                f"(avg **{_fmt_duration(avg)}**/paper so far)"
            )
            step_lo = 8 + int(90 * i / n)
            progress.progress(
                step_lo,
                text=f"Paper {i+1}/{n}: {aid} — HTML / PDF convert / embed…",
            )
            status.markdown(
                f"**Step 2/2 — Paper `{aid}` ({i+1}/{n})**\n"
                "1. Fetch arXiv HTML (or PDF fallback) → 2. Clean Markdown → 3. Chunk → 4. Embed & store"
            )
            log(f"Ingesting {aid} ({i+1}/{n})…")
            t1 = time.time()
            row: dict
            try:
                row = ingest_paper(aid, model=model, force_download=force)
            except Exception as e:
                err = str(e)
                if "out of memory" in err.lower() or "oom" in err.lower():
                    log(f"  OOM on {aid}; releasing caches and retrying once…")
                    _release_runtime_memory()
                    try:
                        row = ingest_paper(aid, model=model, force_download=force)
                    except Exception as e2:
                        row = {"arxiv_id": aid, "ok": False, "error": str(e2)}
                        log(f"  failed {aid} after OOM retry: {e2}")
                else:
                    row = {"arxiv_id": aid, "ok": False, "error": err}
                    log(f"  failed {aid}: {e}")
            else:
                if row.get("ok"):
                    log(
                        f"  ok {aid}: {row.get('chunks')} chunks in {time.time()-t1:.1f}s"
                    )
                else:
                    log(f"  fail {aid}: {row.get('error')}")

            dt = time.time() - t1
            per_paper_times.append(dt)
            results.append(row)
            _release_runtime_memory()

            elapsed = sum(per_paper_times)
            avg = elapsed / len(per_paper_times)
            remaining = avg * (n - (i + 1))
            progress.progress(
                8 + int(90 * (i + 1) / n),
                text=(
                    f"Finished {aid} ({i+1}/{n}) · "
                    f"{_fmt_duration(elapsed)} elapsed · ETA ~{_fmt_duration(remaining)}"
                ),
            )
            eta_box.caption(
                f"Progress **{i+1}/{n}** papers · elapsed **{_fmt_duration(elapsed)}** · "
                f"ETA **~{_fmt_duration(remaining)}** · "
                f"last paper **{_fmt_duration(dt)}** · avg **{_fmt_duration(avg)}**/paper"
            )

        progress.progress(100, text="Ingest complete")
        status.success("Ingest finished — see **Results** below.")
        log("Ingest complete")
        return results
    except Exception as e:
        status.empty()
        st.markdown(friendly_error(e))
        log(traceback.format_exc()[-500:])
        return results
    finally:
        _release_runtime_memory()



def run_rank(interest: str, k: int, weights: dict | None) -> list[dict]:
    log(f"Rank start: {(interest[:100] + "…") if len(interest) > 100 else interest!r} k={k}")
    progress = st.progress(0, text="Loading ranking stack…")
    status = st.empty()
    try:
        status.markdown("**Loading corpus + embedding model for ranking**…")
        progress.progress(15, text="Loading model / corpus…")
        from ranking import score_and_rank

        status.markdown("**Scoring papers** (embedding + recency + citation + author)…")
        progress.progress(40, text="Scoring…")
        t0 = time.time()
        ranked = score_and_rank(interest.strip(), k=int(k), weights=weights)
        log(f"Rank done in {time.time()-t0:.1f}s → {len(ranked)} results")
        progress.progress(100, text="Ranking complete")
        status.empty()
        return ranked
    except Exception as e:
        status.empty()
        progress.empty()
        st.markdown(friendly_error(e))
        log(traceback.format_exc()[-500:])
        return []


# ── Session ───────────────────────────────────────────────────────────────

_ensure_log()
if "recent_questions" not in st.session_state:
    st.session_state["recent_questions"] = []
if "last_ask" not in st.session_state:
    st.session_state["last_ask"] = None
if "last_recommend" not in st.session_state:
    st.session_state["last_recommend"] = None

try:
    cs = corpus_status_cached()
    rs = rag_stats_cached()
except Exception as _startup_err:
    st.error("Startup status load failed — see details below.")
    st.exception(_startup_err)
    cs = {"n_papers": 0, "chroma_exists": False}
    rs = {"paper_count": 0, "chunk_count": 0, "papers": []}

# Prefer progress-file paper count when Chroma has chunks but count is 0
if int(rs.get("paper_count") or 0) == 0 and int(rs.get("chunk_count") or 0) > 0:
    try:
        _cat = ingested_papers_catalog(rs)
        if _cat:
            rs = dict(rs)
            rs["paper_count"] = len(_cat)
    except Exception:
        pass

hint_title, hint_body = next_step_hint(cs, rs)
try:
    health = system_health()
except Exception:
    health = {"gemini_key": False, "fallback_key": False, "embedding_model": "—"}
lib_n = int(cs.get("n_papers") or 0)
rag_n = int(rs.get("paper_count") or 0)

with st.sidebar:
    st.markdown("### Workspace")
    m1, m2 = st.columns(2)
    m1.metric("Library", lib_n, help="Titles/abstracts for ranking")
    m2.metric("Full-text", rag_n, help="PDFs for Ask / RAG")
    st.caption(
        f"Recommend index: **{'ready' if cs.get('chroma_exists') else 'not built'}**  ·  "
        f"Chunks: **{rs.get('chunk_count', 0)}**"
    )
    if rs.get("embedding_model"):
        st.caption(f"Embedder: `{rs.get('embedding_model')}`")
        if rs.get("embedding_dim") not in (None, "", "None"):
            st.caption(f"Dim: `{rs.get('embedding_dim')}`")
    st.caption("Library = rank · Full-text = ask")

    if st.button("Refresh status", width="stretch"):
        clear_data_caches()
        log("Status cache cleared")
        st.rerun()

    with st.expander("Full-text papers in RAG"):
        # Only resolve catalog when user opens this expander (still runs body, but cached)
        cat = ingested_papers_catalog(rs) if rag_n else []
        if not cat:
            if int(rs.get("chunk_count") or 0) > 0:
                st.caption(
                    f"{rs.get('chunk_count')} chunks indexed — open **Ask → filter** "
                    "or refresh after ingest to list titles."
                )
            else:
                st.caption("None yet — ingest from **Recommend**.")
        else:
            for p in cat[:40]:
                pid = p.get("paper_id") or ""
                title = (p.get("title") or "")[:70]
                if title:
                    st.markdown(f"**{title}**  \n`{pid}`")
                else:
                    st.markdown(f"`{pid}`")
            if len(cat) > 40:
                st.caption(f"… and {len(cat) - 40} more (search in **Ask → filter**)")

    with st.expander("System health", expanded=False):
        st.write(
            f"Gemini API key: **{'present' if health['gemini_key'] else 'missing'}**"
        )
        st.write(
            f"Fallback LLM key: **{'present' if health['fallback_key'] else 'not set'}**"
        )
        st.caption(f"Embedding model: `{health['embedding_model']}`")
        if not health["gemini_key"]:
            st.warning("Set `GEMINI_API_KEY` in `.env` for generation.")

    with st.expander("API & Cost Telemetry", expanded=False):
        try:
            from src.telemetry import get_telemetry_summary, reset_telemetry

            t_summary = get_telemetry_summary()
            st.metric("Total API Calls", t_summary["total_calls"])
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Total Tokens", f"{t_summary['total_tokens']:,}")
            col_t2.metric("Est. Cost", f"${t_summary['total_cost_usd']:.4f}")
            st.caption(
                f"Latency p50: **{t_summary['p50_latency_ms']}ms**  ·  "
                f"Cache hits: **{t_summary['cache_hit_rate_pct']}%**"
            )

            # Confirmed reset
            if "telemetry_reset_confirm" not in st.session_state:
                st.session_state["telemetry_reset_confirm"] = False

            if not st.session_state["telemetry_reset_confirm"]:
                if st.button("Reset usage stats", key="btn_reset_telemetry_prep"):
                    st.session_state["telemetry_reset_confirm"] = True
                    st.rerun()
            else:
                st.warning("Confirm resetting all usage stats?")
                c_yes, c_no = st.columns(2)
                with c_yes:
                    if st.button("Confirm Reset", type="primary", key="btn_reset_telemetry_do"):
                        reset_telemetry()
                        st.session_state["telemetry_reset_confirm"] = False
                        st.success("Telemetry stats reset.")
                        st.rerun()
                with c_no:
                    if st.button("Cancel", key="btn_reset_telemetry_cancel"):
                        st.session_state["telemetry_reset_confirm"] = False
                        st.rerun()
        except Exception as e:
            st.caption(f"Telemetry not available: {e}")

    st.divider()
    st.caption("CLI (see WORKFLOW.md)")
    st.code(
        "uv run python src/corpus_builder.py search \"…\" -n 12 --build-index\n"
        "uv run python src/recommend.py \"…\" -k 8\n"
        "uv run python -m src.rag.cli ingest <id>\n"
        "uv run python -m src.rag.cli ask \"…\"",
        language="bash",
    )

_render_theme_toggle()

if lib_n > 0 and rag_n < lib_n:
    st.warning(
        f"**Library has {lib_n} papers but only {rag_n} are full-text.**  \n"
        "Ask only searches ingested PDFs. Use **Recommend → Ingest for Q&A**."
    )

if lib_n == 0 or not cs.get("chroma_exists") or rag_n == 0:
    st.info(f"**{hint_title}**  \n{hint_body}")
else:
    st.success(f"**{hint_title}** — {hint_body}")

render_activity_log(expanded=False)

tab_home, tab_lib, tab_rec, tab_ask, tab_nov, tab_help = st.tabs(
    ["Home", "Library", "Recommend", "Ask", "Novelty & Graph", "Help"]
)


# ── HOME ──────────────────────────────────────────────────────────────────

with tab_home:
    st.subheader("Your path")
    st.markdown(
        """
1. **Library** — add papers (search or arXiv IDs) and build the recommend index  
2. **Recommend** — rank by research interest, then **Ingest for Q&A** (full text)  
3. **Ask** — questions with citations (only over **ingested** full-text papers)

**Library ≠ Ask.** Ranking uses titles/abstracts; answers need full-text ingest.
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Add papers")
        st.write("Search arXiv or paste IDs.")
        if lib_n == 0:
            st.warning("Library empty")
        else:
            st.caption(f"{lib_n} in library")
        st.caption("→ **Library** tab")
    with c2:
        st.markdown("#### 2. Rank & ingest")
        st.write("Multi-signal ranking, then full text.")
        if rag_n == 0 and lib_n > 0:
            st.warning("No full-text yet")
        else:
            st.caption(f"{rag_n} full-text")
        st.caption("→ **Recommend** tab")
    with c3:
        st.markdown("#### 3. Ask questions")
        st.write("Grounded answers + citations.")
        if rag_n == 0:
            st.warning("Ingest first")
        else:
            st.success("Ready")
        st.caption("→ **Ask** tab")

    st.divider()
    st.markdown("#### Quick demo")
    st.write(
        "One click: seed library (if empty) → rebuild index → rank → ingest top papers → ask. "
        "Watch the **Activity log** — first embedder load is slow."
    )
    d1, d2 = st.columns(2)
    with d1:
        demo_interest = growing_text_area(
            "Research interest",
            key="home_demo_interest",
            default="mechanistic interpretability circuits",
            min_lines=2,
            max_lines=12,
        )
    with d2:
        demo_q = st.text_input(
            "Question",
            value="What are open problems in mechanistic interpretability?",
            key="home_demo_q",
        )
    demo_k = int_with_input(
        "Top papers to ingest",
        1,
        25,
        3,
        key="home_demo_k",
        help="How many ranked papers to download and embed (1–25).",
    )

    if st.button("Run quick demo", type="primary", key="home_demo_btn"):
        log("=== Quick demo start ===")
        overall = st.progress(0, text="Demo starting…")
        try:
            if lib_n == 0:
                overall.progress(5, text="Searching arXiv…")
                log(f"Demo: search arXiv {demo_interest!r}")
                from corpus_builder import search_arxiv

                papers = search_arxiv(demo_interest, max_results=12)
                if not papers:
                    st.error("arXiv search returned no papers.")
                    log("Demo: search empty")
                else:
                    size = save_library(papers, replace=True)
                    st.write(f"Seeded library: **{size}** papers")
                    log(f"Demo: library size {size}")
                    clear_data_caches()

            overall.progress(25, text="Building recommend index…")
            log("Demo: rebuild_index")
            from corpus_builder import rebuild_index

            rebuild_index()
            log("Demo: index ready")

            overall.progress(45, text="Ranking…")
            ranked = run_rank(demo_interest, demo_k, None)
            ids = [paper_id_from_row(r) for r in ranked[:demo_k]]
            ids = [x for x in ids if x]
            if ranked:
                st.markdown("**Top papers**")
                for i, r in enumerate(ranked[:demo_k], 1):
                    pid = paper_id_from_row(r)
                    title = (r.get("title") or "")[:100]
                    st.markdown(f"{i}. **{title}** (`{pid}`)")
            else:
                st.write("Top papers: (none)")

            if ids:
                overall.progress(60, text="Ingesting full text…")
                results = run_ingest_with_progress(ids, force=False)
                show_ingest_results(results)
                clear_data_caches()

            overall.progress(85, text="Asking (LLM, often 10–20 s)…")
            log(f"Demo: ask {demo_q!r}")
            try:
                from rag.rag_pipeline import ask
            except ImportError:
                from src.rag.rag_pipeline import ask
            t0 = time.time()
            result = ask(demo_q, top_k=6, use_cache=False)
            log(f"Demo: ask done in {time.time()-t0:.1f}s coverage={result.get('coverage')}")
            st.session_state["last_ask"] = {"q": demo_q, "result": result}
            push_recent_question(demo_q)
            overall.progress(100, text="Demo complete")
            render_answer_block(demo_q, result, key_prefix="home_demo")
            clear_data_caches()
            log("=== Quick demo done ===")
        except Exception as e:
            st.markdown(friendly_error(e))
            log(traceback.format_exc()[-800:])


# ── LIBRARY ───────────────────────────────────────────────────────────────

with tab_lib:
    st.subheader("Literature Library & Discovery Workstation")
    st.caption(
        "Search, inspect, and explore the 5,700+ papers in your local catalog. Ingest full-text PDFs on-demand for grounded Q&A."
    )

    lib_nav = st.segmented_control(
        "Library View",
        ["Browse & Search Catalog", "Add New Papers to Library"],
        default="Browse & Search Catalog",
        key="lib_view_mode_segmented",
    )

    if lib_nav == "Browse & Search Catalog":
        from src.catalog import browse_catalog, get_catalog_stats

        c_stats = get_catalog_stats()

        # Catalog Metrics Header
        st.markdown("##### Catalog Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Catalog Papers", f"{c_stats['total_papers']:,}", help="Total metadata records in local SQLite catalog.")
        m2.metric("Full-Text in RAG", f"{c_stats['full_text_count']:,}", help="Papers with full Markdown parsed and chunked in Chroma.")
        m3.metric("Avg Citations", f"{c_stats['avg_citations']:.1f}", help="Average global citations across cited papers.")
        m4.metric("Year Range", f"{c_stats['min_year']} – {c_stats['max_year']}", help="Publication years present in catalog.")

        st.markdown("---")
        st.markdown("##### Filter & Search Library")

        f_col1, f_col2, f_col3 = st.columns([0.45, 0.25, 0.30])
        with f_col1:
            lib_q = st.text_input(
                "🔍 Search titles & abstracts",
                placeholder="e.g. transformers, attention, causal, circuits, NTK…",
                key="lib_search_kw",
            )
        with f_col2:
            cat_list = ["All Categories"] + c_stats.get("top_categories", [])
            sel_cat = st.selectbox("Category", cat_list, key="lib_filter_cat")
            cat_filter = None if sel_cat == "All Categories" else sel_cat
        with f_col3:
            sort_map = {
                "Most Cited": "citations_desc",
                "Newest First": "year_desc",
                "Relevance (BM25)": "relevance",
                "Title (A-Z)": "title_asc",
            }
            sel_sort = st.selectbox("Sort by", list(sort_map.keys()), index=0 if not lib_q else 2, key="lib_filter_sort")
            sort_val = sort_map[sel_sort]

        f_sub1, f_sub2, f_sub3 = st.columns([0.35, 0.35, 0.30])
        with f_sub1:
            ft_only = st.checkbox("Show full-text RAG papers only", value=False, key="lib_chk_ft_only")
        with f_sub2:
            min_cite_filter = st.slider("Minimum citations", 0, 500, 0, step=10, key="lib_slider_cites")
        with f_sub3:
            page_lim = st.selectbox("Papers per page", [10, 25, 50], index=0, key="lib_page_limit")

        with st.spinner("Fetching catalog papers…"):
            papers, total_matches = browse_catalog(
                limit=int(page_lim),
                offset=0,
                query=lib_q.strip() if lib_q else None,
                category=cat_filter,
                min_citations=min_cite_filter if min_cite_filter > 0 else None,
                full_text_only=ft_only,
                sort_by=sort_val,
            )

        st.markdown(f"**Found {total_matches:,} matching papers** (showing top {len(papers)})")

        if not papers:
            st.info("No papers match your search criteria. Try clearing filters or search terms.")
        else:
            for p in papers:
                aid = p["arxiv_id"]
                title = p["title"]
                year = p["year"] or "—"
                cites = p["citation_count"]
                cats = p.get("categories") or []
                cats_str = " · ".join(cats[:3]) if cats else "General"
                has_ft = p["has_full_text"]

                with st.container():
                    c_head, c_status = st.columns([0.75, 0.25])
                    with c_head:
                        st.markdown(f"#### [{title}](https://arxiv.org/abs/{aid})")
                        st.caption(f"**arXiv:** `{aid}` · **Year:** {year} · **Citations:** ⭐ {cites:,} · **Categories:** {cats_str}")
                    with c_status:
                        if has_ft:
                            st.success("✓ Full-Text Ready", icon="✅")
                        else:
                            st.info("○ Metadata Only", icon="📄")

                    if p.get("abstract"):
                        with st.expander("Abstract preview", expanded=False):
                            st.markdown(p["abstract"])

                    b_col1, b_col2, b_col3, b_col4 = st.columns([0.25, 0.25, 0.25, 0.25])
                    with b_col1:
                        if not has_ft:
                            if st.button("📥 Ingest for Q&A", key=f"btn_ingest_{aid}"):
                                with st.spinner(f"Downloading PDF & ingesting {aid}…"):
                                    try:
                                        from src.rag.ingest import ingest_papers
                                        ingest_papers([aid])
                                        clear_data_caches()
                                        st.success(f"Ingested {aid} into RAG!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Ingest failed: {e}")
                        else:
                            md_file = PROJECT_ROOT / "data" / "markdown" / f"{aid}.md"
                            if md_file.exists():
                                with st.expander("📖 Markdown excerpt"):
                                    try:
                                        st.markdown(md_file.read_text(encoding="utf-8")[:1800] + "\n\n*… [truncated]*")
                                    except Exception:
                                        st.caption("Could not load markdown.")

                    with b_col2:
                        if st.button("🔬 Novelty Subspace", key=f"btn_nov_{aid}", help="Pre-fill this paper in the Novelty & Graph Lab"):
                            st.session_state["novelty_target_id"] = aid
                            st.info(f"Target set to `{aid}`! Open the **Novelty & Graph** tab to compute its subspace projection.")

                    with b_col3:
                        if st.button("💬 Ask Question", key=f"btn_ask_{aid}", help="Ask questions about this paper"):
                            st.session_state["ask_prefill_query"] = f"What are the core methodology, findings, and limitations of {title}?"
                            st.info("Question draft prepared! Switch to the **Ask** tab to run grounded retrieval.")

                    with b_col4:
                        st.link_button("arXiv ↗", f"https://arxiv.org/abs/{aid}")

                    st.markdown("---")

    else:
        st.write(
            "Metadata for **ranking**. Full-text for **Ask** is a separate ingest step."
        )
        mode = st.radio(
            "How do you want to add papers?",
            [
                "Search arXiv",
                "Paste arXiv IDs",
                "Ingest from personal ranking",
                "Rebuild index only",
            ],
            horizontal=True,
            key="lib_mode",
        )
        with st.expander("Options", expanded=False):
            replace = st.checkbox(
                "Replace library instead of merging",
                value=False,
                key="lib_replace",
            )
            if replace:
                st.checkbox(
                    "I understand this replaces the current library",
                    value=False,
                    key="lib_replace_confirm",
                )
            build_idx = st.checkbox(
                "Rebuild recommend index after save", value=True, key="lib_build_idx"
            )

        def _guard_replace() -> bool:
            if replace and not st.session_state.get("lib_replace_confirm"):
                st.error("Confirm **Replace library** in Options before continuing.")
                return False
            return True

        if mode == "Search arXiv":
            st.caption(
                "Add one or more keyword blocks. All non-empty queries are searched "
                "and **merged** into the library in one run (duplicates by arXiv id skipped)."
            )
            with st.expander("Tips", expanded=False):
                st.markdown(
                    """
- **Max results** is per topic; ids are merged uniquely.  
- Rebuild the **recommend index** after large adds.  
- Rank ≠ Ask — **ingest** for full-text Q&A.
"""
                )
            if "lib_queries" not in st.session_state:
                st.session_state["lib_queries"] = [st.session_state.get("lib_q") or ""]

            queries = st.session_state["lib_queries"]
            remove_idx = None
            for qi in range(len(queries)):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    growing_text_area(
                        f"Keywords / topic {qi + 1}",
                        key=f"lib_q_{qi}",
                        default=queries[qi] if qi < len(queries) else "",
                        min_lines=2,
                        max_lines=12,
                        placeholder="e.g. mechanistic interpretability circuits",
                    )
                    # sync list from widget state
                    st.session_state["lib_queries"][qi] = st.session_state.get(f"lib_q_{qi}") or ""
                with cols[1]:
                    st.write("")
                    if len(queries) > 1 and st.button("✕", key=f"lib_q_rm_{qi}", help="Remove this topic"):
                        remove_idx = qi
            if remove_idx is not None:
                # drop widget keys and rebuild
                new_q = [st.session_state["lib_queries"][j] for j in range(len(queries)) if j != remove_idx]
                st.session_state["lib_queries"] = new_q or [""]
                for j in range(len(queries) + 2):
                    st.session_state.pop(f"lib_q_{j}", None)
                for j, val in enumerate(st.session_state["lib_queries"]):
                    st.session_state[f"lib_q_{j}"] = val
                st.rerun()

            b_add, b_sp = st.columns([0.35, 0.65])
            with b_add:
                if st.button("＋ Add another topic", key="lib_q_add"):
                    st.session_state["lib_queries"].append("")
                    st.rerun()

            c1, c2, c3 = st.columns(3)
            with c1:
                n = st.number_input(
                    "Max results per topic",
                    min_value=1,
                    max_value=50,
                    value=12,
                    step=1,
                    key="lib_n",
                    help="arXiv hits **per keyword block** (1–50).",
                )
            with c2:
                cat = st.text_input("Category", placeholder="cs.LG", key="lib_cat")
            with c3:
                sort = st.selectbox(
                    "Sort",
                    ["relevance", "lastUpdatedDate", "submittedDate"],
                    key="lib_sort",
                )
            if st.button("Search & add to library", type="primary", key="lib_search_btn"):
                if not _guard_replace():
                    pass
                else:
                    topics = [
                        (st.session_state.get(f"lib_q_{i}") or "").strip()
                        for i in range(len(st.session_state.get("lib_queries") or [""]))
                    ]
                    topics = [x for x in topics if x]
                    if not topics and not (cat or "").strip():
                        st.error("Enter at least one topic/keywords block (or a category).")
                    else:
                        if not topics:
                            topics = [""]  # category-only search once
                        log(f"Library search topics={topics!r} cat={cat!r}")
                        with st.status("Searching arXiv…", expanded=True) as st_status:
                            try:
                                from corpus_builder import search_arxiv

                                merged: dict[str, dict] = {}
                                for t_i, topic in enumerate(topics):
                                    st_status.write(
                                        f"Query {t_i+1}/{len(topics)}: {topic[:60]!r}…"
                                    )
                                    papers = search_arxiv(
                                        topic,
                                        max_results=int(n),
                                        category=(cat or None),
                                        sort_by=sort,
                                    )
                                    for p in papers or []:
                                        pid = (
                                            p.get("arxiv_id")
                                            or p.get("id")
                                            or p.get("paper_id")
                                            or ""
                                        )
                                        pid = str(pid).split("v")[0].strip()
                                        if pid and pid not in merged:
                                            merged[pid] = p
                                papers = list(merged.values())
                                size = save_library(papers, replace=replace)
                                st_status.write(
                                    f"Merged **{len(papers)}** unique papers · library size **{size}**"
                                )
                                st.success(
                                    f"Added/merged **{len(papers)}** unique papers · library **{size}**"
                                )
                                set_results_panel(
                                    "library",
                                    {
                                        "summary": f"Merged {len(papers)} unique papers from {len(topics)} topic(s).",
                                        "detail": papers[:30],
                                    },
                                )
                                log(f"Library search merged={len(papers)} library={size}")
                                st_status.update(state="complete")
                            except Exception as e:
                                st_status.update(state="error")
                                st.markdown(friendly_error(e))
                                papers = []
                        if build_idx and papers:
                            with st.status("Building recommend index…", expanded=True) as s2:
                                try:
                                    from corpus_builder import rebuild_index

                                    rebuild_index()
                                    s2.write("Index ready")
                                    s2.update(label="Index ready", state="complete")
                                    log("Recommend index rebuilt")
                                    st.success("Recommend index ready.")
                                except Exception as e:
                                    s2.update(label="Index failed", state="error")
                                    st.markdown(friendly_error(e))
                            clear_data_caches()
                            st.rerun()
                        elif papers:
                            clear_data_caches()
                            st.rerun()

        elif mode == "Paste arXiv IDs":
            ids_raw = st.text_area(
                "IDs (comma or newline separated)",
                placeholder="1806.07572\n2501.16496",
                height=100,
                key="lib_ids",
            )
            if st.button("Fetch & add", type="primary", key="lib_ids_btn"):
                if not _guard_replace():
                    pass
                else:
                    ids = [
                        x.strip()
                        for x in ids_raw.replace(",", "\n").splitlines()
                        if x.strip()
                    ]
                    if not ids:
                        st.error("Enter at least one ID.")
                    else:
                        log(f"Fetch IDs: {ids}")
                        with st.status("Fetching metadata…", expanded=True) as s:
                            try:
                                from corpus_builder import fetch_by_ids

                                s.write(f"Requesting {len(ids)} IDs from arXiv…")
                                papers = fetch_by_ids(ids)
                                size = save_library(papers, replace=replace)
                                s.write(f"Saved library size {size}")
                                st.success(
                                    f"Fetched **{len(papers)}** · library **{size}**"
                                )
                                log(f"Fetched {len(papers)} library={size}")
                                s.update(state="complete")
                            except Exception as e:
                                s.update(state="error")
                                st.markdown(friendly_error(e))
                                papers = []
                        if build_idx and papers:
                            with st.status("Building recommend index…", expanded=True) as s2:
                                try:
                                    from corpus_builder import rebuild_index

                                    s2.write("Embedding…")
                                    rebuild_index()
                                    s2.update(label="Index ready", state="complete")
                                    st.success("Recommend index ready.")
                                except Exception as e:
                                    s2.update(state="error")
                                    st.markdown(friendly_error(e))
                        clear_data_caches()
                        st.rerun()

        elif mode == "Ingest from personal ranking":
            st.write("Uses `rankings/personal_importance.csv` if present.")
            top_n = int_with_input(
                "Top-N",
                1,
                30,
                5,
                key="lib_rank_n",
                help="How many top personal-ranking papers to ingest (1–30).",
            )
            min_score = float_with_input(
                "Min personal score",
                1.0,
                5.0,
                4.0,
                0.5,
                key="lib_rank_min",
                help="Only papers at or above this personal score.",
            )
            if st.button("Ingest from personal ranking", type="primary", key="lib_rank_btn"):
                log(f"Ingest personal ranking top_n={top_n} min={min_score}")
                try:
                    try:
                        from rag.ingest import ids_from_personal_ranking
                    except ImportError:
                        from src.rag.ingest import ids_from_personal_ranking
                    ids = ids_from_personal_ranking(
                        top_n=int(top_n), min_score=float(min_score)
                    )
                    st.write("IDs:", ids)
                    results = run_ingest_with_progress(list(ids), force=False)
                    show_ingest_results(results)
                    clear_data_caches()
                except Exception as e:
                    st.markdown(friendly_error(e))

        else:
            st.write("Rebuild title/abstract vector index from current library.")
            if st.button("Rebuild recommend index", type="primary", key="lib_rebuild_btn"):
                log("Manual rebuild_index")
                with st.status("Building recommend index…", expanded=True) as s:
                    s.write("Loading model and embedding library texts…")
                    try:
                        from corpus_builder import rebuild_index

                        rebuild_index()
                        s.update(label="Index ready", state="complete")
                        st.success("Recommend index ready.")
                        log("rebuild_index ok")
                        clear_data_caches()
                    except Exception as e:
                        s.update(state="error")
                        st.markdown(friendly_error(e))


# ── RECOMMEND ─────────────────────────────────────────────────────────────

with tab_rec:
    st.subheader("Recommend papers")
    st.write(
        "Ranks the **library** (needs recommend index). Then ingest top hits for Ask."
    )
    interest = growing_text_area(
        "Research interest",
        key="rec_interest",
        default="",
        min_lines=3,
        max_lines=24,
        placeholder="Describe what you want from papers (topics, depth, constraints)…",
        help="Grows with your text. Long profiles are fine — results will not echo the full text.",
    )
    k = int_with_input(
        "Top-k",
        1,
        50,
        8,
        key="rec_k",
        help="How many ranked library papers to show (1–50).",
    )

    st.markdown("##### Ranking weights")
    preset_options = list(WEIGHT_PRESETS.keys()) + ["Custom"]
    # Initialize weight widget state from Balanced
    if "rec_w_embedding" not in st.session_state:
        base = WEIGHT_PRESETS["Balanced (default)"]
        for kk, vv in base.items():
            st.session_state[f"rec_w_{kk}"] = float(vv)
        st.session_state["rec_preset"] = "Balanced (default)"

    def _apply_preset_to_widgets() -> None:
        name = st.session_state.get("rec_preset")
        if name in WEIGHT_PRESETS and WEIGHT_PRESETS[name]:
            for kk, vv in WEIGHT_PRESETS[name].items():
                st.session_state[f"rec_w_{kk}"] = float(vv)

    preset_name = st.selectbox(
        "Weight preset",
        preset_options,
        key="rec_preset",
        on_change=_apply_preset_to_widgets,
        help="Pick a preset or edit numbers below for Custom.",
    )
    st.caption(PRESET_HELP.get(preset_name, ""))

    # Show current preset table for awareness
    with st.expander("Preset weight reference", expanded=False):
        ref_rows = []
        for name, pw in WEIGHT_PRESETS.items():
            ref_rows.append(
                {
                    "Preset": name,
                    "embedding": f"{pw['embedding']:.2f}",
                    "recency": f"{pw['recency']:.2f}",
                    "citation": f"{pw['citation']:.2f}",
                    "author": f"{pw['author']:.2f}",
                }
            )
        st.dataframe(ref_rows, hide_index=True, width="stretch")

    wcols = st.columns(4)
    labels = {
        "embedding": "Embedding",
        "recency": "Recency",
        "citation": "Citation",
        "author": "Author",
    }
    raw_w = {}
    for i, kk in enumerate(WEIGHT_KEYS):
        with wcols[i]:
            raw_w[kk] = st.number_input(
                labels[kk],
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=f"rec_w_{kk}",
                help=f"Weight for {kk} signal (0–1). Values are normalized before ranking.",
            )

    # Detect Custom vs matching preset from the four numbers (do not write selectbox key here)
    matched = _match_preset(raw_w)
    if matched is not None:
        weights = dict(WEIGHT_PRESETS[matched])
        st.success(f"Active weights match **{matched}**: `{_fmt_weights(weights)}`")
        if st.session_state.get("rec_preset") not in (matched,):
            st.caption(
                f"Numbers are the same as **{matched}**. "
                "You can set the dropdown to that preset for clarity."
            )
    else:
        weights = _normalize_weights(raw_w)
        st.info(
            f"**Custom** weights in use: `{_fmt_weights(weights)}` "
            f"(raw sum={sum(float(v) for v in raw_w.values()):.2f}; "
            "normalized so they sum to 1.00 for ranking)."
        )
        if st.session_state.get("rec_preset") != "Custom":
            st.caption(
                "These values no longer match the selected preset — ranking runs in **Custom** mode."
            )

    st.caption(f"Will rank with: **{_fmt_weights(weights)}**")
    with st.expander("What these weights mean", expanded=False):
        st.markdown(
            """
**Embedding** = topic fit · **Recency** = newer · **Citation** = highly cited · **Author** = light prior.

Change a number → **Custom** (normalized to 1.00). Preset table: **Help**.
"""
        )


    if st.button("Rank papers", type="primary", key="rec_btn"):
        if not (interest or "").strip():
            st.error("Enter a research interest.")
        elif lib_n == 0:
            st.error("Library is empty — use the Library tab first.")
        elif not cs.get("chroma_exists"):
            st.error("Recommend index not built — Library → Rebuild index only.")
        else:
            ranked = run_rank(interest.strip(), int(k), weights)
            if ranked:
                st.session_state["last_recommend"] = {
                    "interest": interest.strip(),
                    "ranked": ranked,
                    "weights": weights,
                    "weights_label": _fmt_weights(weights),
                }
                rows = []
                for r in ranked:
                    rows.append(
                        {
                            "id": paper_id_from_row(r),
                            "title": r.get("title") or "",
                            "score": r.get("final_score", r.get("score", "")),
                        }
                    )
                set_results_panel(
                    "rank",
                    {"n": len(rows), "k": int(k), "rows": rows},
                )
                st.rerun()

    prev = st.session_state.get("last_recommend")
    ids_order: list[str] = []
    if prev and prev.get("ranked"):
        ranked = prev["ranked"]
        for r in ranked:
            pid = paper_id_from_row(r)
            if pid:
                ids_order.append(pid)

        # Multi-signal Score Decomposition Chart
        if ranked:
            try:
                from src.novelty.viz import render_ranking_breakdown_chart

                rank_chart = render_ranking_breakdown_chart(ranked, top_k=min(8, len(ranked)))
                if rank_chart is not None:
                    st.altair_chart(rank_chart, width="stretch")
            except Exception as e:
                st.caption(f"Score decomposition note: {e}")

        # Always render the single results panel (rank list)
        if st.session_state.get("results_panel", {}).get("kind") == "rank":
            render_results_panel()
        else:
            # Fallback short header if panel not set
            st.caption(f"Last rank: **{len(ids_order)}** papers")
            with st.expander(f"Paper list ({len(ids_order)})", expanded=False):
                for i, r in enumerate(ranked, 1):
                    pid = paper_id_from_row(r)
                    title = (r.get("title") or "")[:100]
                    score = r.get("final_score", r.get("score", ""))
                    if pid:
                        st.markdown(
                            f"**{i}. [{pid}](https://arxiv.org/abs/{pid})**: {title}"
                        )
                    else:
                        st.markdown(f"**{i}.** {title}")
                    st.caption(f"Score: {score}")

        st.markdown("#### Ingest for Q&A")
        st.write(
            "Downloads PDFs, converts to Markdown, chunks, embeds into RAG. "
            "You can ingest any number up to the current ranked list."
        )
        with st.expander("Time & RAM", expanded=False):
            st.markdown(
                """
~**30–90s**/paper first time (OCR can be slower).  
On 16 GB RAM prefer **5–10**/batch or **Low memory mode**.  
Progress = i/N + ETA. Force re-download only to refresh files.
"""
            )
        max_n = max(1, len(ids_order))
        ingest_n = int_with_input(
            "Ingest top-N",
            1,
            max_n,
            min(5, max_n),
            key="rec_ingest_n",
            help=f"How many ranked hits to fully ingest (1–{max_n}, limited by this rank run).",
        )
        if ingest_n >= 10:
            st.warning(
                f"**{ingest_n}** papers ≈ **{_fmt_duration(45 * ingest_n)}–{_fmt_duration(90 * ingest_n)}** "
                "and high RAM use with EmbeddingGemma. Prefer batches of 5–10 if memory is tight."
            )
        force = st.checkbox("Force re-download / re-embed", value=False, key="rec_force")
        low_mem = st.checkbox(
            "Low memory mode",
            value=False,
            key="rec_low_mem",
            help="Smaller embed batches + aggressive cache release between papers (slower, safer on 16 GB).",
        )
        if st.button(
            f"Ingest top {ingest_n} for Q&A",
            type="primary",
            key="rec_ingest_btn",
        ):
            to_ingest = [x for x in ids_order[:ingest_n] if x]
            results = run_ingest_with_progress(
                to_ingest, force=force, low_memory=low_mem
            )
            show_ingest_results(results)
            clear_data_caches()
            st.rerun()




# ── ASK ───────────────────────────────────────────────────────────────────

with tab_ask:
    st.subheader("Ask the papers")

    if rag_n == 0:
        st.warning(
            "**No full-text papers yet.** Ranking alone is not enough for Q&A."
        )
        st.markdown(
            """
**Next steps**
1. Open **Recommend**, enter your research interest, rank papers  
2. Click **Ingest for Q&A** on the top results  
3. Return here to ask with citations  
"""
        )
        st.info("Or use **Home → Quick demo** for a guided first run.")
    else:
        st.caption(
            f"**{rag_n}** full-text papers · answers usually **10–20 s**. "
            "Editing the question does **not** query the index until you press **Ask**."
        )

    recent = st.session_state.get("recent_questions") or []
    if recent and rag_n > 0:
        st.markdown("**Recent questions**")
        rcols = st.columns(min(len(recent), 4))
        for i, rq in enumerate(recent[:4]):
            short = rq[:40] + ("…" if len(rq) > 40 else "")
            if rcols[i].button(short, key=f"recent_{i}"):
                st.session_state["ask_q"] = rq
                st.rerun()

    paper_filter: list[str] = []
    scope_labels: list[str] = []
    top_k = 6
    use_cache = True

    if rag_n > 0:
        with st.expander("Paper filter & retrieval options", expanded=True):
            # Rebuild catalog if session cache has no titles (broken fast-path)
            catalog = ingested_papers_catalog(rs)
            titled = sum(1 for c in catalog if (c.get("title") or "").strip())
            if catalog and titled < max(3, len(catalog) // 20):
                st.caption(
                    f"Paper list has {len(catalog)} ids but few titles ({titled}). "
                    "Refresh to reload titles from corpus.json."
                )
            c_ref, c_info = st.columns([1, 3])
            with c_ref:
                if st.button("Refresh paper list", key="ask_refresh_catalog"):
                    clear_data_caches()
                    st.session_state.pop("papers_catalog", None)
                    try:
                        rag_papers_catalog_cached.clear()
                    except Exception:
                        pass
                    st.rerun()
            with c_info:
                st.caption("Use after ingest if search finds nothing.")

            if not catalog and int(rs.get("chunk_count") or 0) > 0:
                st.warning(
                    "Paper list is empty but chunks exist — click **Refresh paper list**."
                )

            paper_filter = paper_scope_picker(catalog, state_key="ask_scope_ids")
            if paper_filter:
                id_to_label = {c["paper_id"]: c["label"] for c in catalog}
                scope_labels = [id_to_label.get(i, i) for i in paper_filter]
            else:
                scope_labels = []

            st.divider()
            st.markdown("**Retrieval**")
            top_k = int_with_input(
                "Passages (top-k)",
                1,
                20,
                6,
                key="ask_k",
                help="Chunks sent to the LLM (1–20). Default 6.",
            )
            use_cache = st.checkbox(
                "Use semantic answer cache",
                value=True,
                key="ask_cache",
                help="On = reuse similar questions. Off = always call the LLM.",
            )
            use_hybrid = st.checkbox(
                "Hybrid search (BM25 + Dense)",
                value=True,
                key="ask_hybrid",
                help="Fuses BM25 lexical keyword matching with dense semantic embeddings via RRF and boosts exact arXiv IDs.",
            )

        with st.form("ask_form", clear_on_submit=False):
            question = st.text_area(
                "Question",
                value=st.session_state.get("ask_prefill_query", ""),
                height=100,
                key="ask_q",
                placeholder="e.g. What is circuit discovery in mechanistic interpretability?",
            )
            ask_clicked = st.form_submit_button(
                "Ask",
                type="primary",
                disabled=rag_n == 0,
            )
    else:
        question = ""
        ask_clicked = False

    if st.session_state.get("last_ask") and rag_n > 0:
        if st.button("Clear last answer", key="ask_clear"):
            st.session_state["last_ask"] = None
            st.rerun()

    if ask_clicked and (question or "").strip() and rag_n > 0:
        log(
            f"Ask: {question.strip()!r} top_k={top_k} cache={use_cache} "
            f"filter={paper_filter or 'all'}"
        )
        progress = st.progress(0, text="Starting…")
        status = st.empty()
        try:
            status.markdown("**1/3 Retrieving** relevant chunks…")
            progress.progress(15, text="Retrieving…")
            try:
                from rag.rag_pipeline import ask
            except ImportError:
                from src.rag.rag_pipeline import ask
            status.markdown("**2/3 Generating** with Gemini (often 10–20 s)…")
            progress.progress(40, text="Calling LLM…")
            t0 = time.time()
            result = ask(
                question.strip(),
                top_k=top_k,
                paper_ids=paper_filter or None,
                use_cache=use_cache,
                hybrid=use_hybrid,
            )
            elapsed = time.time() - t0
            log(
                f"Ask done in {elapsed:.1f}s coverage={result.get('coverage')} "
                f"cache_hit={result.get('cache_hit')}"
            )
            progress.progress(100, text=f"Done in {elapsed:.1f}s")
            status.success(f"Finished in **{elapsed:.1f}s**")
            st.session_state["last_ask"] = {
                "q": question.strip(),
                "result": result,
                "scope_ids": list(paper_filter),
                "scope_labels": list(scope_labels),
                "corpus_n": rag_n,
            }
            push_recent_question(question.strip())
            render_answer_block(
                question.strip(),
                result,
                key_prefix="ask_live",
                scope_ids=paper_filter or None,
                scope_labels=scope_labels or None,
                corpus_paper_count=None if paper_filter else rag_n,
            )
        except Exception as e:
            progress.empty()
            status.empty()
            st.markdown(friendly_error(e))
            log(traceback.format_exc()[-800:])

    elif st.session_state.get("last_ask"):
        st.markdown("#### Last answer")
        la = st.session_state["last_ask"]
        st.caption(la.get("q"))
        render_answer_block(
            la["q"],
            la["result"],
            key_prefix="ask_last",
            scope_ids=la.get("scope_ids") or None,
            scope_labels=la.get("scope_labels") or None,
            corpus_paper_count=(
                None if la.get("scope_ids") else la.get("corpus_n") or rag_n
            ),
        )


@st.cache_data(ttl=300)
def _get_citation_library_hubs(top_n: int = 25) -> tuple[dict, list[dict]]:
    try:
        from src.citation_library import store
    except ImportError:
        from citation_library import store
    conn = store.connect()
    try:
        st_data = store.stats(conn)
        counts = store.citation_counts(conn)
        rows = []
        for r in counts[:top_n]:
            year_val = r["year"]
            year_str = str(year_val).strip() if (year_val is not None and str(year_val).strip().isdigit()) else "—"
            c_global = r["global_citation_count"]
            rows.append({
                "Title": str(r["title"] or "Untitled")[:120],
                "Paper ID": str(r["arxiv_id"] or r["canonical_id"]),
                "Year": year_str,
                "In-Library Citers": int(r["library_cite_count"] or 0),
                "Global Citations": int(c_global or 0),
            })
        return st_data, rows
    finally:
        conn.close()


def _get_example_citers_for(target_id: str, limit: int = 10) -> list[str]:
    try:
        from src.citation_library import store
    except ImportError:
        from citation_library import store
    conn = store.connect()
    try:
        return store.example_citers(conn, target_id, limit=limit)
    finally:
        conn.close()


# ── NOVELTY & GRAPH (LAB) ──────────────────────────────────────────────────

with tab_nov:
    st.subheader("Novelty analysis & citation network (offline lab)")
    st.caption(
        "Explore citation graphs and compute reference-relative geometric novelty "
        "profiles. Cleanly isolated from ConfAI poster ranking index."
    )

    nov_mode = st.segmented_control(
        "Lab view",
        ["Novelty scoring", "Dual-paper comparison", "Citation network hubs"],
        default="Novelty scoring",
        key="nov_lab_view_selector",
    )

    if nov_mode == "Novelty scoring":
        st.markdown(
            "Quantifies whether a paper is an **extension** of prior work or an "
            "**orthogonal blend**, measured by projecting its embedding vector onto the subspace "
            "spanned by its cited references."
        )

        # Curated Demo Showcase Archetypes
        with st.container(border=True):
            st.markdown("#### 🎯 Curated Demo Archetypes")
            st.caption(
                "Select a canonical literature pattern to inspect its empirical geometry and subspace projection instantly:"
            )
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                if st.button(
                    "🌟 High Synergy Blend\n\n(Grad-CAM++)",
                    key="arch_blend",
                    help="1710.11063: Synthesizes multiple visual explanation lineage roots",
                    width="stretch",
                ):
                    st.session_state["novelty_target_id"] = "1710.11063"
                    st.session_state["trigger_nov_compute"] = True
                    st.rerun()
            with b_col2:
                if st.button(
                    "📐 Linear Extension\n\n(Facial Attributes)",
                    key="arch_ext",
                    help="1801.02480: Direct follow-up anchored closely to a single parent",
                    width="stretch",
                ):
                    st.session_state["novelty_target_id"] = "1801.02480"
                    st.session_state["trigger_nov_compute"] = True
                    st.rerun()
            with b_col3:
                if st.button(
                    "🚀 Paradigm Shift\n\n(GCN Outlier)",
                    key="arch_outlier",
                    help="1609.02907: Semi-Supervised Classification with Graph Convolutional Networks",
                    width="stretch",
                ):
                    st.session_state["novelty_target_id"] = "1609.02907"
                    st.session_state["trigger_nov_compute"] = True
                    st.rerun()
            with b_col4:
                if st.button(
                    "🛡️ Reference-Starved\n\n(Diagnostic Guard)",
                    key="arch_starved",
                    help="0801.1728: Sparse reference edge case triggering starvation guard",
                    width="stretch",
                ):
                    st.session_state["novelty_target_id"] = "0801.1728"
                    st.session_state["trigger_nov_compute"] = True
                    st.rerun()

        default_aid = st.session_state.get("novelty_target_id", "1806.07572")
        with st.form("nov_score_form"):
            col_id, col_opt = st.columns([2, 3])
            with col_id:
                aid_input = st.text_input(
                    "arXiv paper ID",
                    value=default_aid,
                    placeholder="e.g. 1806.07572, 2404.14082",
                    help="Paper to score relative to its references.",
                    key="nov_aid_input",
                )
            with col_opt:
                allow_part = st.checkbox(
                    "Allow partial abstract coverage",
                    value=True,
                    help="Falls back to reference titles if <70% of references have abstracts available.",
                    key="nov_allow_part",
                )
            submit_score = st.form_submit_button(
                "Compute novelty profile", type="primary"
            )

        trigger_auto = st.session_state.pop("trigger_nov_compute", False)
        target_aid = paper_id_from_row({"id": aid_input or ""}) if (submit_score or trigger_auto) else None

        if target_aid:
            log(f"Novelty: scoring {target_aid} (allow_partial={allow_part})")
            with st.spinner(f"Resolving references and computing embedding projection for {target_aid}…"):
                try:
                    from src.novelty.score import score_paper
                    prof = score_paper(target_aid, allow_partial=allow_part)
                except Exception as e:
                    st.markdown(friendly_error(e))
                    prof = None
            if prof:
                st.session_state["nov_last_prof"] = prof
                st.session_state["nov_last_aid"] = target_aid
        else:
            prof = st.session_state.get("nov_last_prof")
            target_aid = st.session_state.get("nov_last_aid", default_aid)

        if prof:
            if prof.get("ok"):
                st.success(f"Novelty profile computed for `{target_aid}`")
                typ = prof.get("classification") or prof.get("typology_hint", "unknown")
                conf = prof.get("confidence")
                typ_lower = typ.lower()
                if "starved" in typ_lower or "uncertain" in typ_lower:
                    badge_color = "red"
                elif "blend" in typ_lower or "synergy" in typ_lower:
                    badge_color = "violet"
                elif "extension" in typ_lower or "linear" in typ_lower:
                    badge_color = "blue"
                elif "outlier" in typ_lower or "paradigm" in typ_lower:
                    badge_color = "amber"
                else:
                    badge_color = "gray"

                badge_label = typ.replace("_", " ").title()
                if conf:
                    badge_label = f"{badge_label} · {str(conf).title()} Confidence"
                st.badge(badge_label, icon=":material/psychology:", color=badge_color)

                warnings = prof.get("quality_warnings") or []
                if warnings:
                    for w in warnings:
                        st.warning(f"**Geometric Quality Guard:** {w}")
                elif prof.get("coverage_warning"):
                    st.warning(f"**Coverage Notice:** {prof['coverage_warning']}")

                m_val = float(prof.get("m") or 0.0)
                delta_val = float(prof.get("Delta") or 0.0)
                r2_val = float(prof.get("R2_plus") or 0.0)
                focal_y = float(math.sqrt(max(0.0, delta_val)))
                viz_data = prof.get("viz_2d")
                res_dist = viz_data.get("metrics", {}).get("residual_distance") if viz_data and viz_data.get("metrics") else float(max(0.0, 1.0 - r2_val) ** 0.5)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Direct Similarity (m)",
                    f"{m_val:.3f}",
                    help="Cosine similarity to single closest foundational parent reference.",
                )
                c2.metric(
                    "Synergy Blend (Δ)",
                    f"{delta_val:.3f}",
                    help="Multi-parent combination gain: R²₊ - m².",
                )
                c3.metric(
                    "Cone Variance (R²₊)",
                    f"{r2_val:.3f}",
                    help="Total fraction of focal paper representation explained by non-negative reference cone.",
                )
                c4.metric(
                    "Novelty Residual (√1-R²₊)",
                    f"{res_dist:.3f}",
                    help="Radial distance outside prior art references to unit horizon ‖x‖ = 1.",
                )

                # Mathematical Decomposition Card
                with st.container(border=True):
                    st.markdown("#### 🔬 Moreau-Cone Geometric Decomposition")
                    dom_id = (viz_data.get("metrics", {}).get("dominant_parent_id") if viz_data else None) or prof.get("dominant_parent_id") or "Dominant Ref"
                    st.latex(
                        rf"\mathbf{{x}}_{{\text{{focal}}}} \approx {m_val:.3f}\,\mathbf{{e}}_1 + {focal_y:.3f}\,\mathbf{{e}}_2 + {res_dist:.3f}\,\mathbf{{e}}_{{\perp}}"
                    )
                    st.caption(
                        rf"Orthonormal coordinates: dominant parent axis $\mathbf{{e}}_1$ ({dom_id}), "
                        rf"orthogonal synergy dimension $\mathbf{{e}}_2$ ($\sqrt{{\Delta}} = {focal_y:.3f}$), "
                        rf"and out-of-cone novelty residual $\mathbf{{e}}_{{\perp}}$ ($\sqrt{{1 - R^2_+}} = {res_dist:.3f}$)."
                    )

                # Interactive Layer Controls
                st.markdown("##### Visual Subspace Controls")
                t_c1, t_c2, t_c3, t_c4 = st.columns(4)
                with t_c1:
                    show_cone = st.checkbox("Reference cone rays", value=True, key="nov_toggle_cone")
                with t_c2:
                    show_horizon = st.checkbox("Unit sphere horizon", value=True, key="nov_toggle_horizon")
                with t_c3:
                    show_zones = st.checkbox("Discovery zone labels", value=True, key="nov_toggle_zones")
                with t_c4:
                    show_callouts = st.checkbox("On-plot callouts", value=True, key="nov_toggle_callouts")

                if viz_data and viz_data.get("ok"):
                    try:
                        from src.novelty.viz import render_altair_subspace_chart

                        chart = render_altair_subspace_chart(
                            viz_data,
                            show_cone=show_cone,
                            show_labels=show_callouts,
                            show_zones=show_zones,
                            width="stretch",
                        )
                        if chart is not None:
                            st.altair_chart(chart, width="stretch")
                    except Exception as e:
                        st.caption(f"Subspace chart note: {e}")

                with st.expander("💡 How to Read This Subspace & Metrics", expanded=False):
                    st.markdown(
                        r"""
- **Direct Similarity ($m$):** How closely the paper mirrors its single most influential cited parent.
- **Synergy Blend ($\Delta$):** The non-linear synthesis gain ($R^2_+ - m^2$) from combining multiple distinct lineages.
- **Cone Variance ($R^2_+$):** The total fraction of the paper's representation accounted for by prior art references.
- **Novelty Residual ($\sqrt{1 - R^2_+}$):** The radial distance outside the reference cone to the unit circle $\|x\|=1$. Higher values indicate greater creative departure from existing literature.
- **Dominant Parent ($e_1$):** Direction of the closest foundational paper placed at $(1, 0)$.
- **Blend Direction ($e_2$):** Orthogonal dimension capturing cross-reference synthesis.
                        """
                    )

                stats = prof.get("stats") or {}
                n_refs_total = stats.get("n_refs") or stats.get("n_raw", 0)
                n_with_abs = stats.get("n_with_abstract") or stats.get("n_abstracts", 0)
                n_emb = stats.get("n_refs_embedded") or stats.get("n_total", 0)
                d_eff_val = prof.get("d_eff_refs") or prof.get("d_eff_cache") or prof.get("d_eff", "n/a")
                frac_abs = stats.get("frac_with_abstract", 0.0)

                with st.expander("Reference & embedding details", expanded=False):
                    st.markdown(
                        f"- **Total references resolved:** {n_refs_total}\n"
                        f"- **References embedded:** {n_emb}\n"
                        f"- **Abstracts available:** {n_with_abs} ({frac_abs:.1%})\n"
                        f"- **Embedding model:** `{prof.get('model', 'default')}` (dim={prof.get('dim', 'auto')})\n"
                        f"- **Effective dimensionality ($d_{{eff}}$):** {f'{d_eff_val:.2f}' if isinstance(d_eff_val, (int, float)) else d_eff_val}"
                    )

                attr_list = prof.get("attribution") or []
                if attr_list:
                    with st.expander("Foundational parent references (attribution weights)", expanded=True):
                        st.caption("Top references explaining the focal paper's placement in the embedding cone (NNLS weights):")
                        attr_rows = []
                        missing_parents = []
                        for item in attr_list[:12]:
                            rid = item.get("paper_id") or item.get("ref_id") or item.get("canonical_id") or ""
                            title = item.get("title") or rid
                            aid_clean = rid.replace("arxiv:", "") if "arxiv:" in rid else (rid if re.match(r"^\d{4}\.\d{4,5}", rid) else "")
                            in_rag_status = "In RAG" if aid_clean and (PROJECT_ROOT / "data" / "markdown" / f"{aid_clean}.md").exists() else "Missing"
                            if in_rag_status == "Missing" and aid_clean:
                                missing_parents.append(aid_clean)
                            url = item.get("url") or (f"https://arxiv.org/abs/{aid_clean}" if aid_clean else f"https://www.semanticscholar.org/paper/{rid.replace('s2:', '')}")
                            provider = item.get("provider") or ("arXiv" if aid_clean else "Semantic Scholar")
                            display_id = (aid_clean if aid_clean else (rid[:16] + "…" if len(rid) > 20 else rid))
                            share_val = float(item.get("share") or 0.0)
                            weight_val = float(item.get("weight") or 0.0)
                            attr_rows.append({
                                "Reference Title": title,
                                "Source": provider,
                                "Paper ID": display_id,
                                "Weight (λ)": round(weight_val, 4),
                                "Attribution Share": round(share_val, 4),
                                "Paper Link": url,
                                "Full-Text Status": in_rag_status,
                            })
                        st.dataframe(
                            attr_rows,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Paper Link": st.column_config.LinkColumn("Paper Link", display_text="Open ↗"),
                                "Attribution Share": st.column_config.ProgressColumn(
                                    "Attribution Share",
                                    format="%.1f%%",
                                    min_value=0.0,
                                    max_value=1.0,
                                ),
                                "Weight (λ)": st.column_config.NumberColumn("Weight (λ)", format="%.4f"),
                            },
                        )

                        if missing_parents:
                            if st.button(f"📥 Ingest {len(missing_parents)} missing foundational parents into RAG", key="ingest_missing_parents_btn"):
                                with st.spinner(f"Ingesting foundational parents: {missing_parents}…"):
                                    from src.rag.ingest import ingest_papers
                                    res_parents = ingest_papers(missing_parents)
                                    st.success(f"Ingested {len(res_parents)} parent papers into RAG!")
                                    clear_data_caches()
                                    st.rerun()
            else:
                st.error(
                    prof.get("message")
                    or prof.get("error")
                    or "Could not compute novelty profile."
                )
                stats = prof.get("stats") or {}
                if stats:
                    st.caption(
                        f"References found: {stats.get('n_refs', 0)} "
                        f"(with abstract: {stats.get('n_with_abstract', 0)})"
                    )

    elif nov_mode == "Dual-paper comparison":
        st.markdown(
            "Compare two research papers directly in geometric subspace. Visualizes their "
            "evolution trajectory vector, synergy shift, and shared vs unique reference lineages."
        )

        preset_pairs = {
            "Grad-CAM vs Grad-CAM++ (Synergy Evolution)": ("1610.02391", "1710.11063"),
            "ResNet vs DenseNet (Architecture Evolution)": ("1512.03385", "1608.06993"),
            "Word2Vec vs GloVe (Embedding Formulation Shift)": ("1301.3781", "1405.3531"),
            "Custom Paper Pair": ("", ""),
        }

        chosen_pair_key = st.selectbox(
            "Comparison Archetypes",
            list(preset_pairs.keys()),
            key="comp_preset_select",
        )
        def_a, def_b = preset_pairs[chosen_pair_key]

        with st.form("dual_paper_compare_form"):
            c_a, c_b = st.columns(2)
            with c_a:
                aid_a_in = st.text_input(
                    "Paper A arXiv ID (Baseline)",
                    value=def_a or "1610.02391",
                    placeholder="e.g. 1610.02391",
                    key="comp_aid_a",
                )
            with c_b:
                aid_b_in = st.text_input(
                    "Paper B arXiv ID (Comparison)",
                    value=def_b or "1710.11063",
                    placeholder="e.g. 1710.11063",
                    key="comp_aid_b",
                )
            allow_part_comp = st.checkbox(
                "Allow partial abstract coverage for references",
                value=True,
                key="comp_allow_part",
            )
            submit_comp = st.form_submit_button("Compare Subspace Trajectories", type="primary")

        if submit_comp:
            aid_a = paper_id_from_row({"id": aid_a_in or ""})
            aid_b = paper_id_from_row({"id": aid_b_in or ""})
            if not aid_a or not aid_b:
                st.warning("Please provide valid arXiv IDs for both papers.")
            else:
                with st.spinner(f"Scoring both papers ({aid_a} and {aid_b}) and computing comparative trajectory…"):
                    try:
                        from src.novelty.score import score_paper
                        prof_a = score_paper(aid_a, allow_partial=allow_part_comp)
                        prof_b = score_paper(aid_b, allow_partial=allow_part_comp)
                        st.session_state["comp_last_prof"] = {
                            "prof_a": prof_a,
                            "prof_b": prof_b,
                            "aid_a": aid_a,
                            "aid_b": aid_b,
                        }
                    except Exception as e:
                        st.markdown(friendly_error(e))

        cached_comp = st.session_state.get("comp_last_prof")
        if cached_comp:
            pa = cached_comp["prof_a"]
            pb = cached_comp["prof_b"]
            aid_a = cached_comp["aid_a"]
            aid_b = cached_comp["aid_b"]

            if pa and pb and pa.get("ok") and pb.get("ok"):
                title_a = (pa.get("title") or aid_a)[:40]
                title_b = (pb.get("title") or aid_b)[:40]
                proj_a = pa.get("viz_2d") or {}
                proj_b = pb.get("viz_2d") or {}

                try:
                    from src.novelty.viz import render_altair_comparison_chart
                    comp_chart = render_altair_comparison_chart(
                        proj_a,
                        proj_b,
                        title_a=f"{aid_a}: {title_a}",
                        title_b=f"{aid_b}: {title_b}",
                        height=500,
                    )
                    if comp_chart is not None:
                        st.altair_chart(comp_chart, width="stretch")
                except Exception as e:
                    st.caption(f"Comparison chart note: {e}")

                # Comparative Delta Metrics Table
                st.markdown("#### 📊 Comparative Metric Trajectory")
                mA = float(pa.get("m") or 0.0)
                mB = float(pb.get("m") or 0.0)
                deltaA = float(pa.get("Delta") or 0.0)
                deltaB = float(pb.get("Delta") or 0.0)
                r2A = float(pa.get("R2_plus") or 0.0)
                r2B = float(pb.get("R2_plus") or 0.0)
                resA = max(0.0, 1.0 - r2A) ** 0.5
                resB = max(0.0, 1.0 - r2B) ** 0.5
                typA = pa.get("classification") or pa.get("typology_hint", "unknown")
                typB = pb.get("classification") or pb.get("typology_hint", "unknown")
                refsA = (pa.get("stats") or {}).get("n_refs", 0)
                refsB = (pb.get("stats") or {}).get("n_refs", 0)

                delta_rows = [
                    {
                        "Geometric Metric": "Direct Anchor Similarity (m)",
                        f"Paper A ({aid_a})": f"{mA:.3f}",
                        f"Paper B ({aid_b})": f"{mB:.3f}",
                        "Evolution Shift (B - A)": f"{mB - mA:+.3f}",
                    },
                    {
                        "Geometric Metric": "Synergy Blend Gain (Δ)",
                        f"Paper A ({aid_a})": f"{deltaA:.3f}",
                        f"Paper B ({aid_b})": f"{deltaB:.3f}",
                        "Evolution Shift (B - A)": f"{deltaB - deltaA:+.3f}",
                    },
                    {
                        "Geometric Metric": "Cone Variance (R²₊)",
                        f"Paper A ({aid_a})": f"{r2A:.3f}",
                        f"Paper B ({aid_b})": f"{r2B:.3f}",
                        "Evolution Shift (B - A)": f"{r2B - r2A:+.3f}",
                    },
                    {
                        "Geometric Metric": "Novelty Residual (√1-R²₊)",
                        f"Paper A ({aid_a})": f"{resA:.3f}",
                        f"Paper B ({aid_b})": f"{resB:.3f}",
                        "Evolution Shift (B - A)": f"{resB - resA:+.3f}",
                    },
                    {
                        "Geometric Metric": "Resolved References",
                        f"Paper A ({aid_a})": str(refsA),
                        f"Paper B ({aid_b})": str(refsB),
                        "Evolution Shift (B - A)": f"{refsB - refsA:+d}",
                    },
                    {
                        "Geometric Metric": "Typology Classification",
                        f"Paper A ({aid_a})": typA.replace("_", " ").title(),
                        f"Paper B ({aid_b})": typB.replace("_", " ").title(),
                        "Evolution Shift (B - A)": "Shift" if typA != typB else "Identical",
                    },
                ]
                st.dataframe(delta_rows, width="stretch", hide_index=True)

                # Reference Lineage Overlap Analysis
                st.markdown("#### 🔗 Cited Literature Lineage Overlap")
                set_a = set()
                for item in (pa.get("attribution") or []):
                    rid = item.get("paper_id") or item.get("ref_id") or item.get("canonical_id")
                    if rid:
                        set_a.add(str(rid))
                set_b = set()
                for item in (pb.get("attribution") or []):
                    rid = item.get("paper_id") or item.get("ref_id") or item.get("canonical_id")
                    if rid:
                        set_b.add(str(rid))

                common_refs = sorted(set_a.intersection(set_b))
                unique_a = sorted(set_a - set_b)
                unique_b = sorted(set_b - set_a)

                o1, o2, o3 = st.columns(3)
                o1.metric("Shared Foundational Parents", f"{len(common_refs)}")
                o2.metric(f"Unique to {aid_a}", f"{len(unique_a)}")
                o3.metric(f"Unique to {aid_b}", f"{len(unique_b)}")

                if common_refs:
                    with st.expander(f"Shared Literature Ancestors ({len(common_refs)})", expanded=False):
                        for cr in common_refs:
                            clean_cr = cr.replace("arxiv:", "")
                            st.markdown(f"- [`{cr}`](https://arxiv.org/abs/{clean_cr})" if clean_cr.replace(".", "").isdigit() else f"- `{cr}`")
            else:
                if pa and not pa.get("ok"):
                    st.error(f"Paper A ({aid_a}) error: {pa.get('error') or pa.get('message')}")
                if pb and not pb.get("ok"):
                    st.error(f"Paper B ({aid_b}) error: {pb.get('error') or pb.get('message')}")

    elif nov_mode == "Citation network hubs":
        st.markdown(
            "Inspect the top cited landmark papers and citation density across "
            "the local citation graph database (`data/citation_library.db`)."
        )
        try:
            hub_stats, hub_rows = _get_citation_library_hubs(top_n=30)
            k1, k2, k3 = st.columns(3)
            k1.metric("Cached papers", f"{hub_stats.get('papers_cached', 0):,}")
            k2.metric("Citation edges", f"{hub_stats.get('edges', 0):,}")
            k3.metric("Completed fetches", f"{hub_stats.get('fetch_ok', 0):,}")

            st.markdown("#### Top landmark papers in local corpus")
            try:
                st.dataframe(hub_rows, width="stretch", hide_index=True)
            except TypeError:
                st.dataframe(hub_rows, use_container_width=True, hide_index=True)

            with st.expander("Inspect citing papers for a landmark", expanded=False):
                inspect_id = st.text_input(
                    "Landmark arXiv ID or S2 ID",
                    value=hub_rows[0]["Paper ID"] if hub_rows else "1706.03762",
                    key="nov_hub_inspect_id",
                )
                if st.button("Find citing papers", key="nov_hub_find_citers"):
                    clean_id = paper_id_from_row({"id": inspect_id}) or inspect_id.strip()
                    citers = _get_example_citers_for(clean_id, limit=15)
                    if citers:
                        st.markdown(f"**Found {len(citers)} citing papers in local graph:**")
                        for cid in citers:
                            st.markdown(f"- [`{cid}`](https://arxiv.org/abs/{cid})")
                    else:
                        st.info(f"No in-library citers recorded yet for `{inspect_id}`.")
        except Exception as e:
            st.warning("Citation library database is not initialized yet or empty.")
            st.caption(f"Detail: {e}")


with tab_help:
    st.subheader("Interactive Research Guide & System Architecture")
    st.markdown(
        """
This platform integrates **Literature Discovery**, **Multi-Signal Ranking**, **Citation-Grounded Hybrid RAG**, and **Geometric Novelty Subspace Modeling** into a closed-loop research workflow.

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│       1. DISCOVER       │      │        2. RANK          │      │        3. INGEST        │      │         4. ASK          │
│ Browse 5,700+ Catalog   │ ──►  │ Multi-Signal Weighting  │ ──►  │ Selective PDF Download │ ──►  │ Citation-Grounded Q&A   │
│ Or Search Live arXiv    │      │ Embedding + Recency +   │      │ PyMuPDF4LLM Markdown    │      │ Hybrid RRF Retrieval    │
│ Titles & Abstracts      │      │ Citations + Authors     │      │ Chunk + Chroma Index    │      │ Strict [1], [2] Citings │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```
"""
    )

    with st.expander("🔬 How 2D Novelty & Subspace Geometry Works", expanded=True):
        st.markdown(
            r"""
When a paper is scored relative to its cited references, its 768-dimensional embedding vector is projected onto a 2D orthonormal plane defined by its literature:

- **Horizontal Axis ($e_1$):** Direction of the single closest foundational reference (the **Dominant Parent**, shown as a **Red Dot** at $(1, 0)$).
- **Vertical Axis ($e_2$):** Orthogonal synergy dimension capturing combination across other cited lineages.
- **Scattered Dots ($y_j$):** Every cited paper from the bibliography.
  - **Indigo/Purple Dots:** Active foundational parents that contribute positive weight ($\lambda_j > 0$) to the reconstruction.
  - **Muted Grey Dots:** Background references ($\lambda_j = 0$) cited for context or tools.
- **Blue Cone Projection Dot ($\hat{x} = (m, \sqrt{\Delta})$):** The non-negative weighted center of gravity ($\sum \lambda_j y_j$) inside the cone formed by references.
- **Orange Dashed Radial Line:** The **Novelty Residual Gap** ($\sqrt{1 - R^2_+}$) from $\hat{x}$ out to the unit horizon $\|x\| = 1$. The longer this line, the greater the paper's creative departure from prior art.

| Metric | Name | Meaning |
| :--- | :--- | :--- |
| **$m$** | Direct Similarity | Max cosine similarity to the closest foundational parent paper. |
| **$\Delta$** | Synergy Blend | Multi-parent combination gain ($R^2_+ - m^2$). Measures genuine synthesis of ideas. |
| **$R^2_+$** | Cone Variance | Fraction of paper representation explained by the non-negative reference cone. |
| **$\sqrt{1 - R^2_+}$** | Novelty Residual | Radial distance outside prior art to the unit sphere. |
"""
        )

    with st.expander("📚 Library vs. Full-Text Ingest (Why Two Steps?)", expanded=False):
        st.markdown(
            """
| Feature | **Catalog Library** | **Full-Text Ingestion (RAG)** |
| :--- | :--- | :--- |
| **Data Held** | Titles, abstracts, citations, years, categories | Complete sectioned Markdown parsed from PDFs |
| **Storage** | High-performance SQLite FTS5 (`data/catalog.sqlite3`) | Chunked ChromaDB vector store (`data/chroma_rag`) |
| **Purpose** | Fast browsing, filtering, and initial topic ranking | Deep question answering, evidence extraction, citation grounding |
| **Laptop Scalability** | Millions of papers easily fit | Only ingest papers you actually care to study |
"""
        )

    with st.expander("⚖️ Multi-Signal Ranking Weights Guide", expanded=False):
        st.markdown(
            """
Adjust the sliders under **Recommend** to customize how papers are ranked:

- **Embedding Semantic Weight ($w_{\text{emb}}$):** Relevance of paper abstract to your research interest query.
- **Recency Decay Weight ($w_{\text{rec}}$):** Exponential half-life boost favoring recent publications.
- **Global Citations Weight ($w_{\text{cit}}$):** Normalized impact based on worldwide citation counts.
- **Author Corpus Weight ($w_{\text{auth}}$):** Continuity boost for authors who frequently publish in this specific domain.
"""
        )

    with st.expander("💻 CLI & Autonomous Scripts Reference", expanded=False):
        st.code(
            "\n".join(
                [
                    "# 1. Start Modern Web Studio (Starlette + SSE Streaming UI)",
                    "make serve           # Runs on http://127.0.0.1:8000",
                    "",
                    "# 2. Start Streamlit Discovery Lab",
                    "make app             # Runs on http://localhost:8501",
                    "",
                    "# 3. Run Autonomous Topic-Based Ingestion",
                    "uv run python scripts/bulk_topic_ingest.py --topics 'mechanistic interpretability' 'sparse autoencoders' --max-per-topic 25",
                    "",
                    "# 4. Run Batch Novelty Scoring for Library Papers",
                    "uv run python scripts/batch_novelty_library.py --limit 50 --output results/batch_novelty_report.json",
                    "",
                    "# 5. Run Verification & Health Checks",
                    "make check           # 11/11 repository hygiene checks",
                    "make test            # 60+ core unit tests",
                ]
            ),
            language="bash",
        )

    report_path = ROOT / "results" / "EVALUATION_REPORT.md"
    if report_path.exists():
        with st.expander("📊 Benchmark Evaluation Report", expanded=False):
            try:
                st.markdown(report_path.read_text(encoding="utf-8")[:4000])
            except Exception:
                pass

