#!/usr/bin/env python3
"""
Unified web UI for ArXiv Paper Suggester.

  uv sync
  uv run python -m streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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



EXAMPLE_INTERESTS = [
    "mechanistic interpretability circuits",
    "neural tangent kernel generalization",
    "statistical mechanics of deep learning",
    "random matrix theory neural networks",
]

EXAMPLE_QUESTIONS = [
    "What is the neural tangent kernel and why does it matter?",
    "What happens to the NTK in the infinite-width limit?",
    "What are open problems in mechanistic interpretability regarding network decomposition?",
    "How can circuits in transformers be discovered automatically?",
]

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
                if pid:
                    st.markdown(
                        f"**{i}. [{pid}](https://arxiv.org/abs/{pid})** — {title}"
                    )
                else:
                    st.markdown(f"**{i}.** {title}")
                st.caption(f"Score: {score}")
    elif kind == "ingest":
        results = payload.get("results") or []
        ok_n = sum(1 for r in results if r.get("ok"))
        st.caption(f"Ingest finished: **{ok_n}/{len(results)}** succeeded")
        with st.expander(f"Per-paper status ({len(results)})", expanded=True):
            rows = []
            for r in results:
                chunks = r.get("chunks")
                if not r.get("ok"):
                    chunks_disp = None
                else:
                    try:
                        chunks_disp = int(chunks) if chunks is not None else None
                    except (TypeError, ValueError):
                        chunks_disp = None
                rows.append(
                    {
                        "arXiv ID": str(r.get("arxiv_id") or ""),
                        "Status": "ok" if r.get("ok") else "fail",
                        "Chunks": chunks_disp,
                        "Title / error": str(r.get("title") or r.get("error") or "")[:120],
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


@st.cache_data(ttl=15, show_spinner=False)
def corpus_status_cached():
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


@st.cache_data(ttl=15, show_spinner=False)
def rag_stats_cached():
    try:
        from rag.embed_store import index_stats

        return index_stats()
    except Exception:
        try:
            from src.rag.embed_store import index_stats

            return index_stats()
        except Exception as e:
            return {
                "paper_count": 0,
                "chunk_count": 0,
                "papers": [],
                "error": str(e),
                "embedding_model": "—",
            }


def clear_data_caches():
    corpus_status_cached.clear()
    rag_stats_cached.clear()


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


def render_answer_block(question: str, result: dict, *, key_prefix: str = "ask") -> None:
    if result.get("cache_hit"):
        st.info(
            f"Cached answer (similarity {result.get('cache_score')}). "
            "Turn off cache in Advanced for a fresh LLM call."
        )
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
                text=f"Paper {i+1}/{n}: {aid} — download / convert / embed…",
            )
            status.markdown(
                f"**Step 2/2 — Paper `{aid}` ({i+1}/{n})**\n"
                "1. Download PDF → 2. Convert to Markdown → 3. Chunk → 4. Embed & store"
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
        papers = rs.get("papers") or []
        if not papers:
            st.caption("None yet.")
        else:
            for p in papers:
                st.markdown(f"`{p.get('paper_id')}`  \n{p.get('title', '')[:70]}")

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

tab_home, tab_lib, tab_rec, tab_ask, tab_help = st.tabs(
    ["Home", "Library", "Recommend", "Ask", "Help"]
)


# ── HOME ──────────────────────────────────────────────────────────────────

with tab_home:
    st.subheader("What do you want to do?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Add papers")
        st.write("Search arXiv or paste IDs.")
        st.caption("→ Library")
    with c2:
        st.markdown("#### 2. Rank by interest")
        st.write("Multi-signal ranking.")
        st.caption("→ Recommend")
    with c3:
        st.markdown("#### 3. Ask questions")
        st.write("Grounded answers + citations.")
        st.caption("→ Ask (after ingest)")

    st.divider()
    st.markdown("#### Quick demo")
    st.write(
        "Runs: seed library (if empty) → rebuild index → rank → ingest top papers → ask. "
        "Watch the **Activity log** and progress labels — first embedder load is slow."
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
            st.write("Top IDs:", ", ".join(ids) if ids else "(none)")

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
    st.subheader("Paper library")
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
        with st.expander("Library tips", expanded=False):
            st.markdown(
                """
- Each block has its own **max results**; the final library is **unique by arXiv id**.
- After big adds, **rebuild the recommend index** before ranking.
- Ranking uses abstracts only; **Ask** needs a separate **ingest** step.
- Details: **Help** tab.
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
    ex_cols = st.columns(len(EXAMPLE_INTERESTS))
    for i, ex in enumerate(EXAMPLE_INTERESTS):
        label = ex[:28] + ("…" if len(ex) > 28 else "")
        if ex_cols[i].button(label, key=f"ex_int_{i}"):
            st.session_state["rec_interest"] = ex
            st.rerun()

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
    with st.expander("How ranking weights work", expanded=False):
        st.markdown(
            """
**Embedding** — semantic fit of your interest text to title/abstract.  
**Recency** — newer papers score higher.  
**Citation** — higher citation counts score higher.  
**Author** — small prior from author-side signal.

Edit any number to leave the preset; values are **normalized** to sum to 1.00.  
Full tables and when to use each preset: **Help** tab.
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
                            f"**{i}. [{pid}](https://arxiv.org/abs/{pid})** — {title}"
                        )
                    else:
                        st.markdown(f"**{i}.** {title}")
                    st.caption(f"Score: {score}")

        st.markdown("#### Ingest for Q&A")
        st.write(
            "Downloads PDFs, converts to Markdown, chunks, embeds into RAG. "
            "You can ingest any number up to the current ranked list."
        )
        with st.expander("Ingest time, RAM, and batch size", expanded=False):
            st.markdown(
                """
**Time (first time, typical):** ~30–90s per paper (OCR-heavy PDFs can take several minutes).  
**RAM:** 10+ papers with EmbeddingGemma can pressure a 16 GB machine — prefer **5–10** per batch or enable **Low memory mode**.  
**Progress** shows i/N, elapsed, and ETA.  
**Force re-download** only when you need fresh PDF/Markdown/embeddings.  
More detail: **Help** tab.
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
        st.warning("No full-text papers yet. Ingest from **Recommend** first.")
    else:
        st.caption(
            f"Searching **{rag_n}** full-text papers · "
            "Typical answer time **10–20 seconds** (LLM)."
        )

    recent = st.session_state.get("recent_questions") or []
    if recent:
        st.markdown("**Recent questions**")
        rcols = st.columns(min(len(recent), 4))
        for i, rq in enumerate(recent[:4]):
            short = rq[:40] + ("…" if len(rq) > 40 else "")
            if rcols[i].button(short, key=f"recent_{i}"):
                st.session_state["ask_q"] = rq
                st.rerun()

    question = st.text_area("Question", height=80, key="ask_q")
    eq = st.columns(len(EXAMPLE_QUESTIONS))
    for i, exq in enumerate(EXAMPLE_QUESTIONS):
        if eq[i].button(exq[:30] + "…", key=f"ex_q_{i}"):
            st.session_state["ask_q"] = exq
            st.rerun()

    with st.expander("Advanced — passages, cache, paper filter", expanded=False):
        st.markdown(
            """
**Passages (top-k)** = chunks sent to the LLM (not total chunks in the index).  
Default **6**. Use **4–6** for narrow facts, **8–12** for comparisons, **10–15** only for broad multi-paper questions.  
Top-k need **not** scale with library size (2000+ chunks still start at 6–8). Prefer ingesting better papers over k ≫ 15.  
See **Help** for the full table.
"""
        )
        top_k = int_with_input(
            "Passages (top-k)",
            1,
            20,
            6,
            key="ask_k",
            help="Chunks sent to the LLM after retrieval (1–20). Default 6.",
        )
        use_cache = st.checkbox(
            "Use semantic answer cache",
            value=True,
            key="ask_cache",
            help="Reuse near-duplicate questions (faster). Off = always call the LLM.",
        )
        paper_filter = st.multiselect(
            "Restrict to papers",
            options=[
                p.get("paper_id")
                for p in (rs.get("papers") or [])
                if p.get("paper_id")
            ],
            key="ask_filter",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        ask_clicked = st.button(
            "Ask",
            type="primary",
            disabled=rag_n == 0 or not (question or "").strip(),
            key="ask_btn",
        )
    with col_b:
        if st.session_state.get("last_ask"):
            if st.button("Clear last answer", key="ask_clear"):
                st.session_state["last_ask"] = None
                st.rerun()

    if ask_clicked and (question or "").strip():
        log(f"Ask: {question.strip()!r} top_k={top_k} cache={use_cache}")
        progress = st.progress(0, text="Starting…")
        status = st.empty()
        try:
            status.markdown("**1/3 Retrieving** relevant chunks from full-text index…")
            progress.progress(15, text="Retrieving…")
            try:
                from rag.rag_pipeline import ask
            except ImportError:
                from src.rag.rag_pipeline import ask
            status.markdown(
                "**2/3 Generating** with Gemini (often 10–20 s — watch Activity log)…"
            )
            progress.progress(40, text="Calling LLM…")
            t0 = time.time()
            result = ask(
                question.strip(),
                top_k=top_k,
                paper_ids=paper_filter or None,
                use_cache=use_cache,
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
            }
            push_recent_question(question.strip())
            render_answer_block(question.strip(), result, key_prefix="ask_live")
        except Exception as e:
            progress.empty()
            status.empty()
            st.markdown(friendly_error(e))
            log(traceback.format_exc()[-800:])

    elif st.session_state.get("last_ask"):
        st.markdown("#### Last answer")
        la = st.session_state["last_ask"]
        st.caption(la.get("q"))
        render_answer_block(la["q"], la["result"], key_prefix="ask_last")


# ── HELP ──────────────────────────────────────────────────────────────────

with tab_help:
    st.subheader("How to use this app")
    st.markdown(
        """
### The basic loop

1. **Library** — add papers (keywords and/or arXiv IDs). This is *metadata* for ranking.
2. **Recommend** — rank the library for your research interest; optionally **ingest** top papers.
3. **Ask** — question answering over **ingested full text only**, with citations.

| Term | What it is | Used by |
|------|------------|---------|
| **Library** | Titles + abstracts | Rank / recommend |
| **Recommend index** | Embeddings of those abstracts | Rank |
| **Full-text / RAG** | PDF → Markdown → chunks → vectors | **Ask** only |

If Ask says the topic is missing, **ingest more relevant papers** — ranking alone is not enough.

---

### Library

- **Multiple topics:** add several keyword blocks; one run searches all and **merges** unique arXiv IDs.
- **Max results per topic:** how many hits each block contributes (1–50), not a global unique cap.
- **Replace library:** overwrites the saved corpus; confirm when enabled.
- **Rebuild index:** required after large library changes before ranking quality is trustworthy.

---

### Recommend — ranking weights

Four signals (shown to 2 decimals; normalized to sum to **1.00** if you edit them):

| Signal | Meaning |
|--------|---------|
| **Embedding** | Semantic match of your interest text to title/abstract |
| **Recency** | Prefer newer papers |
| **Citation** | Prefer higher citation counts |
| **Author** | Light prior from author-side signal |

**Presets**

| Preset | embedding | recency | citation | author | When to use |
|--------|-----------|---------|----------|--------|-------------|
| Balanced (default) | 0.50 | 0.20 | 0.25 | 0.05 | General literature dig |
| Precision | 0.65 | 0.10 | 0.20 | 0.05 | Library already on-topic; trust similarity |
| Prefer recent | 0.45 | 0.35 | 0.15 | 0.05 | Fast-moving areas |
| Citation heavy | 0.35 | 0.10 | 0.50 | 0.05 | Landmarks / surveys |
| Custom | *yours* | | | | Any mix that is not a preset |

If your numbers match a preset within ~0.01, the UI notes that. Otherwise ranking runs in **Custom**.

**Top-k (rank list)** — how many library papers to *show* after ranking (not the same as Ask passages).

---

### Ingest (full text)

- **Ingest top-N** is limited by the **current ranked list** (if you ranked 25, max is 25).
- **Rough time per paper (first time, EmbeddingGemma):** download ~2–10s · OCR/convert ~10–60s+ · embed ~5–30s → often **~30–90s/paper**; OCR-heavy PDFs can take several minutes.
- **RAM:** large batches (10+) on a 16 GB Mac can stress MPS/unified memory. Prefer batches of **5–10**, use **Low memory mode**, or ingest in waves.
- **Force re-download** only when you need to refresh a PDF/Markdown/embedding.
- Progress shows **i/N**, elapsed time, and **ETA** from this run’s average.

---

### Ask — passages, cache, filters

**Passages (top-k)** = how many *retrieved chunks* are sent to the LLM — **not** how many chunks exist in the index.

Retrieval still searches the whole full-text index (e.g. 2000+ chunks). The model only sees the top-k snippets.

| Situation | Suggested top-k |
|-----------|-----------------|
| Short definition / fact | **4–6** |
| Typical research question | **6–8** (default **6** is solid) |
| Compare methods / multi-paper | **8–12** |
| Broad survey-style question | **10–15** |
| Rarely needed | **>15** — usually noise + cost; improve ingest set first |

**Does top-k need to grow with the library?**  
No. 200 vs 2000 chunks: still start at **6–8**. A bigger index needs *better ranking and more relevant ingest*, not automatically larger k.

**Coverage** on the answer is from retrieval scores, not a calibrated probability:

- **high** — strong top hits; k is probably enough  
- **partial** — answer may be incomplete; try a narrower question or ingest more  
- **low / none** — topic likely missing from full-text  

**Semantic cache** — reuses near-duplicate questions (faster, free). Turn **off** for a fresh LLM call.

**Restrict to papers** — limit retrieval to selected arXiv IDs (useful when one paper dominates the hits).

Typical **LLM** answer time: **~8–20 seconds** (plus retrieval).

---

### Time & performance expectations

| Step | Typical wait |
|------|----------------|
| First embedder load | **1–2 min** (then reused in-session) |
| Rank ~20–50 library papers | ~5–30s after model is warm |
| Ingest 1 paper | ~30–90s (wide range) |
| Ask (Gemini) | ~8–20s |

Open the **Activity log** if something feels stuck; the terminal shows the same events.

---

### Appearance & CLI

- **Dark / Light** buttons top-right of the title.
- Full CLI path: `WORKFLOW.md`. Short version:

```bash
uv sync
# optional: .env with GEMINI_API_KEY and GEMINI_MODEL=gemini-3.8-flash

uv run python src/corpus_builder.py search "your topic" -n 12 --build-index
uv run python src/recommend.py "your interest" -k 8
uv run python -m src.rag.cli ingest <arxiv_id> ...
uv run python -m src.rag.cli ask "Your question"
uv run python -m streamlit run src/app.py
```
"""
    )
    report_path = ROOT / "results" / "EVALUATION_REPORT.md"
    if report_path.exists():
        with st.expander("Evaluation report preview"):
            st.markdown(report_path.read_text(encoding="utf-8")[:5000])
