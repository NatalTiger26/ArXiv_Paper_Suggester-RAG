#!/usr/bin/env python3
"""
Unified web UI for ArXiv Paper Suggester.

  uv run streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
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

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 0.5rem;
    }
    /* Slightly calmer captions in dark/light themes */
    [data-testid="stSidebar"] { min-width: 280px; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

WEIGHT_PRESETS = {
    "Balanced (default)": None,
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
    "Balanced (default)": "Equal-ish mix of semantic match, recency, citations, and author signal.",
    "Precision (embedding ↑)": "Trust topic similarity more — best when your library is already on-topic.",
    "Prefer recent": "Boost newer papers — useful for fast-moving subfields.",
    "Citation heavy": "Prefer highly cited work — can bias toward older landmarks.",
}


# ── Helpers ───────────────────────────────────────────────────────────────


def friendly_error(exc: BaseException | str) -> str:
    """Map known failures to short recovery guidance (heuristic: error recovery)."""
    msg = str(exc)
    low = msg.lower()
    if "dimension" in low and ("768" in msg or "512" in msg or "embedding" in low):
        return (
            "**Embedding dimension mismatch.** The vector store was built with a different "
            "`EMBEDDING_DIM` than the current config.\n\n"
            "Fix:\n```bash\n"
            "uv run python -m src.rag.cli reset-index\n"
            "uv run python -m src.rag.cli ingest --force <arxiv_ids…>\n"
            "```"
        )
    if "api key" in low or "apikey" in low or "authentication" in low or "401" in low:
        return (
            "**API key missing or rejected.** Set `GEMINI_API_KEY` in `.env` "
            "(and optional `FALLBACK_LLM_API_KEY` / NVIDIA key). Restart the app after editing `.env`."
        )
    if "not found" in low and "model" in low:
        return (
            "**Model name not available.** Check `GEMINI_MODEL` in `.env` "
            "(e.g. `gemini-2.0-flash` or the model your key supports)."
        )
    if "connection" in low or "timeout" in low or "network" in low:
        return (
            f"**Network problem while calling a remote service.**\n\nDetails: `{msg[:300]}`"
        )
    if "no such file" in low or "corpus" in low:
        return (
            "**Library data missing.** Add papers on the **Library** tab, then rebuild the recommend index."
        )
    return f"**Something went wrong.**\n\n`{msg[:500]}`"


@st.cache_data(ttl=30, show_spinner=False)
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


@st.cache_data(ttl=30, show_spinner=False)
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



def save_library(papers: list, replace: bool = False) -> int:
    """Merge or replace corpus.json; return new size."""
    from corpus_builder import load_corpus, save_corpus, merge_papers
    if replace:
        save_corpus(papers)
        return len(papers)
    merged = merge_papers(load_corpus(), papers)
    save_corpus(merged)
    return len(merged)


def clear_data_caches():
    corpus_status_cached.clear()
    rag_stats_cached.clear()


def system_health() -> dict:
    """Lightweight health (no secrets displayed)."""
    try:
        from settings import FALLBACK_LLM_API_KEY, EMBEDDING_MODEL
    except Exception:
        FALLBACK_LLM_API_KEY = os.getenv("FALLBACK_LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
    gemini = os.getenv("GEMINI_API_KEY", "")
    return {
        "gemini_key": bool(gemini and str(gemini).strip()),
        "fallback_key": bool(FALLBACK_LLM_API_KEY and str(FALLBACK_LLM_API_KEY).strip()),
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
            "Open **Library** → “Rebuild index only”, or add papers with “Rebuild index” checked.",
        )
    if rag_n == 0:
        return (
            "Recommend papers, then ingest for Q&A",
            "Use **Recommend** to rank papers, then **Ingest for Q&A**. Ask only sees full-text papers.",
        )
    return (
        "Ready to ask",
        "Library and full-text index are set up. Use **Ask** for grounded answers with citations.",
    )


def push_recent_question(q: str) -> None:
    q = (q or "").strip()
    if not q:
        return
    hist = st.session_state.setdefault("recent_questions", [])
    hist = [x for x in hist if x != q]
    hist.insert(0, q)
    st.session_state["recent_questions"] = hist[:8]


def show_ingest_results(results: list[dict]) -> None:
    """Per-paper success/fail table after ingest."""
    if not results:
        st.warning("No ingest results.")
        return
    ok_n = sum(1 for r in results if r.get("ok"))
    st.write(f"Ingest finished: **{ok_n}/{len(results)}** succeeded.")
    rows = []
    for r in results:
        rows.append(
            {
                "arXiv ID": r.get("arxiv_id", ""),
                "Status": "✓ ok" if r.get("ok") else "✗ fail",
                "Chunks": r.get("chunks", "—") if r.get("ok") else "—",
                "Title / error": (r.get("title") or r.get("error") or "")[:80],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


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


def render_answer_block(question: str, result: dict, *, show_export: bool = True) -> None:
    """Always-visible answer + sources + download (not hidden in a collapsed expander)."""
    if result.get("cache_hit"):
        st.info(
            f"Cached answer (similarity {result.get('cache_score')}). "
            "Turn off cache in Advanced if you want a fresh generation."
        )

    cov = result.get("coverage") or "n/a"
    cov_help = (
        "**Coverage** estimates whether retrieved passages look on-topic "
        "(from similarity scores). "
        "`high` ≈ safe to trust more · `partial` ≈ incomplete · "
        "`low`/`none` ≈ library may not contain the answer."
    )
    if cov == "high":
        st.success(f"Coverage: **{cov}**", icon="✅")
    elif cov == "partial":
        st.warning(f"Coverage: **{cov}** — answer may be incomplete.")
    else:
        st.error(f"Coverage: **{cov}** — library may not contain this topic.")
    st.caption(cov_help)

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

    if show_export:
        md = answer_to_markdown(question, result)
        st.download_button(
            "Download answer (Markdown)",
            data=md,
            file_name="answer.md",
            mime="text/markdown",
            key=f"dl_{hash(question) % 10_000_000}",
        )
        try:
            from rag.rag_pipeline import export_markdown
        except ImportError:
            try:
                from src.rag.rag_pipeline import export_markdown
            except ImportError:
                export_markdown = None
        if export_markdown is not None:
            if st.button("Also save under results/", key=f"save_{hash(question) % 99991}"):
                path = export_markdown(result, question)
                st.success(f"Saved `{path}`")


def run_ingest_with_progress(ids: list[str], force: bool = False) -> list[dict]:
    """Staged progress: load model → each paper."""
    ids = [i.strip() for i in ids if i and str(i).strip()]
    if not ids:
        return []
    progress = st.progress(0, text="Loading embedding model…")
    status = st.empty()
    try:
        try:
            from rag.embed_store import load_embedding_model
            from rag.ingest import ingest_paper
        except ImportError:
            from src.rag.embed_store import load_embedding_model
            from src.rag.ingest import ingest_paper

        status.info("Stage 1/2 — loading embedder (first time can take a minute)…")
        model = load_embedding_model()
        progress.progress(5, text="Embedder ready")
        results = []
        n = len(ids)
        for i, aid in enumerate(ids):
            frac = 0.05 + 0.95 * (i / max(n, 1))
            progress.progress(min(int(frac * 100), 99), text=f"Ingesting {aid} ({i+1}/{n})")
            status.info(
                f"Stage 2/2 — paper **{aid}** ({i+1}/{n}): download → Markdown → chunk → embed"
            )
            results.append(ingest_paper(aid, model=model, force_download=force))
        progress.progress(100, text="Done")
        status.empty()
        return results
    except Exception as e:
        status.empty()
        st.error("")
        st.markdown(friendly_error(e))
        return []


# ── Session defaults ──────────────────────────────────────────────────────

if "recent_questions" not in st.session_state:
    st.session_state["recent_questions"] = []
if "last_ask" not in st.session_state:
    st.session_state["last_ask"] = None
if "last_recommend" not in st.session_state:
    st.session_state["last_recommend"] = None
if "confirm_replace" not in st.session_state:
    st.session_state["confirm_replace"] = False


# ── Sidebar ───────────────────────────────────────────────────────────────

cs = corpus_status_cached()
rs = rag_stats_cached()
hint_title, hint_body = next_step_hint(cs, rs)
health = system_health()
lib_n = int(cs.get("n_papers") or 0)
rag_n = int(rs.get("paper_count") or 0)

with st.sidebar:
    st.markdown("### Workspace")
    m1, m2 = st.columns(2)
    m1.metric("Library", lib_n, help="Titles/abstracts used for ranking")
    m2.metric("Full-text", rag_n, help="PDFs ingested for Ask / RAG")
    st.caption(
        f"Recommend index: **{'ready' if cs.get('chroma_exists') else 'not built'}**  ·  "
        f"Chunks: **{rs.get('chunk_count', 0)}**"
    )
    if rs.get("embedding_model"):
        st.caption(f"Embedder: `{rs.get('embedding_model')}`")
        if rs.get("embedding_dim") not in (None, "", "None"):
            st.caption(f"Dim: `{rs.get('embedding_dim')}`")

    st.caption(
        "Library = metadata for ranking.  \nFull-text = papers you can **ask** about."
    )

    if st.button("Refresh status", use_container_width=True):
        clear_data_caches()
        st.rerun()

    with st.expander("Full-text papers in RAG"):
        papers = rs.get("papers") or []
        if not papers:
            st.caption("None yet — ingest from Recommend or Library.")
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
            st.warning("Set `GEMINI_API_KEY` in `.env` to generate answers.")

    st.divider()
    st.caption("CLI equivalents")
    st.code(
        "uv run python src/recommend.py \"…\"\n"
        "uv run python -m src.rag.cli ask \"…\"\n"
        "uv run python src/final_eval.py --retrieval-only",
        language="bash",
    )


# ── Header + banners ──────────────────────────────────────────────────────

st.title("ArXiv Paper Suggester")
st.caption("Find papers · read full text · ask with citations")

# Library vs full-text mismatch banner (P0)
if lib_n > 0 and rag_n < lib_n:
    st.warning(
        f"**Library has {lib_n} papers but only {rag_n} are full-text.**  \n"
        "Ask only searches ingested full text. Use **Recommend → Ingest for Q&A** "
        "or **Library** to ingest more PDFs."
    )

if lib_n == 0 or not cs.get("chroma_exists") or rag_n == 0:
    st.info(f"**{hint_title}**  \n{hint_body}")
else:
    st.success(f"**{hint_title}** — {hint_body}")

tab_home, tab_lib, tab_rec, tab_ask, tab_help = st.tabs(
    ["Home", "Library", "Recommend", "Ask", "Help"]
)


# ══════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════

with tab_home:
    st.subheader("What do you want to do?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Add papers")
        st.write("Search arXiv or paste IDs into your library.")
        st.caption("→ Library tab")
    with c2:
        st.markdown("#### 2. Rank by interest")
        st.write("Multi-signal ranking over your library.")
        st.caption("→ Recommend tab")
    with c3:
        st.markdown("#### 3. Ask questions")
        st.write("Answers grounded in full text, with citations.")
        st.caption("→ Ask tab (after ingest)")

    st.divider()
    st.markdown("#### Quick demo")
    st.write(
        "One-click path: ensure library → rank → ingest top papers → answer a question. "
        "Useful for presentations."
    )
    d1, d2 = st.columns(2)
    with d1:
        demo_interest = st.text_input(
            "Research interest",
            value="mechanistic interpretability circuits",
            key="home_demo_interest",
        )
    with d2:
        demo_q = st.text_input(
            "Question",
            value="What are open problems in mechanistic interpretability?",
            key="home_demo_q",
        )
    demo_k = st.slider("How many top papers to ingest", 1, 6, 3, key="home_demo_k")

    if st.button("Run quick demo", type="primary", key="home_demo_btn"):
        progress = st.progress(0, text="Starting…")
        try:
            if lib_n == 0:
                progress.progress(10, text="Searching arXiv for seed papers…")
                from corpus_builder import search_arxiv

                papers = search_arxiv(demo_interest, max_results=12)
                if not papers:
                    st.error("arXiv search returned no papers.")
                else:
                    save_library(papers, replace=True)
                    st.write(f"Seeded library with {len(papers)} papers.")
                    clear_data_caches()

            progress.progress(30, text="Building recommend index…")
            try:
                from corpus_builder import rebuild_index

                rebuild_index()
            except Exception as e:
                st.markdown(friendly_error(e))

            progress.progress(50, text="Ranking…")
            from ranking import score_and_rank

            ranked = score_and_rank(demo_interest, k=int(demo_k))
            ids = []
            for r in ranked[:demo_k]:
                pid = r.get("arxiv_id") or r.get("id") or r.get("paper_id")
                if pid:
                    ids.append(str(pid).replace("arXiv:", "").split("v")[0])
            st.write("Top IDs:", ", ".join(ids) if ids else "(none)")

            if ids:
                progress.progress(65, text="Ingesting full text…")
                results = run_ingest_with_progress(ids, force=False)
                show_ingest_results(results)
                clear_data_caches()

            progress.progress(90, text="Asking…")
            try:
                from rag.rag_pipeline import ask
            except ImportError:
                from src.rag.rag_pipeline import ask
            result = ask(demo_q, top_k=6, use_cache=False)
            st.session_state["last_ask"] = {"q": demo_q, "result": result}
            push_recent_question(demo_q)
            progress.progress(100, text="Done")
            render_answer_block(demo_q, result)
            clear_data_caches()
        except Exception as e:
            st.markdown(friendly_error(e))


# ══════════════════════════════════════════════════════════════════════════
# LIBRARY
# ══════════════════════════════════════════════════════════════════════════

with tab_lib:
    st.subheader("Paper library")
    st.write(
        "Build the **metadata library** used for ranking. "
        "Full-text ingest for Ask is a separate step (Recommend or ingest below)."
    )

    mode = st.radio(
        "How do you want to add papers?",
        ["Search arXiv", "Paste arXiv IDs", "Ingest from personal ranking", "Rebuild index only"],
        horizontal=True,
        key="lib_mode",
    )

    with st.expander("Options", expanded=False):
        replace = st.checkbox(
            "Replace library instead of merging",
            value=False,
            key="lib_replace",
            help="Deletes the previous library contents when saving new search/ID results.",
        )
        if replace:
            st.session_state["confirm_replace"] = st.checkbox(
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
        q = st.text_input(
            "Keywords",
            placeholder="e.g. mechanistic interpretability circuits",
            key="lib_q",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("Max results", 5, 40, 12, key="lib_n")
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
            elif not (q or "").strip() and not (cat or "").strip():
                st.error("Enter keywords or a category.")
            else:
                with st.spinner("Searching arXiv…"):
                    try:
                        from corpus_builder import (
                            search_arxiv,
                            load_corpus,
                            save_corpus,
                            merge_corpus,
                        )

                        papers = search_arxiv(
                            q or "",
                            max_results=int(n),
                            category=(cat or None),
                            sort_by=sort,
                        )
                    except Exception as e:
                        st.markdown(friendly_error(e))
                        papers = None
                if papers is not None:
                    if not papers:
                        st.warning("No papers found.")
                    else:
                        st.write(f"Found **{len(papers)}** papers:")
                        for p in papers[:8]:
                            st.markdown(
                                f"- `{p.get('id') or p.get('arxiv_id')}` — "
                                f"{(p.get('title') or '')[:90]}"
                            )
                        try:
                            size = save_library(papers, replace=replace)
                            st.success(f"Library size: **{size}**")
                        except Exception as e:
                            st.markdown(friendly_error(e))
                        if build_idx:
                            with st.spinner("Building recommend index…"):
                                try:
                                    from corpus_builder import rebuild_index

                                    rebuild_index()
                                    st.success("Recommend index ready.")
                                except Exception as e:
                                    st.warning(f"Index build failed: {e}")
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
                    with st.spinner("Fetching metadata…"):
                        try:
                            from corpus_builder import fetch_by_ids

                            papers = fetch_by_ids(ids)
                            size = save_library(papers, replace=replace)
                            st.success(f"Fetched **{len(papers)}** papers · library size **{size}**.")
                        except Exception as e:
                            st.markdown(friendly_error(e))
                            papers = []
                    if build_idx and papers:
                        with st.spinner("Building recommend index…"):
                            try:
                                from corpus_builder import rebuild_index

                                rebuild_index()
                                st.success("Recommend index ready.")
                            except Exception as e:
                                st.warning(str(e))
                    clear_data_caches()
                    st.rerun()

    elif mode == "Ingest from personal ranking":
        st.write(
            "Pull arXiv IDs from `rankings/personal_importance.csv` (high personal scores) "
            "and ingest full text for Ask."
        )
        top_n = st.slider("Top-N", 1, 15, 5, key="lib_rank_n")
        min_score = st.slider(
            "Min personal score", 1.0, 5.0, 4.0, 0.5, key="lib_rank_min"
        )
        if st.button("Ingest from personal ranking", type="primary", key="lib_rank_btn"):
            try:
                try:
                    from rag.ingest import ingest_from_personal_ranking
                except ImportError:
                    from src.rag.ingest import ingest_from_personal_ranking
                with st.spinner("Ingesting…"):
                    results = ingest_from_personal_ranking(
                        top_n=int(top_n), min_score=float(min_score)
                    )
                show_ingest_results(results)
                clear_data_caches()
            except Exception as e:
                st.markdown(friendly_error(e))

    else:  # Rebuild index only
        st.write("Rebuild the recommend (title/abstract) vector index from the current library.")
        if st.button("Rebuild recommend index", type="primary", key="lib_rebuild_btn"):
            with st.spinner("Building…"):
                try:
                    from corpus_builder import rebuild_index

                    rebuild_index()
                    st.success("Recommend index ready.")
                    clear_data_caches()
                except Exception as e:
                    st.markdown(friendly_error(e))


# ══════════════════════════════════════════════════════════════════════════
# RECOMMEND
# ══════════════════════════════════════════════════════════════════════════

with tab_rec:
    st.subheader("Recommend papers")
    st.write("Rank library papers for a research interest (embedding + recency + citation + author).")

    interest = st.text_input("Research interest", key="rec_interest")
    st.caption("Examples: " + " · ".join(f"`{e}`" for e in EXAMPLE_INTERESTS[:3]))
    ex_cols = st.columns(len(EXAMPLE_INTERESTS))
    for i, ex in enumerate(EXAMPLE_INTERESTS):
        if ex_cols[i].button(ex[:28] + ("…" if len(ex) > 28 else ""), key=f"ex_int_{i}"):
            st.session_state["rec_interest"] = ex
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Top-k", 3, 20, 8, key="rec_k")
    with c2:
        preset_name = st.selectbox(
            "Weight preset",
            list(WEIGHT_PRESETS.keys()),
            key="rec_preset",
            help=PRESET_HELP.get("Balanced (default)"),
        )
        st.caption(PRESET_HELP.get(preset_name, ""))

    weights = WEIGHT_PRESETS.get(preset_name)

    if st.button("Rank papers", type="primary", key="rec_btn"):
        if not (interest or "").strip():
            st.error("Enter a research interest.")
        elif lib_n == 0:
            st.error("Library is empty — add papers on the Library tab first.")
        else:
            with st.spinner("Scoring library…"):
                try:
                    from ranking import score_and_rank

                    ranked = score_and_rank(
                        interest.strip(),
                        k=int(k),
                        weights=weights,
                    )
                    st.session_state["last_recommend"] = {
                        "interest": interest.strip(),
                        "ranked": ranked,
                    }
                except Exception as e:
                    st.markdown(friendly_error(e))

    prev = st.session_state.get("last_recommend")
    if prev and prev.get("ranked"):
        st.markdown(f"#### Results for _{prev['interest']}_")
        ranked = prev["ranked"]
        ids_order = []
        for i, r in enumerate(ranked, 1):
            pid = (
                r.get("arxiv_id")
                or r.get("id")
                or r.get("paper_id")
                or ""
            )
            pid = str(pid).replace("arXiv:", "").split("v")[0]
            ids_order.append(pid)
            title = (r.get("title") or "")[:100]
            score = r.get("final_score", r.get("score", ""))
            st.markdown(f"**{i}. [{pid}](https://arxiv.org/abs/{pid})** — {title}")
            st.caption(f"Score: {score}")

        st.markdown("#### Ingest for Q&A")
        st.write(
            "Full-text ingest downloads PDFs and builds the RAG index so **Ask** can cite them."
        )
        ingest_n = st.slider(
            "Ingest top-N from this ranking",
            1,
            min(len(ids_order), 10),
            min(5, len(ids_order)),
            key="rec_ingest_n",
        )
        force = st.checkbox("Force re-download / re-embed", value=False, key="rec_force")
        if st.button(
            f"Ingest top {ingest_n} for Q&A",
            type="primary",
            key="rec_ingest_btn",
        ):
            to_ingest = [x for x in ids_order[:ingest_n] if x]
            results = run_ingest_with_progress(to_ingest, force=force)
            show_ingest_results(results)
            clear_data_caches()
            st.info("Status refreshed after rerun — open the **Ask** tab when ready.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ASK
# ══════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.subheader("Ask the papers")
    if rag_n == 0:
        st.warning(
            "No full-text papers yet. Ingest from **Recommend** or **Library** before asking."
        )
    else:
        st.caption(
            f"Searching **{rag_n}** full-text papers · "
            "Typical answer time **10–20 seconds** (LLM generation)."
        )

    # Recent questions
    recent = st.session_state.get("recent_questions") or []
    if recent:
        st.markdown("**Recent questions**")
        rcols = st.columns(min(len(recent), 4))
        for i, rq in enumerate(recent[:4]):
            if rcols[i].button(rq[:40] + ("…" if len(rq) > 40 else ""), key=f"recent_{i}"):
                st.session_state["ask_q"] = rq
                st.rerun()

    question = st.text_area(
        "Question",
        height=80,
        key="ask_q",
        placeholder="e.g. What is the neural tangent kernel and why does it matter?",
    )
    st.caption("Try: " + " · ".join(f"`{q[:42]}…`" if len(q) > 42 else f"`{q}`" for q in EXAMPLE_QUESTIONS[:2]))
    eq = st.columns(len(EXAMPLE_QUESTIONS))
    for i, exq in enumerate(EXAMPLE_QUESTIONS):
        if eq[i].button(exq[:30] + "…", key=f"ex_q_{i}"):
            st.session_state["ask_q"] = exq
            st.rerun()

    with st.expander("Advanced", expanded=False):
        top_k = st.slider("Passages to retrieve (top-k)", 3, 12, 6, key="ask_k")
        use_cache = st.checkbox(
            "Use semantic answer cache",
            value=True,
            key="ask_cache",
            help="Reuse answers to very similar questions. Turn off for a guaranteed fresh LLM call.",
        )
        paper_filter = st.multiselect(
            "Restrict to papers (optional)",
            options=[p.get("paper_id") for p in (rs.get("papers") or []) if p.get("paper_id")],
            key="ask_filter",
        )

    col_a, col_b = st.columns([1, 1])
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

    active_ids = paper_filter or None

    if ask_clicked and (question or "").strip():
        with st.spinner("Retrieving passages and generating answer (often 10–20 s)…"):
            try:
                try:
                    from rag.rag_pipeline import ask
                except ImportError:
                    from src.rag.rag_pipeline import ask
                result = ask(
                    question.strip(),
                    top_k=top_k,
                    paper_ids=active_ids,
                    use_cache=use_cache,
                )
                st.session_state["last_ask"] = {
                    "q": question.strip(),
                    "result": result,
                }
                push_recent_question(question.strip())
            except Exception as e:
                st.markdown(friendly_error(e))
                result = None
        if st.session_state.get("last_ask"):
            la = st.session_state["last_ask"]
            render_answer_block(la["q"], la["result"])

    elif st.session_state.get("last_ask"):
        st.markdown("#### Last answer")
        la = st.session_state["last_ask"]
        st.caption(la.get("q"))
        render_answer_block(la["q"], la["result"])


# ══════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════

with tab_help:
    st.subheader("Help & concepts")
    st.markdown(
        """
**Library vs full-text**

| Term | Meaning |
|------|---------|
| **Library** | Titles/abstracts used to *rank* papers for an interest |
| **Full-text (RAG)** | PDFs that were downloaded, chunked, and embedded — what **Ask** searches |

**Coverage** on an answer is a rough signal from retrieval scores, not a calibrated probability.
Prefer `high`; treat `low` / `none` as “maybe not in the library.”

**Typical latency**

- Retrieve only: tens of milliseconds  
- Full answer: often **10–20 seconds** (LLM)

**CLI** (debugging / batch)

```bash
uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python -m src.rag.cli ingest 1806.07572 2501.16496
uv run python -m src.rag.cli ask "What is the NTK?"
uv run python src/final_eval.py --retrieval-only
```

**Evaluation report** (metrics, CIs, refusal, latency):  
`results/EVALUATION_REPORT.md`

**Common fixes**

- Dimension mismatch → `uv run python -m src.rag.cli reset-index` then re-ingest  
- No answers → ingest full-text papers first; check Gemini API key under **System health**
"""
    )

    report_path = ROOT / "results" / "EVALUATION_REPORT.md"
    if report_path.exists():
        with st.expander("Preview evaluation report"):
            st.markdown(report_path.read_text(encoding="utf-8")[:6000])
            if report_path.stat().st_size > 6000:
                st.caption("Truncated preview — open the file for the full report.")
