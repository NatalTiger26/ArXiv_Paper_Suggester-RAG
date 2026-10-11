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

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 0.5rem;
    }
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
    "Balanced (default)": "Mix of semantic match, recency, citations, and author signal.",
    "Precision (embedding ↑)": "Trust topic similarity more — when the library is already on-topic.",
    "Prefer recent": "Boost newer papers — fast-moving subfields.",
    "Citation heavy": "Prefer highly cited work — can favor older landmarks.",
}


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
    st.write(f"Ingest finished: **{ok_n}/{len(results)}** succeeded.")
    rows = []
    for r in results:
        rows.append(
            {
                "arXiv ID": r.get("arxiv_id", ""),
                "Status": "✓ ok" if r.get("ok") else "✗ fail",
                "Chunks": r.get("chunks", "—") if r.get("ok") else "—",
                "Title / error": (r.get("title") or r.get("error") or "")[:100],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    log(f"Ingest summary: {ok_n}/{len(results)} ok")


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


def render_answer_block(question: str, result: dict) -> None:
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
    st.download_button(
        "Download answer (Markdown)",
        data=md,
        file_name="answer.md",
        mime="text/markdown",
        key=f"dl_{abs(hash(question)) % 10_000_000}",
    )


def run_ingest_with_progress(ids: list[str], force: bool = False) -> list[dict]:
    ids = [str(i).strip() for i in ids if i and str(i).strip()]
    if not ids:
        return []
    log(f"Ingest start: {len(ids)} papers force={force} → {ids}")
    progress = st.progress(0, text="Preparing…")
    status = st.empty()
    results: list[dict] = []
    try:
        try:
            from rag.embed_store import load_embedding_model
            from rag.ingest import ingest_paper
        except ImportError:
            from src.rag.embed_store import load_embedding_model
            from src.rag.ingest import ingest_paper

        status.markdown("**Step 1/2 — Loading embedding model** (first load can take 1–2 min)…")
        progress.progress(3, text="Loading embedder…")
        t0 = time.time()
        model = load_embedding_model()
        log(f"Embedder ready in {time.time() - t0:.1f}s")
        progress.progress(8, text="Embedder ready")

        n = len(ids)
        for i, aid in enumerate(ids):
            step_lo = 8 + int(90 * i / n)
            progress.progress(
                step_lo,
                text=f"Paper {i+1}/{n}: {aid} — download PDF…",
            )
            status.markdown(
                f"**Step 2/2 — Paper `{aid}` ({i+1}/{n})**  \n"
                f"1. Download PDF  →  2. Convert to Markdown  →  3. Chunk  →  4. Embed & store"
            )
            log(f"Ingesting {aid} ({i+1}/{n})…")
            t1 = time.time()
            try:
                row = ingest_paper(aid, model=model, force_download=force)
            except Exception as e:
                row = {"arxiv_id": aid, "ok": False, "error": str(e)}
                log(f"  failed {aid}: {e}")
            else:
                if row.get("ok"):
                    log(
                        f"  ok {aid}: {row.get('chunks')} chunks in {time.time()-t1:.1f}s"
                    )
                else:
                    log(f"  fail {aid}: {row.get('error')}")
            results.append(row)
            progress.progress(
                8 + int(90 * (i + 1) / n),
                text=f"Finished {aid} ({i+1}/{n})",
            )
        progress.progress(100, text="Ingest complete")
        status.success("Ingest finished — see table below.")
        log("Ingest complete")
        return results
    except Exception as e:
        status.empty()
        st.markdown(friendly_error(e))
        log(traceback.format_exc()[-500:])
        return results


def run_rank(interest: str, k: int, weights: dict | None) -> list[dict]:
    log(f"Rank start: {interest!r} k={k}")
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

    if st.button("Refresh status", use_container_width=True):
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

st.title("ArXiv Paper Suggester")
st.caption("Find papers · read full text · ask with citations")

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
    demo_k = st.slider("Top papers to ingest", 1, 6, 3, key="home_demo_k")

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
            render_answer_block(demo_q, result)
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
        q = st.text_input("Keywords", key="lib_q")
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
                log(f"Library search: {q!r} cat={cat!r}")
                with st.status("Searching arXiv…", expanded=True) as st_status:
                    st_status.write("Calling arXiv API…")
                    try:
                        from corpus_builder import search_arxiv

                        papers = search_arxiv(
                            q or "",
                            max_results=int(n),
                            category=(cat or None),
                            sort_by=sort,
                        )
                        st_status.write(f"Found {len(papers)} papers")
                        log(f"Search returned {len(papers)}")
                    except Exception as e:
                        st.markdown(friendly_error(e))
                        papers = None
                        st_status.update(label="Search failed", state="error")
                if papers is not None:
                    if not papers:
                        st.warning("No papers found.")
                    else:
                        for p in papers[:8]:
                            st.markdown(
                                f"- `{p.get('arxiv_id') or p.get('id')}` — "
                                f"{(p.get('title') or '')[:90]}"
                            )
                        try:
                            size = save_library(papers, replace=replace)
                            st.success(f"Library size: **{size}**")
                            log(f"Library saved size={size}")
                        except Exception as e:
                            st.markdown(friendly_error(e))
                        if build_idx:
                            with st.status("Building recommend index…", expanded=True) as s2:
                                s2.write("Embedding titles/abstracts (may take a minute)…")
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
        top_n = st.slider("Top-N", 1, 15, 5, key="lib_rank_n")
        min_score = st.slider(
            "Min personal score", 1.0, 5.0, 4.0, 0.5, key="lib_rank_min"
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
    interest = st.text_input("Research interest", key="rec_interest")
    ex_cols = st.columns(len(EXAMPLE_INTERESTS))
    for i, ex in enumerate(EXAMPLE_INTERESTS):
        label = ex[:28] + ("…" if len(ex) > 28 else "")
        if ex_cols[i].button(label, key=f"ex_int_{i}"):
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
        )
        st.caption(PRESET_HELP.get(preset_name, ""))
    weights = WEIGHT_PRESETS.get(preset_name)

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
                }

    prev = st.session_state.get("last_recommend")
    if prev and prev.get("ranked"):
        st.markdown(f"#### Results for _{prev['interest']}_")
        ranked = prev["ranked"]
        ids_order = []
        for i, r in enumerate(ranked, 1):
            pid = paper_id_from_row(r)
            ids_order.append(pid)
            title = (r.get("title") or "")[:100]
            score = r.get("final_score", r.get("score", ""))
            if pid:
                st.markdown(f"**{i}. [{pid}](https://arxiv.org/abs/{pid})** — {title}")
            else:
                st.markdown(f"**{i}.** {title}")
            st.caption(f"Score: {score}")

        st.markdown("#### Ingest for Q&A")
        st.write("Downloads PDFs, converts to Markdown, chunks, embeds into RAG.")
        ingest_n = st.slider(
            "Ingest top-N",
            1,
            max(1, min(len(ids_order), 10)),
            min(5, max(1, len(ids_order))),
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
            st.info("Open the **Ask** tab when ready.")
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

    with st.expander("Advanced", expanded=False):
        top_k = st.slider("Passages (top-k)", 3, 12, 6, key="ask_k")
        use_cache = st.checkbox(
            "Use semantic answer cache",
            value=True,
            key="ask_cache",
            help="Reuse near-duplicate questions. Off = always call the LLM.",
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
            render_answer_block(question.strip(), result)
        except Exception as e:
            progress.empty()
            status.empty()
            st.markdown(friendly_error(e))
            log(traceback.format_exc()[-800:])

    elif st.session_state.get("last_ask"):
        st.markdown("#### Last answer")
        la = st.session_state["last_ask"]
        st.caption(la.get("q"))
        render_answer_block(la["q"], la["result"])


# ── HELP ──────────────────────────────────────────────────────────────────

with tab_help:
    st.subheader("Help & CLI workflow")
    st.markdown(
        """
**Library vs full-text**

| Term | Meaning |
|------|---------|
| **Library** | Titles/abstracts for *ranking* |
| **Full-text** | Ingested PDFs that **Ask** can search |

**If the UI feels stuck**

1. Open **Activity log** at the top (also prints to the terminal).
2. First embedder load can take 1–2 minutes with no other output — that is normal.
3. Full answers often take **10–20 seconds** (LLM).

**Exact CLI workflow** is in `WORKFLOW.md` at the project root. Short version:

```bash
uv sync
# optional: cp .env.example .env  and set GEMINI_API_KEY

uv run python src/corpus_builder.py search "mechanistic interpretability" -n 12 --build-index
uv run python src/recommend.py "mechanistic interpretability circuits" -k 8
uv run python -m src.rag.cli ingest 1806.07572 2501.16496 2304.14997
uv run python -m src.rag.cli status
uv run python -m src.rag.cli ask "What is the neural tangent kernel?"
uv run python -m streamlit run src/app.py
```
"""
    )
    report_path = ROOT / "results" / "EVALUATION_REPORT.md"
    if report_path.exists():
        with st.expander("Evaluation report preview"):
            st.markdown(report_path.read_text(encoding="utf-8")[:5000])
