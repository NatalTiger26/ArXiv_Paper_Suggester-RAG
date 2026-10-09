"""
Streamlit web UI for the RAG extension.

  streamlit run src/rag/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="ArXiv Paper RAG",
    page_icon="📄",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "What is the neural tangent kernel and why does it matter?",
    "What happens to the NTK in the infinite-width limit?",
    "What are open problems in mechanistic interpretability regarding network decomposition?",
    "How can circuits in transformers be discovered automatically?",
]


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embed_model():
    from src.rag.embed_store import load_embedding_model

    return load_embedding_model()


st.title("ArXiv Paper Suggester → RAG")
st.caption(
    "Recommend → ingest full text → ask with citations. "
    "Embeddings: EmbeddingGemma 2 (local). Answers: Gemini."
)

# Warm model once per server process
try:
    get_embed_model()
except Exception as e:
    st.warning(f"Embedding model not loaded yet: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Knowledge base")
    try:
        from src.rag.embed_store import index_stats

        stats = index_stats()
        c1, c2 = st.columns(2)
        c1.metric("Papers", stats["paper_count"])
        c2.metric("Chunks", stats["chunk_count"])
        if stats["papers"]:
            with st.expander("Ingested papers", expanded=False):
                for p in stats["papers"]:
                    st.markdown(f"**`{p['paper_id']}`**  \n{p.get('title', '')[:90]}")
    except Exception as e:
        st.warning(f"Index not ready: {e}")
        stats = {"paper_count": 0, "papers": []}

    st.divider()
    st.subheader("Ingest")
    mode = st.radio(
        "Source",
        ["By arXiv ID", "From recommendation", "From personal ranking"],
        horizontal=False,
    )

    if mode == "By arXiv ID":
        ids_raw = st.text_area(
            "arXiv IDs",
            placeholder="2501.16496\n1806.07572",
            height=90,
        )
        if st.button("Ingest IDs", type="primary", use_container_width=True):
            ids = [x.strip() for x in ids_raw.replace(",", "\n").splitlines() if x.strip()]
            if not ids:
                st.error("Enter at least one ID")
            else:
                with st.spinner("Downloading → Markdown → embedding…"):
                    from src.rag.ingest import ingest_papers

                    results = ingest_papers(ids)
                ok = sum(1 for r in results if r.get("ok"))
                st.success(f"Ingested {ok}/{len(results)}")
                for r in results:
                    if r.get("ok"):
                        st.write(f"✓ `{r['arxiv_id']}` — {r.get('title', '')[:50]} ({r['chunks']} chunks)")
                    else:
                        st.write(f"✗ `{r['arxiv_id']}`: {r.get('error')}")
                st.rerun()

    elif mode == "From recommendation":
        q = st.text_input("Research interest", placeholder="mechanistic interpretability")
        k = st.slider("Top-k", 1, 10, 4)
        if st.button("Recommend + Ingest", type="primary", use_container_width=True):
            if not q.strip():
                st.error("Enter an interest")
            else:
                with st.spinner("Recommending…"):
                    try:
                        from ranking import score_and_rank
                    except ImportError:
                        from src.ranking import score_and_rank
                    recs = score_and_rank(query=q, k=k)
                    ids = [
                        str(r.get("arxiv_id") or r.get("paper_id"))
                        for r in recs
                        if r.get("arxiv_id") or r.get("paper_id")
                    ]
                with st.spinner("Ingesting…"):
                    from src.rag.ingest import ingest_papers

                    results = ingest_papers(ids)
                st.success(f"Ingested {sum(1 for r in results if r.get('ok'))}/{len(results)}")
                st.rerun()

    else:
        top_n = st.slider("Top-N from personal ranking", 1, 20, 6)
        min_score = st.slider("Min score", 1.0, 5.0, 4.0, 0.5)
        if st.button("Ingest from ranking", type="primary", use_container_width=True):
            with st.spinner("Ingesting personal ranking…"):
                from src.rag.ingest import ingest_from_personal_ranking

                results = ingest_from_personal_ranking(top_n=top_n, min_score=min_score)
            st.success(f"Ingested {sum(1 for r in results if r.get('ok'))}/{len(results)}")
            st.rerun()

    st.divider()
    st.caption("CLI: `python -m src.rag.cli --help`")


# ── Main: ask ─────────────────────────────────────────────────────────────

st.header("Ask a question")

ex = st.selectbox("Example questions", ["(custom)"] + EXAMPLE_QUESTIONS)
question = st.text_area(
    "Your question",
    value="" if ex == "(custom)" else ex,
    height=100,
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.number_input("Chunks", 1, 20, 6)
with col2:
    restrict = st.checkbox("Restrict to selected papers")
with col3:
    use_cache = st.checkbox("Semantic cache", value=True)

active_ids = None
if restrict and stats.get("papers"):
    choices = {
        f"{p['paper_id']} — {p.get('title', '')[:50]}": p["paper_id"]
        for p in stats["papers"]
    }
    selected = st.multiselect("Active set", list(choices.keys()))
    active_ids = [choices[s] for s in selected] or None

b1, b2 = st.columns([1, 1])
ask_clicked = b1.button("Ask", type="primary", use_container_width=True)
export_clicked = b2.button("Ask + Export Markdown", use_container_width=True)

if (ask_clicked or export_clicked) and question.strip():
    if stats.get("paper_count", 0) == 0:
        st.warning("Ingest at least one paper first (sidebar).")
    else:
        with st.spinner("Retrieving + generating…"):
            from src.rag.rag_pipeline import ask, export_markdown

            result = ask(
                question.strip(),
                top_k=int(top_k),
                paper_ids=active_ids,
                use_cache=use_cache,
            )
        if result.get("cache_hit"):
            st.info(f"Semantic cache hit (score={result.get('cache_score')})")
        st.caption(f"Coverage: **{result.get('coverage', 'n/a')}**")

        st.subheader("Answer")
        st.markdown(result["answer"])

        if result.get("references"):
            st.subheader("References")
            for r in result["references"]:
                with st.expander(
                    f"[{r['n']}] {r['title'] or r['paper_id']}  (score {r['score']})"
                ):
                    st.markdown(
                        f"**arXiv:** `{r['paper_id']}`  \n**Section:** {r['section']}"
                    )
                    st.caption(r.get("snippet", ""))
                    st.markdown(f"[Open on arXiv](https://arxiv.org/abs/{r['paper_id']})")
                    st.code(
                        f"[{r['n']}] {r.get('title')} (arXiv:{r.get('paper_id')})",
                        language=None,
                    )
        else:
            st.info("No references returned.")

        if export_clicked:
            path = export_markdown(result, question.strip())
            st.success(f"Exported → `{path}`")

st.divider()
with st.expander("Evaluation"):
    st.markdown(
        "Run from CLI for a full report:\n\n"
        "```bash\n"
        "python -m src.rag.cli eval --retrieval-only   # free\n"
        "python -m src.rag.cli eval                    # includes Gemini\n"
        "```\n\n"
        "Reports land in `results/rag_eval/`."
    )
