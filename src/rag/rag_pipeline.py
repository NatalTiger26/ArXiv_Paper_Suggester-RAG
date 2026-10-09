"""
End-to-end RAG: retrieve → (optional semantic cache) → Gemini → answer + references.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    FALLBACK_LLM_API_KEY,
    FALLBACK_LLM_BASE_URL,
    FALLBACK_LLM_MODEL,
    GEMINI_MODEL,
    MIN_RELEVANCE_SCORE,
    PROJECT_ROOT,
    TEMPERATURE,
)
from .embed_store import embed_texts, load_embedding_model, query_chunks

load_dotenv()


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question using ONLY the provided paper excerpts.

Rules:
- Base every claim on the excerpts. If the excerpts do not contain enough information, say so clearly and do not invent a definition.
- When you use information from an excerpt, cite it inline like [1], [2], …
- At the end, list the references you used in the same numbering (title + section).
- Prefer precise, technical language suitable for a researcher.
- Do not invent paper titles, results, equations, or citations that are not in the excerpts.
"""

LOW_COVERAGE_PROMPT = """You are a careful research assistant. The retrieved excerpts may be only weakly related to the question.

Rules:
- If the excerpts do NOT contain a real answer, say clearly that the ingested papers do not cover this question.
- Suggest that the user ingest a more relevant paper (give any arXiv IDs or titles visible in the excerpts only if relevant).
- Do NOT invent definitions or results.
- If there is partial related context, summarise only that and mark it as indirect.
"""


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep highest-scoring chunk per (paper_id, section)."""
    best: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for c in chunks:
        key = (str(c.get("paper_id") or ""), str(c.get("section") or ""))
        prev = best.get(key)
        if prev is None:
            best[key] = c
            order.append(key)
        elif float(c.get("score") or 0) > float(prev.get("score") or 0):
            best[key] = c
    return [best[k] for k in order]


def _build_context(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, list[dict]]:
    chunks = _dedupe_chunks(chunks)
    parts = []
    used = []
    total = 0
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.get('title') or c.get('paper_id')} — § {c.get('section') or 'body'}"
        body = (c.get("text") or "").strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(c)
        total += len(block)
    return "\n---\n".join(parts), used


def _call_openai_compatible(
    prompt: str,
    system: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """OpenAI-compatible chat (NVIDIA Integrate API, local vLLM, etc.)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip/uv install openai  (needed for fallback LLM)")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Prefer Gemini; on failure or missing key, use OpenAI-compatible failsafe."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_err: Exception | None = None

    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            try:
                chat = client.chats.create(
                    model=GEMINI_MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=TEMPERATURE,
                    ),
                )
                response = chat.send_message(prompt)
                return (response.text or "").strip()
            except Exception as chat_err:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=TEMPERATURE,
                    ),
                )
                return (response.text or "").strip()
        except Exception as e:
            gemini_err = e
            print(f"[llm] Gemini unavailable ({e}); trying fallback…")

    if FALLBACK_LLM_API_KEY:
        print(f"[llm] Using fallback model {FALLBACK_LLM_MODEL} @ {FALLBACK_LLM_BASE_URL}")
        return _call_openai_compatible(
            prompt,
            system,
            api_key=FALLBACK_LLM_API_KEY,
            base_url=FALLBACK_LLM_BASE_URL,
            model=FALLBACK_LLM_MODEL,
        )

    msg = (
        "No working generation API.\n"
        "  Set GEMINI_API_KEY, or\n"
        "  Set FALLBACK_LLM_API_KEY (e.g. NVIDIA_API_KEY) for an OpenAI-compatible endpoint.\n"
        f"  Default fallback endpoint: {FALLBACK_LLM_BASE_URL}\n"
        f"  Default fallback model: {FALLBACK_LLM_MODEL}"
    )
    if gemini_err:
        msg += f"\n  Gemini error was: {gemini_err}"
    raise SystemExit(msg)


def _coverage(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "none"
    top = float(chunks[0].get("score") or 0)
    if top < MIN_RELEVANCE_SCORE:
        return "low"
    if top < 0.70:
        return "partial"
    return "high"


def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    paper_ids: list[str] | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    return_chunks: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Full RAG query with optional semantic cache and coverage gating.
    """
    model = load_embedding_model()

    if use_cache:
        from .cache import lookup

        hit = lookup(question, paper_ids, lambda texts: embed_texts(texts, model=model))
        if hit is not None:
            if return_chunks:
                hit.setdefault("chunks", [])
            return hit

    chunks = query_chunks(question, top_k=top_k, paper_ids=paper_ids, model=model)
    cov = _coverage(chunks)

    if not chunks or cov == "none":
        result = {
            "answer": (
                "Not covered by the ingested papers.\n\n"
                "No relevant passages were retrieved. "
                "Ingest a paper that addresses this topic "
                "(e.g. `python -m src.rag.cli ingest <arxiv_id>`) and try again."
            ),
            "references": [],
            "coverage": "none",
            "cache_hit": False,
        }
        if return_chunks:
            result["chunks"] = []
        return result

    context, used = _build_context(chunks, max_context_chars)
    system = LOW_COVERAGE_PROMPT if cov == "low" else SYSTEM_PROMPT
    user_prompt = textwrap.dedent(
        f"""
        Question: {question}

        Paper excerpts (retrieval coverage: {cov}):
        {context}

        Write a clear answer with inline citations [1], [2], … and a short reference list at the end.
        If coverage is low, prefer an explicit "not covered" statement over speculation.
        """
    ).strip()

    answer = _call_gemini(user_prompt, system=system)

    references = []
    for i, c in enumerate(used, 1):
        references.append(
            {
                "n": i,
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", ""),
                "section": c.get("section", ""),
                "score": c.get("score"),
                "snippet": (c.get("text") or "")[:280].replace("\n", " "),
            }
        )

    result: dict[str, Any] = {
        "answer": answer,
        "references": references,
        "coverage": cov,
        "cache_hit": False,
    }
    if return_chunks:
        result["chunks"] = used

    if use_cache and cov in ("high", "partial"):
        from .cache import store

        store(
            question,
            paper_ids,
            result,
            lambda texts: embed_texts(texts, model=model),
        )

    return result


def format_answer(result: dict[str, Any]) -> str:
    """Pretty-print for CLI."""
    lines = []
    if result.get("cache_hit"):
        lines.append(f"(semantic cache hit, score={result.get('cache_score')})")
    if result.get("coverage"):
        lines.append(f"(coverage: {result['coverage']})")
    if lines:
        lines.append("")
    lines.append(result.get("answer") or "")
    lines.append("")
    lines.append("── References ──")
    for r in result.get("references") or []:
        lines.append(
            f"[{r['n']}] {r.get('title') or r.get('paper_id')}  "
            f"(arXiv:{r.get('paper_id')}, § {r.get('section')}, score={r.get('score')})"
        )
        if r.get("snippet"):
            lines.append(f"     {r['snippet'][:160]}…")
    if not result.get("references"):
        lines.append("(none)")
    return "\n".join(lines)


def export_markdown(result: dict[str, Any], question: str, path: str | Path | None = None) -> Path:
    """Write answer + references to a Markdown file for notes / slides."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "rag_answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path else out_dir / f"answer_{ts}.md"

    lines = [
        f"# {question}",
        "",
        f"_coverage: {result.get('coverage')} · cache_hit: {result.get('cache_hit')}_",
        "",
        result.get("answer") or "",
        "",
        "## References",
        "",
    ]
    for r in result.get("references") or []:
        lines.append(
            f"- **[{r['n']}]** {r.get('title') or r.get('paper_id')} "
            f"(arXiv:{r.get('paper_id')}, § {r.get('section')}, score={r.get('score')})"
        )
        if r.get("snippet"):
            lines.append(f"  - {r['snippet'][:200]}")
        lines.append(f"  - https://arxiv.org/abs/{r.get('paper_id')}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
