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

try:
    from settings import get_gemini_model
except Exception:
    try:
        from src.settings import get_gemini_model
    except Exception:
        def get_gemini_model() -> str:
            return (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()



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


def _chunk_rank_score(c: dict[str, Any]) -> float:
    """Extract ranking score, strictly preferring hybrid RRF score over raw dense score."""
    if not isinstance(c, dict):
        return 0.0
    if c.get("rrf_score") is not None:
        try:
            return float(c["rrf_score"])
        except (ValueError, TypeError):
            pass
    try:
        return float(c.get("score") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def _dedupe_chunks(
    chunks: list[dict[str, Any]],
    max_per_paper: int = 3,
    allow_adjacent: bool = True,
) -> list[dict[str, Any]]:
    """Deduplicate, expand adjacent windows, and enforce per-paper diversity caps."""
    if not chunks:
        return []

    # Sort chunks by hybrid ranking score descending
    sorted_chunks = sorted(chunks, key=_chunk_rank_score, reverse=True)
    paper_counts: dict[str, int] = {}
    best: dict[tuple[str, str], dict] = {}
    selected: list[dict] = []

    unique_papers = len({c.get("paper_id") for c in sorted_chunks if c.get("paper_id")})
    cap = max_per_paper if unique_papers > 1 else 999

    for c in sorted_chunks:
        pid = str(c.get("paper_id") or "")
        sec = str(c.get("section") or "")
        key = (pid, sec)

        if paper_counts.get(pid, 0) >= cap:
            continue

        if key not in best:
            best[key] = c
            selected.append(c)
            paper_counts[pid] = paper_counts.get(pid, 0) + 1
        elif allow_adjacent:
            # Allow adjacent chunk continuation in same section
            prev = best[key]
            try:
                prev_idx = int(prev.get("chunk_index") or -99)
                curr_idx = int(c.get("chunk_index") or -99)
                if abs(prev_idx - curr_idx) == 1 and paper_counts.get(pid, 0) < cap:
                    selected.append(c)
                    paper_counts[pid] = paper_counts.get(pid, 0) + 1
            except (ValueError, TypeError):
                pass

    return sorted(selected, key=_chunk_rank_score, reverse=True)


def _expand_parent_context(c: dict[str, Any]) -> str:
    """Expand child chunk with surrounding parent markdown section context up to ~300 tokens."""
    body = (c.get("text") or "").strip()
    source_md = c.get("source_md") or ""
    pid = c.get("paper_id") or ""
    start = c.get("char_start")
    end = c.get("char_end")
    if not body:
        return body
    try:
        from .config import MARKDOWN_DIR
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else None
        md_file = None
        if source_md:
            candidate = Path(MARKDOWN_DIR) / source_md
            if candidate.exists():
                md_file = candidate
        if md_file is None and pid:
            for cand in (Path(MARKDOWN_DIR) / f"{pid}.md", Path(MARKDOWN_DIR) / f"{pid}.mmd"):
                if cand.exists():
                    md_file = cand
                    break
        if md_file and md_file.exists():
            full_text = md_file.read_text(encoding="utf-8", errors="ignore")
            raw_slice = body.split(": ", 1)[-1] if " > § " in body else body
            clean_slice = raw_slice[:120].strip()

            c_start = None
            c_end = None

            # Verify whether provided offsets match the slice
            if start_i is not None and end_i is not None and start_i >= 0 and end_i > start_i:
                direct_slice = full_text[start_i:end_i]
                if clean_slice and clean_slice in direct_slice:
                    c_start = start_i
                    c_end = end_i

            # Dynamic locate fallback for legacy chunks with section-relative offsets
            if c_start is None and clean_slice:
                pos = full_text.find(clean_slice)
                if pos != -1:
                    c_start = pos
                    c_end = pos + len(raw_slice)

            if c_start is not None and c_end is not None:
                exp_start = max(0, c_start - 250)
                exp_end = min(len(full_text), c_end + 250)
                expanded = full_text[exp_start:exp_end].strip()
                if len(expanded) > len(body):
                    return expanded
    except Exception:
        pass
    return body


def _build_context(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, list[dict]]:
    deduped = _dedupe_chunks(chunks)
    # Strictly maintain hybrid RRF ranking order!
    deduped.sort(key=_chunk_rank_score, reverse=True)
    parts = []
    used = []
    total = 0
    for i, c in enumerate(deduped, 1):
        header = f"[{i}] {c.get('title') or c.get('paper_id')} : Section {c.get('section') or 'body'}"
        body = _expand_parent_context(c)
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


def _extract_suggested_gemini_model(err: Exception) -> str | None:
    """Parse Google 404 text like: use models/gemini-3.8-flash"""
    import re

    text = str(err)
    m = re.search(r"models/(gemini-[a-z0-9._-]+)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\b(gemini-[a-z0-9._-]+)\b", text, re.I)
    if m and "models/" in text:
        return m.group(1)
    return None


def _gemini_generate(client, model: str, prompt: str, system: str) -> str:
    import time
    from google.genai import types

    t0 = time.perf_counter()
    response = None
    try:
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=TEMPERATURE,
            ),
        )
        response = chat.send_message(prompt)
        text = (response.text or "").strip()
    except Exception:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=TEMPERATURE,
            ),
        )
        text = (response.text or "").strip()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        from src.telemetry import record_call

        usage = getattr(response, "usage_metadata", None)
        p_tok = getattr(usage, "prompt_token_count", 0) or 0
        c_tok = getattr(usage, "candidates_token_count", 0) or 0
        record_call(
            call_type="rag_ask",
            model=model,
            prompt_tokens=p_tok,
            candidate_tokens=c_tok,
            latency_ms=latency_ms,
        )
    except Exception:
        pass

    return text


_gemini_client = None
_gemini_client_key: str | None = None


def _get_gemini_client(api_key: str):
    global _gemini_client, _gemini_client_key
    if _gemini_client is not None and _gemini_client_key == api_key:
        return _gemini_client
    from google import genai

    _gemini_client = genai.Client(api_key=api_key)
    _gemini_client_key = api_key
    return _gemini_client


def _call_gemini(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Prefer Gemini, and on failure or missing key, use OpenAI-compatible failsafe."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_err: Exception | None = None

    if api_key:
        try:
            client = _get_gemini_client(api_key)
            requested = get_gemini_model()
            candidates: list[str] = []
            for m in (
                requested,
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-flash-latest",
            ):
                if m and isinstance(m, str) and m not in candidates:
                    candidates.append(m)

            last_err: Exception | None = None
            for model in candidates:
                try:
                    text = _gemini_generate(client, model, prompt, system)
                    return text
                except Exception as e:
                    last_err = e
                    suggested = _extract_suggested_gemini_model(e)
                    if suggested and suggested not in candidates:
                        candidates.append(suggested)
                    print(f"[llm] Gemini model {model!r} retry with fallback candidate: {e}", flush=True)
                    continue
            gemini_err = last_err
            print(f"[llm] Gemini unavailable after trying {candidates}, trying fallback...", flush=True)
        except Exception as e:
            gemini_err = e
            print(f"[llm] Gemini client error ({e}), trying fallback...", flush=True)

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
    raise RuntimeError(msg)


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
    hybrid: bool = True,
) -> dict[str, Any]:
    """
    Full RAG query with optional semantic cache and coverage gating.
    Computes query embedding ONCE for single-pass retrieval and caching.
    """
    model = load_embedding_model()
    # Embed question once for single-pass retrieval and cache lookup
    qvec = embed_texts([question], model=model)[0]
    gen_model = get_gemini_model()

    if use_cache:
        from .cache import lookup

        hit = lookup(
            question,
            paper_ids,
            query_vector=qvec,
            model_id=gen_model,
        )
        if hit is not None:
            if return_chunks:
                hit.setdefault("chunks", [])
            return hit

    fetch_k = max(top_k * 3, 16) if hybrid else top_k
    effective_paper_ids = paper_ids
    if hybrid and not effective_paper_ids:
        from .hybrid import extract_query_identifiers

        explicit_ids = extract_query_identifiers(question)
        if explicit_ids:
            effective_paper_ids = explicit_ids

    chunks = query_chunks(
        question,
        top_k=fetch_k,
        paper_ids=effective_paper_ids,
        model=model,
        query_vector=qvec,
    )

    if hybrid and len(chunks) > 1:
        from .hybrid import hybrid_rerank

        chunks = hybrid_rerank(chunks, question, top_k=top_k)

    cov = _coverage(chunks)
    top_score = float(chunks[0].get("score") or 0.0) if chunks else 0.0
    sixth_score = float(chunks[min(5, len(chunks) - 1)].get("score") or 0.0) if chunks else 0.0

    # Calibrated 3-feature coverage refusal gate
    from src.rag.hybrid import extract_query_identifiers
    q_ids = extract_query_identifiers(question)
    q_tokens = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_.\-]+\b", question) if len(w) > 3]
    top_chunk_text = (chunks[0].get("text") or "").lower() if chunks else ""
    top_chunk_pid = str(chunks[0].get("paper_id") or "").lower() if chunks else ""

    has_entity_hit = False
    if q_ids and any(qid.lower() in top_chunk_pid or qid.lower() in top_chunk_text for qid in q_ids):
        has_entity_hit = True
    elif any(tok in top_chunk_text for tok in q_tokens if tok in ["lora", "transformer", "diffusion", "residual", "attention"]):
        has_entity_hit = True

    score_margin = min(1.0, max(0.0, top_score - sixth_score) / 0.10)
    s_cov = 0.65 * top_score + 0.20 * float(has_entity_hit) + 0.15 * score_margin

    # Refusal guardrail based on calibrated coverage score
    if not chunks or cov == "none" or s_cov < MIN_RELEVANCE_SCORE:
        result = {
            "answer": (
                "Not covered by the ingested papers.\n\n"
                f"The highest retrieval coverage score was {s_cov:.3f} (top score: {top_score:.3f}), which falls below "
                f"the confidence threshold ({MIN_RELEVANCE_SCORE:.2f}). "
                "No passages in the current index provide verified evidence for this question. "
                "Ingest a paper that addresses this topic and try again."
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
            query_vector=qvec,
            model_id=gen_model,
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
