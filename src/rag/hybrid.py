"""Hybrid retrieval: Dense embeddings + Lexical BM25 + Section-Aware Intent + Entity Resolution.

Improves retrieval for:
1. Exact identifier queries (arXiv IDs like '1806.07572').
2. Technical acronyms and method names (e.g. 'Grad-CAM', 'NTK', 'SHAP', 'LoRA') via FTS5 entity resolution.
3. Section-aware query intents (e.g. boosting Abstract/Intro for summary questions, Experiments for benchmark questions).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)


def extract_query_identifiers(query: str) -> list[str]:
    """Extract arXiv IDs or explicit paper references from a question."""
    matches = ARXIV_ID_PATTERN.findall(query)
    clean = []
    for m in matches:
        aid = re.sub(r"v\d+$", "", m, flags=re.I).strip()
        if aid not in clean:
            clean.append(aid)
    return clean


def detect_query_intent(query: str) -> str:
    """Classify user query into structural paper section intents."""
    q_low = query.lower()
    if any(k in q_low for k in ["limitation", "weakness", "failure", "bottleneck", "future work", "drawback"]):
        return "LIMITATIONS_FUTURE"
    if any(k in q_low for k in ["dataset", "benchmark", "accuracy", "baseline", "experiment", "metric", "hyperparameter", "results", "empirical"]):
        return "EMPIRICAL_EVAL"
    if any(k in q_low for k in ["proof", "theorem", "lemma", "formula", "derivation", "convergence", "bound"]):
        return "THEORETICAL_TECH"
    if any(k in q_low for k in ["contribution", "overview", "summary", "propose", "introduce", "main idea", "novel", "what is", "explain"]):
        return "CONTRIBUTION_SUMMARY"
    return "GENERAL"


def section_intent_boost(section_name: str, intent: str) -> float:
    """Return priority score boost based on section alignment with user intent."""
    if not section_name:
        return 0.0
    sec = section_name.lower()
    if intent == "CONTRIBUTION_SUMMARY":
        if any(k in sec for k in ["abstract", "introduction", "conclusion", "overview", "contribution"]):
            return 0.015
    elif intent == "LIMITATIONS_FUTURE":
        if any(k in sec for k in ["limitation", "discussion", "future work", "conclusion"]):
            return 0.020
    elif intent == "EMPIRICAL_EVAL":
        if any(k in sec for k in ["experiment", "result", "evaluation", "benchmark", "dataset", "ablation"]):
            return 0.020
    elif intent == "THEORETICAL_TECH":
        if any(k in sec for k in ["method", "architecture", "model", "proof", "theory", "formulation"]):
            return 0.015
    return 0.0


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_.\-]+\b", text) if len(w) > 1]


class BM25Scorer:
    """Lightweight in-memory BM25 scorer over a set of document chunks."""

    def __init__(self, corpus: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_tokens = []

        total_tokens = 0
        for doc in corpus:
            text = f"{doc.get('title', '')} {doc.get('paper_id', '')} {doc.get('section', '')} {doc.get('text', '')}"
            tokens = _tokenize(text)
            self.doc_tokens.append(Counter(tokens))
            self.doc_len.append(len(tokens))
            total_tokens += len(tokens)
            for t in set(tokens):
                self.doc_freq[t] += 1

        self.N = len(corpus)
        self.avg_doc_len = total_tokens / max(self.N, 1)

    def score(self, query: str) -> list[float]:
        q_tokens = _tokenize(query)
        scores = [0.0] * self.N
        for t in q_tokens:
            df = self.doc_freq.get(t, 0)
            if df == 0:
                continue
            # Standard Lucene/BM25 IDF
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for i in range(self.N):
                tf = self.doc_tokens[i].get(t, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[i] / max(self.avg_doc_len, 1e-6)))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)
        return scores


def hybrid_rerank(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 6,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense retrieval order with BM25 lexical ranking via Reciprocal Rank Fusion (RRF).

    Includes:
    1. Exact arXiv ID boosting.
    2. FTS5 method/title entity resolution boosting.
    3. Section-aware intent prior boosting.
    """
    if not chunks or len(chunks) <= 1:
        return chunks[:top_k]

    from src.core.acronyms import expand_query_acronyms

    expanded_query = expand_query_acronyms(query)
    explicit_ids = extract_query_identifiers(query)

    # Resolve entity acronyms / method titles (e.g. Grad-CAM, NTK, SHAP)
    matched_entities: list[str] = []
    try:
        from src.catalog import resolve_entities_in_query
        matched_entities = resolve_entities_in_query(expanded_query, limit=4)
    except Exception:
        pass

    intent = detect_query_intent(query)

    # Inspect hard entity presence to adapt dense vs BM25 weights
    uppercase_acronyms = [
        w for w in re.findall(r"\b[A-Za-z0-9_\-]{2,}\b", query)
        if (sum(1 for c in w if c.isupper()) >= 2 or any(c.isupper() for c in w[1:]))
        and not w.isdigit()
    ]
    has_hard_entity = bool(explicit_ids or uppercase_acronyms)

    if has_hard_entity:
        w_dense = 1.00
        w_bm25 = 1.00
        entity_boost = 0.05
    else:
        w_dense = 1.25
        w_bm25 = 0.75
        entity_boost = 0.025

    # 1. Dense ranking order
    dense_order = {c["chunk_id"]: rank for rank, c in enumerate(chunks, 1)}

    # 2. BM25 scoring over retrieved pool
    bm25 = BM25Scorer(chunks)
    bm25_scores = bm25.score(expanded_query)
    bm25_ranked = sorted(
        range(len(chunks)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    bm25_order = {chunks[idx]["chunk_id"]: rank for rank, idx in enumerate(bm25_ranked, 1)}

    # 3. Reciprocal Rank Fusion (RRF) with identifier and section intent boost
    combined: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        cid = c["chunk_id"]
        dense_r = dense_order[cid]
        bm25_r = bm25_order.get(cid, len(chunks))

        rrf_score = (w_dense / (rrf_k + dense_r)) + (w_bm25 / (rrf_k + bm25_r))

        pid = str(c.get("paper_id") or "")
        # Boost 1: Exact explicit arXiv ID in query
        if any(e == pid or e in pid for e in explicit_ids):
            rrf_score += entity_boost

        # Boost 2: Method / entity title match from catalog
        if any(m == pid or m in pid for m in matched_entities):
            rrf_score += 0.025

        # Boost 3: Section intent alignment
        sec_boost = section_intent_boost(str(c.get("section") or ""), intent)
        rrf_score += sec_boost

        # Update chunk metadata with hybrid information
        c_copy = dict(c)
        c_copy["dense_rank"] = dense_r
        c_copy["bm25_rank"] = bm25_r
        c_copy["rrf_score"] = round(rrf_score, 5)
        c_copy["intent"] = intent
        combined.append((rrf_score, c_copy))

    combined.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in combined[:top_k]]
