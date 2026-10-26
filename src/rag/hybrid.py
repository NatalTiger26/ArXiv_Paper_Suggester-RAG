"""Hybrid retrieval: Dense embeddings + Lexical BM25 / token matching with RRF.

Improves retrieval for exact identifier queries (e.g. arXiv IDs like '1806.07572')
and technical acronyms where dense cosine similarity alone can be ambiguous.
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

    Also applies exact paper_id identifier boosting if an arXiv ID is in the query.
    """
    if not chunks or len(chunks) <= 1:
        return chunks[:top_k]

    explicit_ids = extract_query_identifiers(query)

    # 1. Dense ranking order
    dense_order = {c["chunk_id"]: rank for rank, c in enumerate(chunks, 1)}

    # 2. BM25 scoring over retrieved pool
    bm25 = BM25Scorer(chunks)
    bm25_scores = bm25.score(query)
    bm25_ranked = sorted(
        range(len(chunks)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    bm25_order = {chunks[idx]["chunk_id"]: rank for rank, idx in enumerate(bm25_ranked, 1)}

    # 3. Reciprocal Rank Fusion (RRF) with identifier boost
    combined: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        cid = c["chunk_id"]
        dense_r = dense_order[cid]
        bm25_r = bm25_order.get(cid, len(chunks))

        rrf_score = (1.0 / (rrf_k + dense_r)) + (1.0 / (rrf_k + bm25_r))

        # Direct boost if chunk matches explicit arXiv ID in query
        pid = str(c.get("paper_id") or "")
        if any(e == pid or e in pid for e in explicit_ids):
            rrf_score += 0.05  # Strong boost for exact identifier matches

        # Update chunk metadata with hybrid information
        c_copy = dict(c)
        c_copy["dense_rank"] = dense_r
        c_copy["bm25_rank"] = bm25_r
        c_copy["rrf_score"] = round(rrf_score, 5)
        combined.append((rrf_score, c_copy))

    combined.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in combined[:top_k]]
