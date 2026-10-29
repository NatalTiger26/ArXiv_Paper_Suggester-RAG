"""Multi-focal subspace projection tool for discovering conceptual bridge papers.

Identifies bridge literature connecting two distinct research domains by evaluating
dual subspace projection affinity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def find_conceptual_bridges(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_ids: list[str],
    top_k: int = 10,
    min_affinity: float = 0.25,
) -> list[dict[str, Any]]:
    """Discover candidate papers maximizing dual cone projection between two domains.

    Args:
        vector_a: Normalized float32 embedding for domain A (D,)
        vector_b: Normalized float32 embedding for domain B (D,)
        corpus_vectors: Normalized float32 matrix for corpus (N, D)
        corpus_ids: List of canonical/arxiv IDs corresponding to corpus vectors
        top_k: Number of bridge candidates to return
        min_affinity: Minimum individual domain projection threshold

    Returns:
        List of bridge candidate dicts sorted by harmonic bridge score
    """
    if len(corpus_vectors) == 0 or len(corpus_ids) == 0:
        return []

    va = np.asarray(vector_a, dtype=np.float32)
    vb = np.asarray(vector_b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a > 0:
        va = va / norm_a
    if norm_b > 0:
        vb = vb / norm_b

    # Vectorized cosine projection across entire corpus
    sim_a = np.maximum(0.0, np.dot(corpus_vectors, va))
    sim_b = np.maximum(0.0, np.dot(corpus_vectors, vb))

    # Harmonic bridge score requires substantial projection onto both domains
    denom = sim_a + sim_b + 1e-8
    bridge_scores = (2.0 * sim_a * sim_b) / denom

    # Filter out papers that only align with one side
    mask = (sim_a >= min_affinity) & (sim_b >= min_affinity)
    candidate_indices = np.where(mask)[0]

    if len(candidate_indices) == 0:
        # Fallback to top harmonic scores without strict min_affinity threshold
        candidate_indices = np.argsort(bridge_scores)[::-1][:top_k]
    else:
        # Sort filtered candidates by bridge score descending
        sub_scores = bridge_scores[candidate_indices]
        sorted_sub = np.argsort(sub_scores)[::-1][:top_k]
        candidate_indices = candidate_indices[sorted_sub]

    results: list[dict[str, Any]] = []
    for idx in candidate_indices:
        results.append(
            {
                "paper_id": corpus_ids[idx],
                "bridge_score": round(float(bridge_scores[idx]), 4),
                "domain_a_affinity": round(float(sim_a[idx]), 4),
                "domain_b_affinity": round(float(sim_b[idx]), 4),
            }
        )

    return results
