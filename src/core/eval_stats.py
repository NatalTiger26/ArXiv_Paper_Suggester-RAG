"""Statistical calculation helpers for system evaluation.

Provides bootstrap confidence intervals, paired differences, Wilson score intervals,
Spearman rank correlation, Kendall tau, and NDCG calculations.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate percentile bootstrap confidence interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 1:
        return (round(vals[0], 4), round(vals[0], 4))

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = min(int((1.0 - alpha / 2.0) * n_boot), n_boot - 1)
    return (round(means[lo_idx], 4), round(means[hi_idx], 4))


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate paired bootstrap difference, 95% CI, and two-sided p-value for mean(b - a)."""
    assert len(a) == len(b), "Sequences must have equal length for paired testing"
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci95": (0.0, 0.0), "p_value": 1.0, "n": 0}

    deltas = [float(b[i]) - float(a[i]) for i in range(n)]
    mean_delta = sum(deltas) / n

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()

    lo_idx = int(0.025 * n_boot)
    hi_idx = min(int(0.975 * n_boot), n_boot - 1)

    if mean_delta >= 0.0:
        p_val = 2.0 * sum(1 for x in boot_means if x <= 0.0) / n_boot
    else:
        p_val = 2.0 * sum(1 for x in boot_means if x >= 0.0) / n_boot
    p_val = min(1.0, max(0.0, p_val))

    return {
        "delta": round(mean_delta, 4),
        "ci95": (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4)),
        "p_value": round(p_val, 6),
        "n": n,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / float(n) + (z * z) / (4.0 * float(n) * float(n)))) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def spearman_rank_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float | None:
    """Calculate Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return None
    res = stats.spearmanr(x, y)
    val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    return round(val, 4) if not math.isnan(val) else None


def kendall_tau(
    x: Sequence[float],
    y: Sequence[float],
) -> float | None:
    """Calculate Kendall rank correlation tau coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return None
    res = stats.kendalltau(x, y)
    val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    return round(val, 4) if not math.isnan(val) else None


def ndcg_at_k(
    ranked_ids: Sequence[str],
    ground_truth_scores: dict[str, float],
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank cutoff k."""
    k = max(1, min(k, len(ranked_ids)))
    dcg = 0.0
    for i in range(k):
        pid = ranked_ids[i]
        rel = ground_truth_scores.get(pid, 0.0)
        dcg += (2.0**rel - 1.0) / math.log2(i + 2.0)

    ideal_scores = sorted(ground_truth_scores.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(i + 2.0) for i, rel in enumerate(ideal_scores))
    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def jaccard_token_diversity(titles: Sequence[str]) -> float:
    """Calculate vocabulary token diversity across a set of paper titles."""
    if not titles or len(titles) < 2:
        return 1.0
    word_pattern = re.compile(r"[a-z0-9]+")
    token_sets = [
        set(w for w in word_pattern.findall(t.lower()) if len(w) > 2)
        for t in titles
    ]
    token_sets = [s for s in token_sets if s]
    if len(token_sets) < 2:
        return 1.0

    overlaps = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = len(token_sets[i] | token_sets[j])
            inter = len(token_sets[i] & token_sets[j])
            overlaps.append(inter / union if union > 0 else 0.0)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return round(1.0 - avg_overlap, 4)
