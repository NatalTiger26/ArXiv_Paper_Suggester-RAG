#!/usr/bin/env python3
"""
Single end-to-end evaluation for ArXiv Paper Suggester (AIP-aligned).

Replaces the need to juggle eval_suite / eval_advanced / rag.cli eval separately.

  uv run python src/final_eval.py
  uv run python src/final_eval.py --retrieval-only
  uv run python src/final_eval.py --with-generation
  uv run python src/final_eval.py --no-cache          # cold generation (honest refusal)
  uv run python src/final_eval.py --skip-dim-sweep

Writes:
  results/EVALUATION_REPORT.md          # human-readable, with interpretations
  results/rag_eval/final_eval.json      # full machine-readable dump
  results/rag_eval/last_eval.json       # compatibility with older tools
  results/rag_eval/dim_sweep.json       # if dim sweep runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from settings import EMBEDDING_MODEL, EVAL_DIR, PROJECT_ROOT, RESULTS_DIR

GOLD_PATH = EVAL_DIR / "gold_questions.json"
OUT_JSON = EVAL_DIR / "final_eval.json"
OUT_REPORT = RESULTS_DIR / "EVALUATION_REPORT.md"
COMPAT_LAST = EVAL_DIR / "last_eval.json"
DIM_PATH = EVAL_DIR / "dim_sweep.json"

DIMS_DEFAULT = [128, 256, 512, 768]


# ── Stats helpers ─────────────────────────────────────────────────────────


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (round(lo, 4), round(hi, 4))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """
    McNemar on discordant pairs (Lab 2).
    b = A right B wrong, c = B right A wrong.
    Two-sided exact binomial test under p=0.5.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "note": "no discordant pairs"}
    # exact two-sided: sum of binom probs as extreme as observed
    k = min(b, c)
    # P(X <= k) * 2, capped at 1
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2 * tail)
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": round(p, 6),
        "prefer": "A" if b > c else ("B" if c > b else "tie"),
    }


def paired_bootstrap_delta(
    a: list[float], b: list[float], n_boot: int = 2000
) -> dict[str, Any]:
    """Paired bootstrap CI and two-sided p for mean(b-a)."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci95": (0.0, 0.0), "p_value": 1.0}
    deltas = [b[i] - a[i] for i in range(n)]
    mean_d = sum(deltas) / n
    rng = random.Random(42)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
    # two-sided p: fraction of bootstrap means on the other side of 0 from observed
    if mean_d >= 0:
        p = 2 * sum(1 for x in boots if x <= 0) / n_boot
    else:
        p = 2 * sum(1 for x in boots if x >= 0) / n_boot
    p = min(1.0, p)
    return {
        "delta": round(mean_d, 4),
        "ci95": (round(lo, 4), round(hi, 4)),
        "p_value": round(p, 6),
        "n": n,
    }


# ── Retrieval / generation scoring ────────────────────────────────────────


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    return 1.0 if any(e in retrieved or any(e in r for r in retrieved) for e in expected) else 0.0


def mrr_score(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    for i, pid in enumerate(retrieved, 1):
        if any(e == pid or e in pid or pid in e for e in expected):
            return 1.0 / i
    return 0.0


def hit_at_1(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    if not retrieved:
        return 0.0
    pid = retrieved[0]
    return 1.0 if any(e == pid or e in pid or pid in e for e in expected) else 0.0


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    keys = [
        "not covered",
        "do not contain",
        "does not provide",
        "no relevant",
        "not enough information",
        "ingested papers do not",
        "cannot answer",
        "outside the scope",
        "not present in",
        "i don't know",
        "i do not know",
        "unable to answer",
    ]
    return any(k in a for k in keys)


CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_validity(answer: str, n_sources: int) -> dict[str, Any]:
    """Lab 4: every [n] must refer to a supplied source index."""
    nums = [int(x) for x in CITATION_RE.findall(answer or "")]
    if not nums:
        return {
            "has_citation": False,
            "valid": n_sources == 0 or looks_like_refusal(answer),
            "invalid_indices": [],
            "n_citations": 0,
        }
    invalid = sorted({n for n in nums if n < 1 or n > n_sources})
    return {
        "has_citation": True,
        "valid": len(invalid) == 0,
        "invalid_indices": invalid,
        "n_citations": len(nums),
    }


def coverage_from_score(top: float) -> str:
    if top <= 0:
        return "none"
    if top < 0.55:
        return "low"
    if top < 0.70:
        return "partial"
    return "high"


# ── Dim sweep (library abstracts) ─────────────────────────────────────────


def run_dim_sweep(dims: list[int] | None = None) -> dict[str, Any]:
    import numpy as np
    from corpus_index import load_corpus, load_model, paper_id, text_for_embedding

    dims = dims or DIMS_DEFAULT
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    queries = [g for g in gold if g.get("expected_paper_ids")]
    corpus = load_corpus()
    if not corpus:
        return {"error": "empty corpus", "rows": []}

    model = load_model(EMBEDDING_MODEL)
    texts = [text_for_embedding(p) for p in corpus]
    ids = [paper_id(p) for p in corpus]
    full = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    q_full = model.encode(
        [g["question"] for g in queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    def trunc(arr, dim):
        if not dim or dim <= 0 or dim >= arr.shape[-1]:
            return arr
        a = arr[..., :dim].copy()
        norms = np.linalg.norm(a, axis=-1, keepdims=True)
        return a / np.maximum(norms, 1e-12)

    rows = []
    for dim in dims:
        C, Q = trunc(full, dim), trunc(q_full, dim)
        sims = Q @ C.T
        mrrs, recalls, hits = [], [], []
        for i, g in enumerate(queries):
            expected = set(g.get("expected_paper_ids") or [])
            order = np.argsort(-sims[i])
            ranked = [ids[j] for j in order[:10]]
            mrr = 0.0
            for rank, pid in enumerate(ranked, 1):
                if any(e in str(pid) or str(pid) in e or e == pid for e in expected):
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)
            recalls.append(1.0 if mrr > 0 else 0.0)
            hits.append(1.0 if mrr == 1.0 else 0.0)
        rows.append(
            {
                "dim": dim,
                "mean_mrr": round(float(np.mean(mrrs)), 4),
                "mean_recall@10": round(float(np.mean(recalls)), 4),
                "mean_hit@1": round(float(np.mean(hits)), 4),
                "mrr_ci95": bootstrap_ci(mrrs),
                "n_queries": len(queries),
                "n_corpus": len(corpus),
            }
        )
        print(
            f"  dim={dim:4d}  MRR={rows[-1]['mean_mrr']:.3f}  "
            f"CI={rows[-1]['mrr_ci95']}  R@10={rows[-1]['mean_recall@10']:.3f}"
        )

    # Prefer highest MRR; if within 0.02 of full, allow smaller dim
    full_row = max(rows, key=lambda r: r["dim"])
    best = sorted(rows, key=lambda r: (-r["mean_mrr"], r["dim"]))[0]
    if best["dim"] == full_row["dim"]:
        rec = f"Keep full dim={full_row['dim']} (best MRR={best['mean_mrr']})."
    elif full_row["mean_mrr"] - best["mean_mrr"] <= 0.02:
        rec = (
            f"Consider EMBEDDING_DIM={best['dim']} (MRR={best['mean_mrr']} vs "
            f"full {full_row['mean_mrr']}; Δ≤0.02). Rebuild indexes after change."
        )
    else:
        rec = (
            f"Prefer dim={full_row['dim']} (MRR={full_row['mean_mrr']}); "
            f"best lower dim={best['dim']} loses {full_row['mean_mrr'] - best['mean_mrr']:.3f}."
        )
    out = {
        "model": EMBEDDING_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "recommendation": rec,
    }
    DIM_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ── Core eval loop ────────────────────────────────────────────────────────


def run_core(
    top_k: int = 6,
    with_generation: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    from rag.embed_store import index_stats, load_embedding_model, query_chunks

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    stats = index_stats()
    model = load_embedding_model()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    print(f"[final_eval] {len(gold)} gold questions  generation={with_generation}  cache={use_cache}")
    for qi, item in enumerate(gold, 1):
        if qi % 10 == 0 or qi == 1:
            print(f"  … {qi}/{len(gold)}")
        t0 = time.perf_counter()
        chunks = query_chunks(item["question"], top_k=top_k, model=model)
        retrieved: list[str] = []
        for c in chunks:
            pid = c.get("paper_id") or ""
            if pid and pid not in retrieved:
                retrieved.append(pid)
        top_score = float(chunks[0]["score"]) if chunks else 0.0
        expected = item.get("expected_paper_ids") or []
        has_rel = bool(expected)
        rec = recall_at_k(retrieved, expected)
        mrr = mrr_score(retrieved, expected)
        hit1 = hit_at_1(retrieved, expected)
        coverage = coverage_from_score(top_score)

        answer = ""
        refusal = False
        cache_hit = False
        cites: dict[str, Any] = {"has_citation": False, "valid": True, "n_citations": 0}
        n_sources_used = 0
        if with_generation:
            from rag.rag_pipeline import ask

            result = ask(
                item["question"],
                top_k=top_k,
                use_cache=use_cache,
                return_chunks=True,
            )
            answer = result.get("answer") or ""
            coverage = result.get("coverage") or coverage
            refusal = looks_like_refusal(answer)
            cache_hit = bool(result.get("cache_hit"))
            refs = result.get("references") or []
            n_sources_used = len(refs) if refs else len(result.get("chunks") or chunks)
            cites = citation_validity(answer, max(n_sources_used, len(chunks)))
        else:
            refusal = coverage in ("none", "low") and bool(item.get("should_refuse"))

        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        should_refuse = bool(item.get("should_refuse"))
        refusal_ok = (refusal == should_refuse) if with_generation else None

        fail_hint = None
        if has_rel and rec < 1.0:
            fail_hint = "missing_or_ranking"
        elif has_rel and hit1 < 1.0 and rec >= 1.0:
            fail_hint = "ranking"
        elif with_generation and should_refuse and not refusal:
            fail_hint = "false_answer"
        elif with_generation and (not should_refuse) and refusal:
            fail_hint = "over_refuse"

        rows.append(
            {
                "id": item["id"],
                "kind": item.get("kind", ""),
                "difficulty": item.get("difficulty", ""),
                "question": item["question"],
                "expected_paper_ids": expected,
                "retrieved_paper_ids": retrieved[:8],
                "has_relevant": has_rel,
                "top_score": round(top_score, 4),
                "recall@k": rec,
                "mrr": round(mrr, 4),
                "hit_rate@1": hit1,
                "coverage": coverage,
                "should_refuse": should_refuse,
                "did_refuse": refusal,
                "refusal_correct": refusal_ok,
                "cache_hit": cache_hit,
                "citation_valid": cites.get("valid"),
                "citation_has": cites.get("has_citation"),
                "n_citations": cites.get("n_citations"),
                "latency_ms": round(elapsed, 1),
                "fail_hint": fail_hint,
                "answer_preview": (answer[:220] + "…") if len(answer) > 220 else answer,
            }
        )

    ret_rows = [r for r in rows if r["has_relevant"]]
    refuse_rows = [r for r in rows if r["should_refuse"]]
    answerable = [r for r in rows if not r["should_refuse"]]
    n_ret = len(ret_rows) or 1
    n_all = len(rows) or 1

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary: dict[str, Any] = {
        "n_questions": len(rows),
        "n_retrieval_scored": len(ret_rows),
        "n_unanswerable": len(refuse_rows),
        "index_papers": stats.get("paper_count"),
        "index_chunks": stats.get("chunk_count"),
        "embedding_model": stats.get("embedding_model"),
        "embedding_dim": stats.get("embedding_dim"),
        "mean_recall@k": mean([r["recall@k"] for r in ret_rows]),
        "mean_mrr": mean([r["mrr"] for r in ret_rows]),
        "mean_hit_rate@1": mean([r["hit_rate@1"] for r in ret_rows]),
        "mean_top_score": mean([r["top_score"] for r in rows]),
        "mrr_ci95": bootstrap_ci([r["mrr"] for r in ret_rows]),
        "recall_ci95": bootstrap_ci([r["recall@k"] for r in ret_rows]),
        "hit1_ci95": bootstrap_ci([r["hit_rate@1"] for r in ret_rows]),
        "with_generation": with_generation,
        "use_cache": use_cache,
        "cache_hit_rate": mean([1.0 if r["cache_hit"] else 0.0 for r in rows]),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[max(int(0.95 * len(latencies)) - 1, 0)], 1)
        if latencies
        else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Refusal
    if with_generation:
        tp = sum(1 for r in refuse_rows if r["did_refuse"])
        fn = sum(1 for r in refuse_rows if not r["did_refuse"])
        fp = sum(1 for r in answerable if r["did_refuse"])
        tn = sum(1 for r in answerable if not r["did_refuse"])
        summary["refusal_confusion"] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        summary["refusal_recall"] = round(tp / len(refuse_rows), 4) if refuse_rows else None
        summary["refusal_precision"] = round(tp / (tp + fp), 4) if (tp + fp) else None
        summary["refusal_accuracy"] = mean(
            [1.0 if r["refusal_correct"] else 0.0 for r in rows if r["refusal_correct"] is not None]
        )
        if refuse_rows:
            summary["refusal_recall_ci95"] = wilson_ci(tp, len(refuse_rows))
        if (tp + fp) > 0:
            summary["refusal_precision_ci95"] = wilson_ci(tp, tp + fp)

        # Citation validity (Lab 4)
        gen_rows = [r for r in rows if not r["should_refuse"] or r.get("n_citations", 0) > 0]
        # Score all generated answers
        cit_ok = sum(1 for r in rows if r.get("citation_valid"))
        summary["citation_validity_rate"] = round(cit_ok / n_all, 4)
        summary["citation_validity_ci95"] = wilson_ci(cit_ok, n_all)
        summary["answers_with_citation"] = sum(1 for r in rows if r.get("citation_has"))

    # Stratification
    by_kind: dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "unknown" for r in ret_rows}):
        kr = [r for r in ret_rows if (r.get("kind") or "unknown") == kind]
        by_kind[kind] = {
            "n": len(kr),
            "mean_mrr": mean([r["mrr"] for r in kr]),
            "mean_recall@k": mean([r["recall@k"] for r in kr]),
            "mean_hit@1": mean([r["hit_rate@1"] for r in kr]),
            "mrr_ci95": bootstrap_ci([r["mrr"] for r in kr]),
        }
    summary["by_kind"] = by_kind

    by_diff: dict[str, Any] = {}
    for d in sorted({r.get("difficulty") or "unknown" for r in ret_rows}):
        dr = [r for r in ret_rows if (r.get("difficulty") or "unknown") == d]
        by_diff[d] = {
            "n": len(dr),
            "mean_mrr": mean([r["mrr"] for r in dr]),
            "mean_recall@k": mean([r["recall@k"] for r in dr]),
            "mean_hit@1": mean([r["hit_rate@1"] for r in dr]),
        }
    summary["by_difficulty"] = by_diff

    fail_counts: Counter[str] = Counter()
    for r in rows:
        if r.get("fail_hint"):
            fail_counts[r["fail_hint"]] += 1
    summary["fail_hint_counts"] = dict(fail_counts)

    # Missing paper impact
    missing: Counter[str] = Counter()
    for r in ret_rows:
        exp = r.get("expected_paper_ids") or []
        got = r.get("retrieved_paper_ids") or []
        for e in exp:
            if e not in got and not any(e in str(g) for g in got):
                missing[e] += 1
    summary["missing_papers"] = [{"arxiv_id": a, "n": n} for a, n in missing.most_common()]

    # Coverage calibration
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in ret_rows:
        buckets[r.get("coverage") or "n/a"].append(r["recall@k"])
    summary["coverage_calibration"] = {
        k: {"n": len(v), "mean_recall": mean(v)} for k, v in sorted(buckets.items())
    }

    # Score quintiles
    scored = sorted(ret_rows, key=lambda r: r.get("top_score") or 0)
    quintiles = []
    n = len(scored)
    for d in range(5):
        lo, hi = int(d * n / 5), int((d + 1) * n / 5)
        chunk = scored[lo:hi]
        if not chunk:
            continue
        quintiles.append(
            {
                "quintile": d + 1,
                "n": len(chunk),
                "score_lo": min(c["top_score"] for c in chunk),
                "score_hi": max(c["top_score"] for c in chunk),
                "mean_hit@1": mean([c["hit_rate@1"] for c in chunk]),
                "mean_recall": mean([c["recall@k"] for c in chunk]),
            }
        )
    summary["score_quintiles"] = quintiles

    # SLO gate
    gates = {
        "mean_mrr_min": 0.55,
        "mean_recall_min": 0.55,
        "refusal_precision_min": 0.50,
        "citation_validity_min": 0.95,
        "p95_latency_ms_max": 500 if not with_generation else 20000,
    }
    slo = {
        "gates": gates,
        "mrr_pass": (summary["mean_mrr"] or 0) >= gates["mean_mrr_min"],
        "recall_pass": (summary["mean_recall@k"] or 0) >= gates["mean_recall_min"],
        "latency_pass": (summary["latency_p95_ms"] or 0) <= gates["p95_latency_ms_max"],
    }
    if with_generation and summary.get("refusal_precision") is not None:
        slo["refusal_precision_pass"] = summary["refusal_precision"] >= gates["refusal_precision_min"]
        slo["citation_validity_pass"] = (
            summary.get("citation_validity_rate") or 0
        ) >= gates["citation_validity_min"]
    slo["all_pass"] = all(v for k, v in slo.items() if k.endswith("_pass"))
    summary["slo_gate"] = slo

    report = {"summary": summary, "rows": rows}
    return report


def load_baseline_for_paired(current_rows: list[dict]) -> dict[str, Any] | None:
    """
    If results/rag_eval/baseline_v1.json exists (optional snapshot), run paired tests.
    Format: {"rows": [{"id": ..., "recall@k": ..., "mrr": ...}, ...]}
    """
    path = EVAL_DIR / "baseline_v1.json"
    if not path.exists():
        return None
    base = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in base.get("rows") or []}
    a_rec, b_rec, a_mrr, b_mrr = [], [], [], []
    # McNemar on recall success
    b_wrong = c_right = 0  # b: v1 right v2 wrong; c: v1 wrong v2 right
    for r in current_rows:
        if not r.get("has_relevant"):
            continue
        old = by_id.get(r["id"])
        if not old:
            continue
        a_rec.append(float(old.get("recall@k") or 0))
        b_rec.append(float(r.get("recall@k") or 0))
        a_mrr.append(float(old.get("mrr") or 0))
        b_mrr.append(float(r.get("mrr") or 0))
        old_ok = float(old.get("recall@k") or 0) >= 1.0
        new_ok = float(r.get("recall@k") or 0) >= 1.0
        if old_ok and not new_ok:
            b_wrong += 1
        if (not old_ok) and new_ok:
            c_right += 1
    if not a_rec:
        return None
    return {
        "n_paired": len(a_rec),
        "recall_paired": paired_bootstrap_delta(a_rec, b_rec),
        "mrr_paired": paired_bootstrap_delta(a_mrr, b_mrr),
        "mcnemar_recall": mcnemar_exact(b_wrong, c_right),
        "baseline_path": str(path.relative_to(PROJECT_ROOT)),
    }


def maybe_save_baseline(rows: list[dict]) -> None:
    """Save current as baseline_v1 only if file does not exist (first successful run)."""
    path = EVAL_DIR / "baseline_v1.json"
    if path.exists():
        return
    slim = {
        "rows": [
            {
                "id": r["id"],
                "recall@k": r["recall@k"],
                "mrr": r["mrr"],
                "hit_rate@1": r["hit_rate@1"],
            }
            for r in rows
            if r.get("has_relevant")
        ],
        "note": "Auto-saved first final_eval snapshot for future paired tests. Delete to refresh.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"[final_eval] Saved baseline snapshot → {path} (for future paired v1/v2 tests)")


# ── Report writer ─────────────────────────────────────────────────────────


def write_report(
    core: dict[str, Any],
    dim: dict[str, Any] | None,
    paired: dict[str, Any] | None,
) -> Path:
    s = core["summary"]
    rows = core["rows"]
    lines: list[str] = []

    def h(title: str, level: int = 2):
        lines.append("#" * level + " " + title)
        lines.append("")

    def p(*parts: str):
        lines.append(" ".join(parts))
        lines.append("")

    h("Evaluation Report — ArXiv Paper Suggester", 1)
    p(f"_Generated: {s.get('timestamp')}_  ·  Single harness: `src/final_eval.py`")

    h("1. What the system does")
    p(
        "A researcher builds a paper library from arXiv, ranks papers for an interest,",
        "ingests full text, and asks questions answered only from those papers with citations.",
        "Embeddings: local EmbeddingGemma 2. Generation: Gemini (optional NVIDIA failsafe).",
    )

    h("2. How to read the numbers (AIP concepts)")
    lines.extend(
        [
            "| Concept | Meaning here |",
            "|---------|--------------|",
            "| **Recall@k** | Fraction of labelled questions where *any* expected paper appears in the top-k retrieved papers (Lab 3). |",
            "| **MRR** | Mean reciprocal rank of the first expected paper; rewards putting the right paper higher (Lab 3). Prefer over saturated hit@5. |",
            "| **Hit@1** | Whether the top-1 paper is expected. Harder than Recall@k. |",
            "| **95% bootstrap CI** | Uncertainty on the mean over this gold set (Lab 2). Overlapping CIs ⇒ do not claim a difference. |",
            "| **Wilson CI** | Uncertainty on a proportion (refusal, citation validity). |",
            "| **McNemar / paired bootstrap** | Whether v2 beats v1 on the *same* items (Lab 2). Requires `baseline_v1.json`. |",
            "| **Refusal precision / recall** | Of declines, how many should have been declined; of items that should be declined, how many were (Lab 4). **Both** required. |",
            "| **Citation validity** | Every `[n]` in the answer refers to a supplied source — code-checkable (Lab 4). |",
            "| **Failure hints** | Coarse Lab 5 taxonomy: missing_or_ranking, ranking, false_answer, over_refuse. |",
            "| **Cohen’s κ** | Agreement between human and LLM judge on faithfulness/correctness. **Not computed here** unless you add human labels + a judge (Lab 4). Retrieval metrics do not need κ. |",
            "",
        ]
    )

    h("3. Headline results")
    lines.extend(
        [
            "| Metric | Estimate | 95% CI |",
            "|--------|----------|--------|",
            f"| Mean MRR | **{s.get('mean_mrr')}** | {s.get('mrr_ci95')} |",
            f"| Mean Recall@k | **{s.get('mean_recall@k')}** | {s.get('recall_ci95')} |",
            f"| Mean Hit@1 | **{s.get('mean_hit_rate@1')}** | {s.get('hit1_ci95')} |",
            f"| Mean top score | {s.get('mean_top_score')} | — |",
            f"| Index | {s.get('index_papers')} papers / {s.get('index_chunks')} chunks | — |",
            f"| Embedder | {s.get('embedding_model')} @ dim={s.get('embedding_dim')} | — |",
            f"| Latency p50 / p95 | {s.get('latency_p50_ms')} / {s.get('latency_p95_ms')} ms | — |",
            f"| Cache hit rate | {s.get('cache_hit_rate')} | (generation runs) |",
            "",
        ]
    )
    p(
        f"**Interpretation:** On {s.get('n_retrieval_scored')} questions with expected papers,",
        f"MRR={s.get('mean_mrr')} means that on average the first relevant paper sits near rank "
        f"~{round(1 / s['mean_mrr'], 2) if s.get('mean_mrr') else '—'} when found.",
        f"Recall={s.get('mean_recall@k')} means about {round(100 * (s.get('mean_recall@k') or 0))}% of those",
        "questions surface at least one expected paper in the top-k. The CI width reflects n≈80:",
        "small improvements inside the CI should not be treated as proven gains (Lab 2).",
    )

    if s.get("with_generation"):
        h("3.1 Generation, refusal, citations")
        conf = s.get("refusal_confusion") or {}
        lines.extend(
            [
                "| Metric | Value | 95% CI |",
                "|--------|-------|--------|",
                f"| Refusal recall | {s.get('refusal_recall')} | {s.get('refusal_recall_ci95')} |",
                f"| Refusal precision | {s.get('refusal_precision')} | {s.get('refusal_precision_ci95')} |",
                f"| Refusal accuracy | {s.get('refusal_accuracy')} | — |",
                f"| Citation validity rate | {s.get('citation_validity_rate')} | {s.get('citation_validity_ci95')} |",
                f"| Answers with any [n] | {s.get('answers_with_citation')} / {s.get('n_questions')} | — |",
                "",
                "Refusal confusion matrix:",
                "",
                "|  | did refuse | did not |",
                "|--|----------:|--------:|",
                f"| should refuse | TP {conf.get('TP')} | FN {conf.get('FN')} |",
                f"| should answer | FP {conf.get('FP')} | TN {conf.get('TN')} |",
                "",
            ]
        )
        p(
            "**Interpretation:** Refusal recall is “of out-of-scope items, how often we decline.”",
            "Precision is “of our declines, how often we were right to decline.”",
            "A system that refuses everything gets recall 1.0 and is useless (Lab 4).",
            f"Cache hit rate was {s.get('cache_hit_rate')} — if this is near 1.0, refusal numbers may",
            "replay older answers; use `--no-cache` for a cold measurement.",
            "Citation validity near 1.0 is a **code-level** guarantee that cited indices exist;",
            "it does not prove factual correctness (that would need a judge + κ).",
        )
    else:
        p("_Generation metrics skipped (`--retrieval-only`). Re-run with `--with-generation` (and optionally `--no-cache`)._")

    h("4. Stratification (Lab 3)")
    lines.append("### By kind")
    lines.append("")
    lines.append("| Kind | n | MRR | Recall@k | Hit@1 | MRR CI |")
    lines.append("|------|---|-----|----------|-------|--------|")
    for kind, v in (s.get("by_kind") or {}).items():
        lines.append(
            f"| {kind} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} | {v.get('mrr_ci95')} |"
        )
    lines.append("")
    lines.append("### By difficulty")
    lines.append("")
    lines.append("| Difficulty | n | MRR | Recall@k | Hit@1 |")
    lines.append("|------------|---|-----|----------|-------|")
    for d, v in (s.get("by_difficulty") or {}).items():
        lines.append(
            f"| {d} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} |"
        )
    lines.append("")
    p(
        "**Interpretation:** Identifier and comparison kinds are typically harder for dense-only",
        "retrieval (Lab 3 Q44-style exact IDs favor lexical matching). Open-problem and paraphrase",
        "rows usually lead when the supporting papers are ingested. Medium difficulty can look",
        "worse than hard if medium items reference papers still weakly ranked.",
    )

    h("5. Failures and Pareto (Lab 5)")
    lines.append(f"Failure hint counts: `{s.get('fail_hint_counts')}`")
    lines.append("")
    lines.append("| arXiv ID (expected but often not in top-k) | # questions |")
    lines.append("|---------------------------------------------|------------:|")
    for m in (s.get("missing_papers") or [])[:12]:
        lines.append(f"| {m['arxiv_id']} | {m['n']} |")
    lines.append("")
    p(
        "**Interpretation:** `missing_or_ranking` means the expected paper never appeared in top-k —",
        "either it is not ingested, or the embedder ranked other papers higher.",
        "If the PDF is already in the index, the fix is ranking (hybrid BM25, rerank, query rewrite),",
        "not another download. `ranking` means the paper was in top-k but not rank 1.",
        "`false_answer` / `over_refuse` are generation-side (Lab 4 operating point).",
    )

    h("6. Calibration")
    lines.append("### Coverage label vs actual Recall@k")
    lines.append("")
    lines.append("| Coverage | n | Mean Recall@k |")
    lines.append("|----------|---|---------------|")
    for cov, v in (s.get("coverage_calibration") or {}).items():
        lines.append(f"| {cov} | {v['n']} | {v['mean_recall']} |")
    lines.append("")
    lines.append("### Top-score quintiles vs hit rate")
    lines.append("")
    lines.append("| Quintile | Score range | n | Hit@1 | Recall |")
    lines.append("|----------|-------------|---|-------|--------|")
    for q in s.get("score_quintiles") or []:
        lines.append(
            f"| {q['quintile']} | {q['score_lo']}–{q['score_hi']} | {q['n']} | "
            f"{q['mean_hit@1']} | {q['mean_recall']} |"
        )
    lines.append("")
    p(
        "**Interpretation:** If higher score quintiles show higher hit@1, the similarity score is",
        "at least ordinally useful as a confidence signal. If not, do not threshold scores as if",
        "they were calibrated probabilities.",
    )

    h("7. Embedding dimension sweep (Matryoshka)")
    if dim and dim.get("rows"):
        lines.append("| Dim | MRR | MRR CI | Recall@10 | Hit@1 |")
        lines.append("|-----|-----|--------|-----------|-------|")
        for r in dim["rows"]:
            lines.append(
                f"| {r['dim']} | {r['mean_mrr']} | {r.get('mrr_ci95')} | "
                f"{r['mean_recall@10']} | {r['mean_hit@1']} |"
            )
        lines.append("")
        p(f"**Recommendation:** {dim.get('recommendation')}")
        p(
            "This sweep embeds library **titles+abstracts**, not full-text RAG chunks.",
            "Changing `EMBEDDING_DIM` requires resetting the RAG collection and re-ingesting.",
            "If CIs for 512 and 768 overlap heavily, prefer the cheaper dim only for ops cost,",
            "not as a proven quality win (Lab 2 honesty).",
        )
    else:
        p("_Dim sweep not run._")

    h("8. Paired comparison vs baseline (Lab 2)")
    if paired:
        lines.append(f"Baseline: `{paired.get('baseline_path')}` (n={paired.get('n_paired')})")
        lines.append("")
        pr = paired["recall_paired"]
        pm = paired["mrr_paired"]
        mc = paired["mcnemar_recall"]
        lines.extend(
            [
                "| Test | Δ (v2 − v1) | 95% CI | p-value |",
                "|------|-------------|---------|---------|",
                f"| Paired bootstrap Recall@k | {pr['delta']} | {pr['ci95']} | {pr['p_value']} |",
                f"| Paired bootstrap MRR | {pm['delta']} | {pm['ci95']} | {pm['p_value']} |",
                f"| McNemar (recall success) | b={mc['b']} c={mc['c']} | — | {mc['p_value']} |",
                "",
            ]
        )
        p(
            "**Interpretation:** McNemar uses only discordant items (v1 right/v2 wrong vs the reverse).",
            "Small p (e.g. < 0.05) supports a real improvement on this gold set. Large p ⇒ choose on",
            "cost/latency instead of claiming accuracy gains.",
        )
    else:
        p(
            "No `results/rag_eval/baseline_v1.json` with prior per-question scores, so paired p-values",
            "are not computed. The first successful `final_eval` run saves a baseline automatically;",
            "re-run after a change to obtain McNemar / paired bootstrap against that snapshot.",
            "Alternatively, copy an older `last_eval.json` rows into `baseline_v1.json`.",
        )

    h("9. SLO / regression gate (Lab 7)")
    slo = s.get("slo_gate") or {}
    lines.append(f"```json\n{json.dumps(slo, indent=2)}\n```")
    lines.append("")
    p(
        "Gates are project-specific thresholds, not universal truths. Failing refusal precision",
        "while passing retrieval is a typical pattern: retrieval can be demo-ready while",
        "refusal is not deployment-ready.",
    )

    h("10. What is not safe / not claimed")
    lines.extend(
        [
            "- Clinical, legal, or financial decisions without human review.",
            "- Questions about papers outside the RAG index.",
            "- Treating cosine similarity as a calibrated probability.",
            "- Faithfulness/correctness **LLM-judge scores** or **Cohen’s κ** — not run (would need human labels).",
            "- Gold-context decomposition (oracle chunks vs retrieved) — not run; residual error is only coarsely attributed via fail hints.",
            "- Lab 6 attack block rates — out of scope unless tool-use is productized.",
            "",
        ]
    )

    h("11. Executive recommendation")
    fh = s.get("fail_hint_counts") or {}
    top_miss = (s.get("missing_papers") or [{}])[0]
    rec = (
        f"With {s.get('index_papers')} papers indexed, MRR={s.get('mean_mrr')} "
        f"(CI {s.get('mrr_ci95')}) and Recall@k={s.get('mean_recall@k')} on "
        f"{s.get('n_retrieval_scored')} labelled items. "
        f"Residual fail hints: {fh}. "
        f"Most frequent top-k miss: {top_miss.get('arxiv_id')} (n={top_miss.get('n')}) — "
        f"treat as ranking if the PDF is already ingested. "
    )
    if s.get("with_generation"):
        rec += (
            f"Refusal P/R={s.get('refusal_precision')}/{s.get('refusal_recall')}; "
            f"citation validity={s.get('citation_validity_rate')}. "
        )
        if (s.get("cache_hit_rate") or 0) >= 0.9:
            rec += "Generation used a warm cache; cold `--no-cache` run recommended for refusal claims. "
    if dim and dim.get("recommendation"):
        rec += dim["recommendation"] + " "
    rec += (
        "Suitable for a constrained portfolio demo on the ingested MI/NTK library; "
        "not for unsupervised refusal-critical deployment until OOS precision is stronger."
    )
    p(rec)

    h("12. Reproducibility")
    lines.extend(
        [
            "```bash",
            "uv run python src/final_eval.py --retrieval-only",
            "uv run python src/final_eval.py --with-generation --no-cache",
            "uv run python src/final_eval.py                  # retrieval + dim sweep + report",
            "```",
            "",
            "Artifacts: `results/EVALUATION_REPORT.md`, `results/rag_eval/final_eval.json`,",
            "`results/rag_eval/dim_sweep.json`, optional `baseline_v1.json` for paired tests.",
            "",
        ]
    )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_REPORT}")
    return OUT_REPORT


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified final evaluation")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable semantic cache (honest refusal)")
    parser.add_argument("--skip-dim-sweep", action="store_true")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--save-baseline", action="store_true", help="Force write baseline_v1.json")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GOLD_PATH.exists():
        raise SystemExit(f"Missing gold set: {GOLD_PATH}")

    with_gen = args.with_generation and not args.retrieval_only
    # default: retrieval + dim + report; generation only if asked
    if not args.retrieval_only and not args.with_generation:
        with_gen = False  # default path is retrieval-first

    if args.with_generation:
        with_gen = True

    core = run_core(
        top_k=args.k,
        with_generation=with_gen,
        use_cache=not args.no_cache,
    )

    # Persist
    OUT_JSON.write_text(json.dumps(core, indent=2), encoding="utf-8")
    COMPAT_LAST.write_text(json.dumps(core, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    if args.save_baseline or not (EVAL_DIR / "baseline_v1.json").exists():
        if args.save_baseline:
            path = EVAL_DIR / "baseline_v1.json"
            if path.exists():
                path.unlink()
        maybe_save_baseline(core["rows"])

    paired = load_baseline_for_paired(core["rows"])
    # If baseline was just created from this run, paired delta is zero — skip noise
    if paired and paired.get("n_paired") and abs(paired["mrr_paired"]["delta"]) < 1e-9:
        if not (EVAL_DIR / "baseline_v1.json").exists():
            paired = None
        # same snapshot compared to itself
        elif paired["mrr_paired"]["delta"] == 0 and paired["recall_paired"]["delta"] == 0:
            # Check if baseline ids match current exactly as same run
            paired = None
            print("[final_eval] Skipping paired test (baseline is this same run).")

    dim = None
    if not args.skip_dim_sweep:
        print("[final_eval] Dim sweep…")
        try:
            dim = run_dim_sweep()
        except Exception as e:
            print(f"[final_eval] Dim sweep failed: {e}")

    write_report(core, dim, paired)

    s = core["summary"]
    print("\n=== Summary ===")
    print(f"  MRR={s.get('mean_mrr')}  Recall={s.get('mean_recall@k')}  Hit@1={s.get('mean_hit_rate@1')}")
    print(f"  CI MRR={s.get('mrr_ci95')}  papers={s.get('index_papers')}")
    if with_gen:
        print(
            f"  Refusal P/R={s.get('refusal_precision')}/{s.get('refusal_recall')}  "
            f"cite_valid={s.get('citation_validity_rate')}"
        )
    print(f"  Report → {OUT_REPORT}")


if __name__ == "__main__":
    main()
