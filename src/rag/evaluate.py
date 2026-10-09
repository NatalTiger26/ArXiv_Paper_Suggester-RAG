"""
RAG evaluation harness (AI-in-Practice inspired).

Metrics:
  Retrieval — Recall@k, MRR, hit_rate@1, mean top score  (Lab 3)
  Refusal   — precision / recall on should_refuse items  (Lab 4)
  Coverage  — high / partial / low / none distribution
  Latency   — p50 / p95                                   (Lab 7)
  Generation (optional) — refusal correctness when API available

Dim sweep is handled by src/eval_suite.py (in-memory Matryoshka on corpus texts).

  uv run python -m src.rag.cli eval --retrieval-only
  uv run python -m src.rag.cli eval
  uv run python src/eval_suite.py --all
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import EVAL_DIR, PROJECT_ROOT
from .embed_store import index_stats, load_embedding_model, query_chunks
from .rag_pipeline import ask

DEFAULT_GOLD = [
    {
        "id": "ntk_def",
        "kind": "definition",
        "question": "What is the neural tangent kernel and why does it matter?",
        "expected_paper_ids": ["1806.07572"],
        "should_refuse": False,
    },
    {
        "id": "ntk_infinite_width",
        "kind": "definition",
        "question": "What happens to the NTK in the infinite-width limit?",
        "expected_paper_ids": ["1806.07572"],
        "should_refuse": False,
    },
    {
        "id": "oos_ising",
        "kind": "unanswerable",
        "question": "What is the exact critical temperature of the 2D Ising model?",
        "expected_paper_ids": [],
        "should_refuse": True,
    },
]


def gold_path() -> Path:
    return EVAL_DIR / "gold_questions.json"


def ensure_gold() -> list[dict[str, Any]]:
    path = gold_path()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_GOLD, indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _recall_at_k(retrieved_ids: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    return 1.0 if any(e in retrieved_ids for e in expected) else 0.0


def _mrr(retrieved_ids: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    for i, pid in enumerate(retrieved_ids, 1):
        if pid in expected:
            return 1.0 / i
    return 0.0


def _hit_at_1(retrieved_ids: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    return 1.0 if retrieved_ids and retrieved_ids[0] in expected else 0.0


def _looks_like_refusal(answer: str) -> bool:
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
    ]
    return any(k in a for k in keys)


def _coverage_from_score(top: float) -> str:
    if top <= 0:
        return "none"
    if top < 0.55:
        return "low"
    if top < 0.70:
        return "partial"
    return "high"



def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-ish percentile bootstrap CI for the mean."""
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def run_eval(
    top_k: int = 6,
    skip_generation: bool = False,
) -> dict[str, Any]:
    gold = ensure_gold()
    stats = index_stats()
    model = load_embedding_model()

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    print(f"[eval] {len(gold)} gold questions…")
    for qi, item in enumerate(gold, 1):
        if qi % 10 == 0 or qi == 1:
            print(f"  … {qi}/{len(gold)}")
        t0 = time.perf_counter()
        chunks = query_chunks(item["question"], top_k=top_k, model=model)
        retrieved_ids: list[str] = []
        for c in chunks:
            pid = c.get("paper_id") or ""
            if pid and pid not in retrieved_ids:
                retrieved_ids.append(pid)
        top_score = float(chunks[0]["score"]) if chunks else 0.0
        expected = item.get("expected_paper_ids") or []
        # Exclude empty-relevant from retrieval averages later
        has_relevant = bool(expected)

        recall = _recall_at_k(retrieved_ids, expected)
        mrr = _mrr(retrieved_ids, expected)
        hit1 = _hit_at_1(retrieved_ids, expected)

        answer = ""
        coverage = _coverage_from_score(top_score)
        refusal = False
        cache_hit = False
        if not skip_generation:
            result = ask(
                item["question"],
                top_k=top_k,
                use_cache=True,
                return_chunks=False,
            )
            answer = result.get("answer") or ""
            coverage = result.get("coverage") or coverage
            refusal = _looks_like_refusal(answer)
            cache_hit = bool(result.get("cache_hit"))
        else:
            refusal = coverage in ("none", "low") and bool(item.get("should_refuse"))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        should_refuse = bool(item.get("should_refuse"))
        refusal_ok = (refusal == should_refuse) if not skip_generation else None

        # Lightweight failure-mode hint (Lab 5-inspired)
        fail_hint = None
        if has_relevant and recall < 1.0:
            fail_hint = "missing_or_ranking"  # expected paper not in top-k
        elif has_relevant and hit1 < 1.0 and recall >= 1.0:
            fail_hint = "ranking"  # in top-k but not rank 1
        elif should_refuse and not refusal and not skip_generation:
            fail_hint = "false_answer"  # should have refused
        elif not should_refuse and refusal and not skip_generation:
            fail_hint = "over_refuse"

        rows.append(
            {
                "id": item["id"],
                "kind": item.get("kind", ""),
                "difficulty": item.get("difficulty", ""),
                "question": item["question"],
                "expected_paper_ids": expected,
                "retrieved_paper_ids": retrieved_ids[:5],
                "has_relevant": has_relevant,
                "top_score": round(top_score, 4),
                "recall@k": recall,
                "mrr": round(mrr, 4),
                "hit_rate@1": hit1,
                "coverage": coverage,
                "should_refuse": should_refuse,
                "did_refuse": refusal,
                "refusal_correct": refusal_ok,
                "cache_hit": cache_hit,
                "latency_ms": round(elapsed_ms, 1),
                "fail_hint": fail_hint,
                "answer_preview": (answer[:200] + "…") if len(answer) > 200 else answer,
            }
        )

    # Retrieval metrics only over questions that have expected papers (Lab 3 n=42 style)
    ret_rows = [r for r in rows if r["has_relevant"]]
    n_ret = len(ret_rows) or 1
    n_all = len(rows) or 1

    refuse_rows = [r for r in rows if r["should_refuse"]]
    answerable = [r for r in rows if not r["should_refuse"]]

    if not skip_generation:
        tp = sum(1 for r in refuse_rows if r["did_refuse"])
        fn = sum(1 for r in refuse_rows if not r["did_refuse"])
        fp = sum(1 for r in answerable if r["did_refuse"])
        refusal_recall = tp / len(refuse_rows) if refuse_rows else None
        refusal_precision = tp / (tp + fp) if (tp + fp) else None
        refusal_accuracy = sum(1 for r in rows if r["refusal_correct"]) / n_all
    else:
        refusal_recall = refusal_precision = refusal_accuracy = None

    def pct(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        idx = max(int(p * len(xs)) - 1, 0)
        return round(xs[idx], 1)


    by_difficulty: dict[str, dict[str, float]] = {}
    for diff in sorted({r.get("difficulty") or "unknown" for r in ret_rows}):
        dr = [r for r in ret_rows if (r.get("difficulty") or "unknown") == diff]
        if not dr:
            continue
        by_difficulty[diff] = {
            "n": len(dr),
            "mean_mrr": round(sum(r["mrr"] for r in dr) / len(dr), 4),
            "mean_recall@k": round(sum(r["recall@k"] for r in dr) / len(dr), 4),
            "mean_hit@1": round(sum(r["hit_rate@1"] for r in dr) / len(dr), 4),
        }

    fail_counts: dict[str, int] = {}
    for r in rows:
        h = r.get("fail_hint")
        if h:
            fail_counts[h] = fail_counts.get(h, 0) + 1

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r.get("kind") or "unknown" for r in ret_rows}):
        kr = [r for r in ret_rows if (r.get("kind") or "unknown") == kind]
        if not kr:
            continue
        by_kind[kind] = {
            "n": len(kr),
            "mean_mrr": round(sum(r["mrr"] for r in kr) / len(kr), 4),
            "mean_recall@k": round(sum(r["recall@k"] for r in kr) / len(kr), 4),
            "mean_hit@1": round(sum(r["hit_rate@1"] for r in kr) / len(kr), 4),
        }

    summary = {
        "n_questions": len(rows),
        "n_retrieval_scored": len(ret_rows),
        "n_unanswerable": len(refuse_rows),
        "index_papers": stats.get("paper_count"),
        "index_chunks": stats.get("chunk_count"),
        "embedding_model": stats.get("embedding_model"),
        "embedding_dim": stats.get("embedding_dim"),
        "mean_recall@k": round(sum(r["recall@k"] for r in ret_rows) / n_ret, 4),
        "mean_mrr": round(sum(r["mrr"] for r in ret_rows) / n_ret, 4),
        "mean_hit_rate@1": round(sum(r["hit_rate@1"] for r in ret_rows) / n_ret, 4),
        "mean_top_score": round(sum(r["top_score"] for r in rows) / n_all, 4),
        "refusal_recall": None if refusal_recall is None else round(refusal_recall, 4),
        "refusal_precision": None
        if refusal_precision is None
        else round(refusal_precision, 4),
        "refusal_accuracy": None
        if refusal_accuracy is None
        else round(refusal_accuracy, 4),
        "cache_hit_rate": round(sum(1 for r in rows if r["cache_hit"]) / n_all, 4),
        "latency_p50_ms": pct(latencies, 0.50) or round(statistics.median(latencies), 1),
        "latency_p95_ms": pct(latencies, 0.95),
        "by_kind": by_kind,
        "by_difficulty": by_difficulty,
        "mrr_ci95": _bootstrap_ci([r["mrr"] for r in ret_rows]),
        "recall_ci95": _bootstrap_ci([r["recall@k"] for r in ret_rows]),
        "hit1_ci95": _bootstrap_ci([r["hit_rate@1"] for r in ret_rows]),
        "fail_hint_counts": fail_counts,
        "skip_generation": skip_generation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report = {"summary": summary, "rows": rows}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "last_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "last_eval.md").write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"_Generated: {s.get('timestamp')}_",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Questions (all / retrieval-scored) | {s.get('n_questions')} / {s.get('n_retrieval_scored')} |",
        f"| Index papers / chunks | {s.get('index_papers')} / {s.get('index_chunks')} |",
        f"| Embedding | {s.get('embedding_model')} @ dim={s.get('embedding_dim')} |",
        f"| Mean Recall@k | {s.get('mean_recall@k')} |",
        f"| Mean MRR | {s.get('mean_mrr')} |",
        f"| Mean hit@1 | {s.get('mean_hit_rate@1')} |",
        f"| Mean top score | {s.get('mean_top_score')} |",
        f"| Refusal recall / precision | {s.get('refusal_recall')} / {s.get('refusal_precision')} |",
        f"| Refusal accuracy | {s.get('refusal_accuracy')} |",
        f"| Cache hit rate | {s.get('cache_hit_rate')} |",
        f"| Latency p50 / p95 (ms) | {s.get('latency_p50_ms')} / {s.get('latency_p95_ms')} |",
        "",
        "## By question kind (retrieval-scored only)",
        "",
    ]
    if s.get("by_kind"):
        lines.append("| Kind | n | MRR | Recall@k | Hit@1 |")
        lines.append("|------|---|-----|----------|-------|")
        for kind, v in s["by_kind"].items():
            lines.append(
                f"| {kind} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} |"
            )
    else:
        lines.append("_No retrieval-scored rows._")
    lines.extend(["", "## By difficulty", ""])
    if s.get("by_difficulty"):
        lines.append("| Difficulty | n | MRR | Recall@k | Hit@1 |")
        lines.append("|------------|---|-----|----------|-------|")
        for diff, v in s["by_difficulty"].items():
            lines.append(
                f"| {diff} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} |"
            )
    lines.append("")
    if s.get("mrr_ci95"):
        lines.append(
            f"**95% bootstrap CI** — MRR {s.get('mrr_ci95')}, "
            f"Recall@k {s.get('recall_ci95')}, Hit@1 {s.get('hit1_ci95')}"
        )
    if s.get("fail_hint_counts"):
        lines.append(f"**Failure hints:** `{s.get('fail_hint_counts')}`")
    lines.extend(
        [
            "",
            "## Per-question",
            "",
            "| ID | Kind | Recall@k | MRR | Hit@1 | Top | Coverage | Refuse OK | ms |",
            "|----|------|----------|-----|-------|-----|----------|-----------|----|",
        ]
    )
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r.get('kind','')} | {r['recall@k']} | {r['mrr']} | "
            f"{r['hit_rate@1']} | {r['top_score']} | {r['coverage']} | "
            f"{r['refusal_correct']} | {r['latency_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation (AIP concepts)",
            "",
            "- **Recall@k / MRR / hit@1** — retrieval quality (Lab 3). Prefer MRR when hit@5 saturates.",
            "- **Refusal recall & precision** — both required (Lab 4); only one hides failure modes.",
            "- **Unanswerable items are noisy** at small n — report raw counts in the full report.",
            "- **Coverage** — score-based gate before generation (grounding).",
            "- **Latency p50/p95** — service SLO style (Lab 7).",
            "",
            "## Not safe for",
            "",
            "- Unsupervised clinical, legal, or financial decisions.",
            "- Questions outside the ingested full-text set.",
            "- Treating cosine similarity as calibrated probability.",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== RAG Eval Summary ===")
    for k, v in s.items():
        if k == "by_kind":
            print(f"  by_kind: {json.dumps(v)}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote {EVAL_DIR / 'last_eval.md'}")
