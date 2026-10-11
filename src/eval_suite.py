#!/usr/bin/env python3
"""
Full evaluation suite for ArXiv Paper Suggester.

  uv run python src/eval_suite.py --all
  uv run python src/eval_suite.py --retrieval-only
  uv run python src/eval_suite.py --dim-sweep
  uv run python src/eval_suite.py --report-only

Writes:
  results/rag_eval/last_eval.json|md
  results/rag_eval/dim_sweep.json
  results/EVALUATION_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from settings import EMBEDDING_MODEL, EVAL_DIR, PROJECT_ROOT, RESULTS_DIR

DIMS_DEFAULT = [128, 256, 512, 768]


def _matryoshka(vecs, dim: int):
    import numpy as np

    arr = np.asarray(vecs, dtype="float32")
    if not dim or dim <= 0 or dim >= arr.shape[-1]:
        return arr
    arr = arr[..., :dim]
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def run_dim_sweep(dims: list[int] | None = None) -> dict[str, Any]:
    """
    In-memory Matryoshka sweep on the *recommend* corpus (titles+abstracts).

    Does not require re-ingesting RAG Chroma. For each dim, embed corpus + gold
    queries and score MRR / Recall@5 by nearest-neighbour over paper vectors.
    """
    import numpy as np

    dims = dims or DIMS_DEFAULT
    from corpus_index import load_corpus, load_model, paper_id, text_for_embedding
    from rag.evaluate import ensure_gold

    corpus = load_corpus()
    if not corpus:
        return {"error": "empty corpus — run corpus_builder first", "rows": []}

    gold = ensure_gold()
    queries = [g for g in gold if g.get("expected_paper_ids")]

    print(f"[dim-sweep] Loading {EMBEDDING_MODEL}…")
    model = load_model(EMBEDDING_MODEL)
    texts = [text_for_embedding(p) for p in corpus]
    ids = [paper_id(p) for p in corpus]
    print(f"[dim-sweep] Embedding {len(texts)} corpus texts (full dim)…")
    full = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    q_texts = [g["question"] for g in queries]
    print(f"[dim-sweep] Embedding {len(q_texts)} queries…")
    q_full = model.encode(
        q_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    rows = []
    for dim in dims:
        t0 = time.perf_counter()
        C = _matryoshka(full, dim)
        Q = _matryoshka(q_full, dim)
        # cosine sim = dot product (L2-normalised)
        sims = Q @ C.T  # (nq, nc)
        mrrs, recalls, hits = [], [], []
        for i, g in enumerate(queries):
            expected = set(g.get("expected_paper_ids") or [])
            order = np.argsort(-sims[i])
            ranked_ids = [ids[j] for j in order[:10]]
            # MRR
            mrr = 0.0
            for rank, pid in enumerate(ranked_ids, 1):
                # paper_id may be bare arxiv id
                if pid in expected or any(pid.endswith(e) or e in pid for e in expected):
                    mrr = 1.0 / rank
                    break
            # also match if expected id is substring of paper_id
            if mrr == 0.0:
                for rank, pid in enumerate(ranked_ids, 1):
                    if any(e in str(pid) for e in expected):
                        mrr = 1.0 / rank
                        break
            hit = 1.0 if mrr == 1.0 else 0.0
            rec = 1.0 if mrr > 0 else 0.0
            mrrs.append(mrr)
            recalls.append(rec)
            hits.append(hit)
        elapsed = (time.perf_counter() - t0) * 1000
        row = {
            "dim": dim,
            "n_queries": len(queries),
            "n_corpus": len(corpus),
            "mean_mrr": round(float(np.mean(mrrs)), 4),
            "mean_recall@10": round(float(np.mean(recalls)), 4),
            "mean_hit@1": round(float(np.mean(hits)), 4),
            "score_ms": round(elapsed, 1),
            "bytes_per_vector_approx": dim * 4,
        }
        rows.append(row)
        print(
            f"  dim={dim:4d}  MRR={row['mean_mrr']:.3f}  "
            f"R@10={row['mean_recall@10']:.3f}  hit@1={row['mean_hit@1']:.3f}  "
            f"({elapsed:.0f} ms score)"
        )

    out = {
        "model": EMBEDDING_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "recommendation": _recommend_dim(rows),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / "dim_sweep.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return out


def _recommend_dim(rows: list[dict]) -> str:
    if not rows:
        return "No data"
    # Prefer highest MRR; break ties with smaller dim
    best = sorted(rows, key=lambda r: (-r["mean_mrr"], r["dim"]))[0]
    full = max(rows, key=lambda r: r["dim"])
    if best["dim"] == full["dim"]:
        return (
            f"Keep full dim={full['dim']} (best MRR={best['mean_mrr']}). "
            "Lower dims did not match or beat full quality on this gold set."
        )
    drop = full["mean_mrr"] - best["mean_mrr"]
    if drop <= 0.02:
        return (
            f"Consider EMBEDDING_DIM={best['dim']} (MRR={best['mean_mrr']}, "
            f"vs full {full['mean_mrr']}; Δ≤0.02). Rebuild indexes after changing."
        )
    return (
        f"Prefer dim={full['dim']} (MRR={full['mean_mrr']}). "
        f"Best lower dim={best['dim']} loses {drop:.3f} MRR."
    )



def write_evaluation_report(
    rag_report: dict[str, Any] | None,
    dim_report: dict[str, Any] | None,
) -> Path:
    """Portfolio-style report (Lab 7 Part E + Lab 2/3/4 discipline)."""
    s = (rag_report or {}).get("summary") or {}
    rows = (rag_report or {}).get("rows") or []
    dim_rows = (dim_report or {}).get("rows") or []
    rec = (dim_report or {}).get("recommendation") or ""

    n_gold = s.get("n_questions") or len(rows)
    n_ret = s.get("n_retrieval_scored") or 0
    n_oos = s.get("n_unanswerable") or 0

    lines = [
        "# Evaluation Report — ArXiv Paper Suggester",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## 1. What it does",
        "",
        "The system builds a paper library from arXiv, ranks papers for a research interest,",
        "ingests full text, and answers questions **only** from those papers with citations.",
        "Embeddings: local **EmbeddingGemma 2** (Matryoshka dims supported).",
        "Generation: **Gemini** with optional **NVIDIA / OpenAI-compatible** failsafe.",
        "",
        "## 2. How well it works",
        "",
        f"Golden set size: **{n_gold}** questions "
        f"({n_ret} with expected papers for retrieval metrics; {n_oos} unanswerable/refusal).",
        "",
        "### Retrieval (RAG full-text index)",
        "",
        "| Metric | Point estimate | 95% CI (bootstrap) |",
        "|--------|----------------|--------------------|",
        f"| Mean MRR | {s.get('mean_mrr', '—')} | {s.get('mrr_ci95', '—')} |",
        f"| Mean Recall@k | {s.get('mean_recall@k', '—')} | {s.get('recall_ci95', '—')} |",
        f"| Mean hit@1 | {s.get('mean_hit_rate@1', '—')} | {s.get('hit1_ci95', '—')} |",
        f"| Mean top score | {s.get('mean_top_score', '—')} | — |",
        f"| Index | {s.get('index_papers', '—')} papers / {s.get('index_chunks', '—')} chunks | |",
        f"| Embedder | {s.get('embedding_model', '—')} @ dim={s.get('embedding_dim', '—')} | |",
        "",
        "**Statistical honesty (Lab 2):** CIs that overlap a baseline mean no claim of",
        "superiority. Prefer paired comparisons when comparing two system versions on the same items.",
        "",
        "### Refusal / grounding",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Refusal recall | {s.get('refusal_recall', '—')} |",
        f"| Refusal precision | {s.get('refusal_precision', '—')} |",
        f"| Refusal accuracy | {s.get('refusal_accuracy', '—')} |",
        f"| Cache hit rate | {s.get('cache_hit_rate', '—')} |",
        "",
        "Report **both** refusal precision and recall (Lab 4). Small n on unanswerable items",
        "makes ratios noisy — use raw counts from `last_eval.json`.",
        "",
        "### Stratification",
        "",
        "#### By kind",
        "",
    ]
    by_kind = s.get("by_kind") or {}
    if by_kind:
        lines += ["| Kind | n | MRR | Recall@k | Hit@1 |", "|------|---|-----|----------|-------|"]
        for kind, v in by_kind.items():
            lines.append(
                f"| {kind} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} |"
            )
    else:
        lines.append("_No data — run eval after ingesting papers._")

    lines += ["", "#### By difficulty", ""]
    by_diff = s.get("by_difficulty") or {}
    if by_diff:
        lines += ["| Difficulty | n | MRR | Recall@k | Hit@1 |", "|------------|---|-----|----------|-------|"]
        for d, v in by_diff.items():
            lines.append(
                f"| {d} | {v['n']} | {v['mean_mrr']} | {v['mean_recall@k']} | {v['mean_hit@1']} |"
            )
    else:
        lines.append("_No data._")

    lines += [
        "",
        "### Latency",
        "",
        f"p50 **{s.get('latency_p50_ms', '—')}** ms · p95 **{s.get('latency_p95_ms', '—')}** ms",
        "",
        "### Embedding dimension sweep (Matryoshka on library abstracts)",
        "",
    ]
    if dim_rows:
        lines += ["| Dim | MRR | Recall@10 | Hit@1 | ~B/vec |", "|-----|-----|-----------|-------|--------|"]
        for r in dim_rows:
            lines.append(
                f"| {r['dim']} | {r['mean_mrr']} | {r['mean_recall@10']} | "
                f"{r['mean_hit@1']} | {r['bytes_per_vector_approx']} |"
            )
        lines += ["", f"**Recommendation:** {rec}", ""]
    else:
        lines += ["_Run `--dim-sweep`._", ""]

    fail = s.get("fail_hint_counts") or {}
    lines += [
        "## 3. Where it fails",
        "",
        "Automated failure hints (Lab 5-inspired, not full diagnosis):",
        "",
        f"```",
        f"{json.dumps(fail, indent=2) if fail else '{}'}",
        f"```",
        "",
        "| Hint | Meaning | Typical fix |",
        "|------|---------|-------------|",
        "| `missing_or_ranking` | Expected paper not in top-k | Ingest paper or improve ranker |",
        "| `ranking` | Paper in top-k but not #1 | Rerank / raise k |",
        "| `false_answer` | Answered when should refuse | Stricter coverage / prompt |",
        "| `over_refuse` | Refused when answerable | Lower threshold / more context |",
        "",
        "Dominant cluster on your last run should drive the next change — one fix at a time (Lab 5).",
        "",
        "## 4. What it costs",
        "",
        "- Embeddings: **local** EmbeddingGemma 2 (no per-query embedding API).",
        "- Generation: Gemini or fallback LLM; cache cuts repeat cost.",
        "- Dim↓ reduces storage/RAM; quality trade-off is in the sweep table.",
        "",
        "## 5. How fast it is",
        "",
        "| Stage | Typical |",
        "|-------|---------|",
        "| Embed query (warm) | 10–50 ms |",
        "| Retrieve | <50 ms |",
        "| Generate | 2–8 s |",
        "| Cold model load | 5–15 s once |",
        "",
        "## 6. What it is not safe for",
        "",
        "- Clinical, legal, or financial decisions without human review.",
        "- Questions about papers **not** in the RAG index.",
        "- Over-interpreting refusal rates on small OOS slices.",
        "- Semantic cache near-threshold hits without audit.",
        "- Treating cosine scores as calibrated probabilities.",
        "",
        "## 7. What to do next (ranked)",
        "",
        "1. Ingest papers that gold marks as expected but index lacks (kills `missing_or_ranking`).",
        "2. Set `EMBEDDING_DIM` from the sweep; rebuild corpus + RAG indexes.",
        "3. Tune refusal threshold using precision/recall trade-off on the 20 OOS/edge items.",
        "4. Optional: cross-encoder rerank if `ranking` hints dominate.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "uv run python src/eval_suite.py --all",
        "uv run python src/eval_suite.py --all --with-generation",
        "```",
        "",
        "Artifacts: `results/rag_eval/last_eval.{json,md}`, `dim_sweep.json`, this file.",
        "",
        f"Golden set: `results/rag_eval/gold_questions.json` (n={n_gold}).",
        "",
    ]

    out = RESULTS_DIR / "EVALUATION_REPORT.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation suite")
    parser.add_argument("--all", action="store_true", help="RAG eval + dim sweep + report")
    parser.add_argument("--retrieval-only", action="store_true", help="RAG eval without Gemini")
    parser.add_argument("--with-generation", action="store_true", help="RAG eval with Gemini")
    parser.add_argument("--dim-sweep", action="store_true", help="Matryoshka dim sweep")
    parser.add_argument(
        "--dims",
        default="128,256,512,768",
        help="Comma-separated dims for sweep",
    )
    parser.add_argument("--report-only", action="store_true", help="Rebuild report from last JSON")
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()

    if not any(
        [args.all, args.retrieval_only, args.with_generation, args.dim_sweep, args.report_only]
    ):
        args.all = True

    rag_report = None
    dim_report = None

    if args.report_only:
        p = EVAL_DIR / "last_eval.json"
        d = EVAL_DIR / "dim_sweep.json"
        rag_report = json.loads(p.read_text()) if p.exists() else None
        dim_report = json.loads(d.read_text()) if d.exists() else None
        write_evaluation_report(rag_report, dim_report)
        return

    if args.all or args.retrieval_only or args.with_generation:
        from rag.evaluate import print_summary, run_eval

        skip = not args.with_generation and not (args.all and not args.retrieval_only)
        # --all defaults to retrieval-only first for cost; generation if --with-generation
        if args.all and not args.with_generation and not args.retrieval_only:
            skip = True
        if args.with_generation:
            skip = False
        if args.retrieval_only:
            skip = True
        print(f"[eval] running RAG eval (skip_generation={skip})…")
        rag_report = run_eval(top_k=args.k, skip_generation=skip)
        print_summary(rag_report)

    if args.all or args.dim_sweep:
        dims = [int(x) for x in args.dims.split(",") if x.strip()]
        print(f"[eval] dim sweep {dims}…")
        dim_report = run_dim_sweep(dims)

    if args.all or args.report_only or rag_report or dim_report:
        if rag_report is None and (EVAL_DIR / "last_eval.json").exists():
            rag_report = json.loads((EVAL_DIR / "last_eval.json").read_text())
        if dim_report is None and (EVAL_DIR / "dim_sweep.json").exists():
            dim_report = json.loads((EVAL_DIR / "dim_sweep.json").read_text())
        write_evaluation_report(rag_report, dim_report)


if __name__ == "__main__":
    main()
