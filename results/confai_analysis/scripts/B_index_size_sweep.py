#!/usr/bin/env python3
"""B. Retrieval-only index-size sweep via paper_id metadata filter (no rebuild).
Subsets: all gold expected papers + random others → totals 10, 100, 1000, full.
3 seeds. MRR / Recall@6 / Hit@1 overall + per kind.
Bootstrap CIs: (1) by question  (2) by expected-paper cluster.

  uv run python results/confai_analysis/B_index_size_sweep.py
  uv run python results/confai_analysis/B_index_size_sweep.py --sizes 10,100 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

def find_project_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists() or (p / "src").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]

ROOT = find_project_root()
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "results" / "confai_analysis"
TOP_K = 6


def _norm(pid: str) -> str:
    s = (pid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def metrics_for_question(expected: list[str], retrieved: list[str]) -> dict:
    exp = {_norm(x) for x in expected if x}
    ret = [_norm(x) for x in retrieved]
    if not exp:
        return {"recall@k": None, "mrr": None, "hit_rate@1": None}
    hit = any(r in exp for r in ret[:TOP_K])
    recall = 1.0 if hit else 0.0
    mrr = 0.0
    for i, r in enumerate(ret[:TOP_K], 1):
        if r in exp:
            mrr = 1.0 / i
            break
    hit1 = 1.0 if ret and ret[0] in exp else 0.0
    return {"recall@k": recall, "mrr": mrr, "hit_rate@1": hit1}


def bootstrap_mean(values: list[float], *, n_boot: int = 2000, seed: int = 0) -> dict:
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return {"mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def bootstrap_by_cluster(
    rows: list[dict], metric: str, *, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Resample expected-paper clusters; average metric within cluster then across."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(metric) is None:
            continue
        # cluster key = first expected paper id
        exp = r.get("expected_paper_ids") or ["_none_"]
        key = _norm(exp[0]) if exp else "_none_"
        by[key].append(float(r[metric]))
    if not by:
        return {"mean": None, "ci95": [None, None]}
    keys = list(by.keys())
    cluster_means = {k: float(np.mean(v)) for k, v in by.items()}
    overall = float(np.mean(list(cluster_means.values())))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample_keys = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        boots.append(float(np.mean([cluster_means[k] for k in sample_keys])))
    return {
        "mean": overall,
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_clusters": len(keys),
    }


def build_subset(
    gold_ids: list[str],
    all_ids: list[str],
    total: int | None,
    seed: int,
) -> list[str]:
    g = sorted({_norm(x) for x in gold_ids if x})
    pool = sorted({_norm(x) for x in all_ids if x} - set(g))
    if total is None or total <= 0:
        return sorted(set(g) | set(pool))  # full
    if total <= len(g):
        # still keep all gold; note actual size may exceed total
        return g
    need = total - len(g)
    rng = random.Random(seed)
    extra = rng.sample(pool, min(need, len(pool)))
    return sorted(set(g) | set(extra))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,100,1000,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    from rag.embed_store import list_ingested_papers, load_embedding_model, query_chunks

    gold = json.loads((ROOT / "results/rag_eval/gold_questions.json").read_text())
    answerable = [q for q in gold if not q.get("should_refuse")]
    gold_ids: list[str] = []
    for q in gold:
        gold_ids.extend(q.get("expected_paper_ids") or [])

    papers = list_ingested_papers()
    all_ids = [str(p.get("paper_id") or p.get("arxiv_id")) for p in papers]
    print(f"[B] ingested papers={len(all_ids)} answerable Qs={len(answerable)}", flush=True)

    model = load_embedding_model()
    sizes_raw = [s.strip() for s in args.sizes.split(",") if s.strip()]
    results = []

    for size_label in sizes_raw:
        total = None if size_label == "full" else int(size_label)
        for seed in range(args.seeds):
            subset = build_subset(gold_ids, all_ids, total, seed=1000 + seed * 17 + (total or 0))
            print(
                f"[B] size={size_label} seed={seed} subset_n={len(subset)} "
                f"(requested {size_label})",
                flush=True,
            )
            rows = []
            for q in answerable:
                hits = query_chunks(
                    q["question"],
                    top_k=args.top_k,
                    paper_ids=subset,
                    model=model,
                )
                retrieved = [h.get("paper_id") or "" for h in hits]
                m = metrics_for_question(q.get("expected_paper_ids") or [], retrieved)
                rows.append(
                    {
                        "id": q.get("id"),
                        "kind": q.get("kind"),
                        "expected_paper_ids": q.get("expected_paper_ids") or [],
                        "retrieved_paper_ids": retrieved,
                        "top_score": hits[0]["score"] if hits else 0.0,
                        **m,
                    }
                )

            overall = {}
            for metric in ("mrr", "recall@k", "hit_rate@1"):
                vals = [r[metric] for r in rows if r[metric] is not None]
                overall[metric] = {
                    "by_question": bootstrap_mean(vals, seed=seed),
                    "by_expected_paper_cluster": bootstrap_by_cluster(
                        rows, metric, seed=seed + 99
                    ),
                }

            by_kind = {}
            kinds = sorted({r["kind"] for r in rows})
            for kind in kinds:
                kr = [r for r in rows if r["kind"] == kind]
                by_kind[kind] = {}
                for metric in ("mrr", "recall@k", "hit_rate@1"):
                    vals = [r[metric] for r in kr if r[metric] is not None]
                    by_kind[kind][metric] = bootstrap_mean(vals, seed=seed)

            results.append(
                {
                    "size_label": size_label,
                    "requested_total": total,
                    "actual_subset_n": len(subset),
                    "seed": seed,
                    "n_questions": len(rows),
                    "overall": overall,
                    "by_kind": by_kind,
                    "note": (
                        f"Requested total ({total}) <= n_gold_expected ({len(g)}). Subset consists exactly of "
                        "the gold paper set (0 distractors), resulting in identical metrics across seeds."
                        if total is not None and total <= len(g)
                        else "Random sample of distractors added to gold paper set."
                    ),
                }
            )
            # checkpoint
            (OUT / "B_sweep_partial.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

    path = OUT / "B_index_size_sweep.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # short md
    lines = ["# B. Index-size sweep (metadata filter)", ""]
    for block in results:
        o = block["overall"]
        lines.append(
            f"## size={block['size_label']} seed={block['seed']} "
            f"subset_n={block['actual_subset_n']}"
        )
        for metric in ("mrr", "recall@k", "hit_rate@1"):
            bq = o[metric]["by_question"]
            bc = o[metric]["by_expected_paper_cluster"]
            lines.append(
                f"- **{metric}** by-question: {bq['mean']:.4f} CI{bq['ci95']}; "
                f"by-cluster: {bc['mean']:.4f} CI{bc['ci95']}"
            )
        lines.append("")
    (OUT / "B_index_size_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[B] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
