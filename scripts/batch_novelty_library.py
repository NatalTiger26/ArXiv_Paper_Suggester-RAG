#!/usr/bin/env python3
"""
Batch Novelty Subspace Pipeline for ArXiv Scientific Discovery Platform.

Runs 2D reference cone projection and NNLS synthesis attribution across
library catalog papers, classifying them into synergistic blends, linear extensions,
and paradigm shifts.

Usage:
    uv run python scripts/batch_novelty_library.py --limit 50
    uv run python scripts/batch_novelty_library.py --category cs.LG --min-citations 10
    uv run python scripts/batch_novelty_library.py --resume --output-json results/novelty_batch.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_DB = PROJECT_ROOT / "data" / "catalog.sqlite3"
DEFAULT_JSON = PROJECT_ROOT / "results" / "batch_novelty_report.json"
DEFAULT_MD = PROJECT_ROOT / "results" / "batch_novelty_report.md"


def classify_geometry(m: float, delta: float, r2_plus: float) -> str:
    """Classify paper discovery mode from 2D reference cone metrics."""
    res_gap = (max(0.0, 1.0 - r2_plus)) ** 0.5
    if 1.0 - r2_plus >= 0.45 or res_gap >= 0.67:
        return "Paradigm Shift / Subspace Outlier"
    elif delta >= 0.05:
        return "High Synergy Cone Blend"
    elif delta >= 0.02:
        return "Moderate Synthesis"
    elif m >= 0.60:
        return "Linear / Incremental Extension"
    else:
        return "Diffused Reference Combination"


def load_candidate_papers(
    db_path: Path,
    limit: int = 50,
    category: str | None = None,
    min_citations: int = 0,
) -> list[dict[str, Any]]:
    """Query papers from SQLite catalog prioritized by citation count."""
    if not db_path.exists():
        print(f"[batch-novelty] Catalog database not found: {db_path}", file=sys.stderr)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    clauses = ["1=1"]
    params: list[Any] = []

    if category:
        clauses.append("categories LIKE ?")
        params.append(f"%{category}%")

    if min_citations > 0:
        clauses.append("citation_count >= ?")
        params.append(min_citations)

    where = " AND ".join(clauses)
    query = f"""
        SELECT arxiv_id, title, abstract, year, citation_count, categories
        FROM papers
        WHERE {where}
        ORDER BY citation_count DESC, year DESC
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_batch_novelty(
    limit: int = 50,
    category: str | None = None,
    min_citations: int = 0,
    resume: bool = True,
    output_json: Path = DEFAULT_JSON,
    output_md: Path = DEFAULT_MD,
    sleep_s: float = 0.25,
) -> dict[str, Any]:
    from src.novelty.cache import connect, purge_corrupted_embeddings
    from src.novelty.score import score_paper
    from src.rag.embed_store import drain_mps_cache, load_embedding_model

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    # Ensure novelty cache has no corrupted NaN vectors from earlier runs
    try:
        conn = connect()
        purged = purge_corrupted_embeddings(conn)
        if purged > 0:
            print(f"[batch-novelty] Cleaned {purged} corrupted NaN vectors from novelty cache.")
        conn.close()
    except Exception as e:
        print(f"[batch-novelty] Cache purge check warning: {e}")

    # Load existing results if resuming
    existing_records: dict[str, dict[str, Any]] = {}
    if resume and output_json.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for item in prev_data.get("results", []):
                    m_val = item.get("m")
                    d_val = item.get("Delta")
                    if (
                        item.get("ok")
                        and item.get("arxiv_id")
                        and m_val is not None
                        and str(m_val) != "nan"
                        and str(d_val) != "nan"
                    ):
                        existing_records[item["arxiv_id"]] = item
            print(f"[batch-novelty] Loaded {len(existing_records)} existing records for resumption.")
        except Exception as e:
            print(f"[batch-novelty] Could not load prior results: {e}")

    candidates = load_candidate_papers(
        CATALOG_DB, limit=limit, category=category, min_citations=min_citations
    )
    if not candidates:
        print("[batch-novelty] No candidate papers matched the criteria.")
        return {"total": 0, "scored": 0, "results": []}

    print(f"\n[batch-novelty] Preloading local embedding model for fast batch inference...")
    model = load_embedding_model()
    print("[batch-novelty] Model ready. Starting batch evaluation...\n" + "=" * 76)

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, paper in enumerate(candidates, 1):
        aid = paper["arxiv_id"]
        title = paper.get("title") or "Untitled"
        year = paper.get("year")
        cites = paper.get("citation_count") or 0

        # Check resume cache
        if resume and aid in existing_records:
            print(f"[{idx:3d}/{len(candidates)}] {aid} | (cached) {title[:48]}...")
            results.append(existing_records[aid])
            continue

        print(f"[{idx:3d}/{len(candidates)}] Scoring {aid} | {title[:48]}...", flush=True)

        try:
            profile = score_paper(aid, model=model, allow_partial=True)
            if not profile.get("ok"):
                record = {
                    "arxiv_id": aid,
                    "title": title,
                    "year": year,
                    "citation_count": cites,
                    "ok": False,
                    "error": profile.get("error", "scoring_failed"),
                    "message": profile.get("message", ""),
                }
            else:
                import math

                m = float(profile.get("m", 0.0))
                delta = float(profile.get("Delta", 0.0))
                r2_plus = float(profile.get("R2_plus", 0.0))

                if not (math.isfinite(m) and math.isfinite(delta) and math.isfinite(r2_plus)):
                    record = {
                        "arxiv_id": aid,
                        "title": title,
                        "year": year,
                        "citation_count": cites,
                        "ok": False,
                        "error": "non_finite_metrics",
                        "message": f"Non-finite metrics: m={m}, delta={delta}, r2_plus={r2_plus}",
                    }
                else:
                    res_gap = float(max(0.0, 1.0 - r2_plus) ** 0.5)
                    classification = classify_geometry(m, delta, r2_plus)

                    # Format top parents
                    attr_list = []
                    for a in profile.get("attribution", [])[:5]:
                        attr_list.append({
                            "paper_id": a.get("paper_id"),
                            "title": a.get("title"),
                            "weight": round(float(a.get("weight", 0.0)), 4),
                            "share": round(float(a.get("share", 0.0)), 1),
                            "url": a.get("url"),
                        })

                    record = {
                        "arxiv_id": aid,
                        "title": title,
                        "year": year,
                        "citation_count": cites,
                        "ok": True,
                        "m": round(m, 4),
                        "Delta": round(delta, 4),
                        "R2_plus": round(r2_plus, 4),
                        "novelty_residual": round(res_gap, 4),
                        "novelty_score": round(1.0 - r2_plus, 4),
                        "classification": classification,
                        "attribution": attr_list,
                        "n_refs_used": profile.get("n_refs_used", 0),
                        "d_eff": profile.get("d_eff"),
                    }
                    print(f"       -> m={m:.3f}  Δ={delta:.3f}  R²+={r2_plus:.3f}  Res={res_gap:.3f}  [{classification}]")

            results.append(record)

            # Politeness delay between remote queries
            if sleep_s > 0:
                time.sleep(sleep_s)

        except Exception as e:
            print(f"       -> Error: {e}")
            results.append({
                "arxiv_id": aid,
                "title": title,
                "year": year,
                "citation_count": cites,
                "ok": False,
                "error": str(e),
            })
        finally:
            drain_mps_cache()

        # Save checkpoint every 10 papers (and update markdown report every 50)
        if idx % 10 == 0:
            chk = _save_json_report(output_json, results, limit, category, min_citations)
            if idx % 50 == 0:
                _generate_markdown_report(output_md, chk)

    total_time = time.perf_counter() - t_start
    print("=" * 76)
    print(f"[batch-novelty] Processed {len(results)} papers in {total_time:.1f}s.")

    # Final Saves
    report_data = _save_json_report(output_json, results, limit, category, min_citations)
    _generate_markdown_report(output_md, report_data)

    print(f"[batch-novelty] JSON report saved: {output_json}")
    print(f"[batch-novelty] Markdown report saved: {output_md}")
    return report_data


def _save_json_report(
    path: Path,
    results: list[dict[str, Any]],
    limit: int,
    category: str | None,
    min_citations: int,
) -> dict[str, Any]:
    scored_ok = [
        r for r in results
        if r.get("ok")
        and r.get("m") is not None
        and str(r.get("m")) != "nan"
        and str(r.get("Delta")) != "nan"
    ]
    stats: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_requested": "all" if limit == 0 else limit,
        "total_evaluated": len(results),
        "total_successful": len(scored_ok),
        "category_filter": category,
        "min_citations_filter": min_citations,
    }
    if scored_ok:
        stats["mean_m"] = round(sum(r["m"] for r in scored_ok) / len(scored_ok), 4)
        stats["mean_Delta"] = round(sum(r["Delta"] for r in scored_ok) / len(scored_ok), 4)
        stats["mean_R2_plus"] = round(sum(r["R2_plus"] for r in scored_ok) / len(scored_ok), 4)
        stats["mean_residual"] = round(sum(r["novelty_residual"] for r in scored_ok) / len(scored_ok), 4)

    payload = {"stats": stats, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def _generate_markdown_report(path: Path, report_data: dict[str, Any]) -> None:
    stats = report_data.get("stats", {})
    results = report_data.get("results", [])
    valid = [r for r in results if r.get("ok")]

    # Count classifications
    class_counts: dict[str, int] = {}
    for r in valid:
        c = r.get("classification", "Unknown")
        class_counts[c] = class_counts.get(c, 0) + 1

    lines = [
        "# Batch Novelty Subspace Analysis Report",
        "",
        f"**Generated:** {stats.get('timestamp', '—')}  ",
        f"**Papers Evaluated:** {stats.get('total_evaluated', 0)}  ",
        f"**Success Rate:** {stats.get('total_successful', 0)} / {stats.get('total_evaluated', 0)} "
        f"({(stats.get('total_successful', 0) / max(1, stats.get('total_evaluated', 0)) * 100):.1f}%)",
        "",
        "## 1. Aggregate Geometric Metrics",
        "",
        "| Metric | Definition | Cohort Mean | Interpretation |",
        "| :--- | :--- | :--- | :--- |",
        f"| **m** | Cosine to closest parent | `{stats.get('mean_m', 0.0):.4f}` | Baseline anchor strength to closest single prior work |",
        f"| **Δ (Delta)** | Multi-parent cone synergy | `{stats.get('mean_Delta', 0.0):.4f}` | Energy gain from non-negative combination of multiple parents |",
        f"| **R²₊** | Cone reconstruction energy | `{stats.get('mean_R2_plus', 0.0):.4f}` | Proportion of embedding explained by the reference subspace |",
        f"| **√(1 - R²₊)** | Novelty residual gap | `{stats.get('mean_residual', 0.0):.4f}` | Orthogonal departure distance from reference cone to unit circle |",
        "",
        "## 2. Discovery Mode Distribution",
        "",
        "| Discovery Classification | Paper Count | Percentage |",
        "| :--- | :---: | :---: |",
    ]

    total_valid = len(valid) or 1
    for c_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| **{c_name}** | {count} | {(count / total_valid * 100):.1f}% |")

    # Top Synergistic Papers
    lines.extend([
        "",
        "## 3. Top Synergistic Syntheses (Highest Multi-Parent Synergy Δ)",
        "Papers that achieve dramatic innovation by non-linearly fusing multiple distinct research threads:",
        "",
        "| arXiv ID | Title | Year | Citations | Δ (Synergy) | m (Anchor) | R²₊ | Mode |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    top_synergy = sorted(valid, key=lambda x: -x.get("Delta", 0.0))[:10]
    for p in top_synergy:
        title = (p.get("title") or "")[:45].replace("|", "-")
        aid = p.get("arxiv_id")
        lines.append(
            f"| [{aid}](https://arxiv.org/abs/{aid}) | {title}... | {p.get('year') or '—'} | "
            f"{p.get('citation_count', 0):,} | **{p.get('Delta', 0.0):.4f}** | {p.get('m', 0.0):.3f} | "
            f"{p.get('R2_plus', 0.0):.3f} | {p.get('classification')} |"
        )

    # Top Paradigm Shifts
    lines.extend([
        "",
        "## 4. Top Subspace Outliers / Paradigm Shifts (Highest Residual √(1 - R²₊))",
        "Papers with large orthogonal departures that cannot be reconstructed from their references:",
        "",
        "| arXiv ID | Title | Year | Citations | Novelty Residual | R²₊ | m (Anchor) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    top_novel = sorted(valid, key=lambda x: -x.get("novelty_residual", 0.0))[:10]
    for p in top_novel:
        title = (p.get("title") or "")[:45].replace("|", "-")
        aid = p.get("arxiv_id")
        lines.append(
            f"| [{aid}](https://arxiv.org/abs/{aid}) | {title}... | {p.get('year') or '—'} | "
            f"{p.get('citation_count', 0):,} | **{p.get('novelty_residual', 0.0):.4f}** | "
            f"{p.get('R2_plus', 0.0):.3f} | {p.get('m', 0.0):.3f} |"
        )

    # Top Anchored Works
    lines.extend([
        "",
        "## 5. Top Foundationally Rooted Works (Highest Single Parent m)",
        "Works that build directly and deeply on a primary ancestor paper:",
        "",
        "| arXiv ID | Title | Year | Citations | m (Anchor) | Δ (Synergy) | Primary Parent Attribution |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])
    top_anchored = sorted(valid, key=lambda x: -x.get("m", 0.0))[:10]
    for p in top_anchored:
        title = (p.get("title") or "")[:40].replace("|", "-")
        aid = p.get("arxiv_id")
        attr = p.get("attribution", [])
        top_parent_str = "None"
        if attr:
            top_parent_str = f"`{attr[0].get('paper_id')}` (λ={attr[0].get('weight', 0):.2f})"
        lines.append(
            f"| [{aid}](https://arxiv.org/abs/{aid}) | {title}... | {p.get('year') or '—'} | "
            f"{p.get('citation_count', 0):,} | **{p.get('m', 0.0):.4f}** | {p.get('Delta', 0.0):.3f} | "
            f"{top_parent_str} |"
        )

    lines.extend([
        "",
        "---",
        "### Methodology & Theoretical Foundations",
        "Every paper embedding $\\mathbf{x}$ is projected onto the non-negative convex cone of its cited references $\\mathbf{Y} = [\\mathbf{y}_1, \\dots, \\mathbf{y}_k]$ via Non-Negative Least Squares (NNLS):",
        "",
        "$$\\boldsymbol{\\lambda}^* = \\arg\\min_{\\boldsymbol{\\lambda} \\ge 0} \\|\\mathbf{x} - \\mathbf{Y}\\boldsymbol{\\lambda}\\|^2, \\quad \\hat{\\mathbf{x}} = \\mathbf{Y}\\boldsymbol{\\lambda}^*$$",
        "",
        "The reconstructed energy is $R^2_+ = \\|\\hat{\\mathbf{x}}\\|^2$, the foundational anchor is $m = \\max_j (\\mathbf{x}^T \\mathbf{y}_j)$, and multi-parent synergy is defined as $\\Delta = R^2_+ - \\max(m, 0)^2$.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch Novelty Subspace Pipeline for ArXiv Catalog"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of papers to evaluate (0 for all)"
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Filter by arXiv category (e.g. cs.LG)"
    )
    parser.add_argument(
        "--min-citations", type=int, default=0, help="Minimum global citation count"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Recalculate already scored papers"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to output JSON summary",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_MD,
        help="Path to output Markdown report",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep delay between external API calls (seconds)",
    )

    args = parser.parse_args()
    run_batch_novelty(
        limit=args.limit,
        category=args.category,
        min_citations=args.min_citations,
        resume=not args.no_resume,
        output_json=args.output_json,
        output_md=args.output_md,
        sleep_s=args.sleep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
