"""Aggregate citation counts and export libraries / ingest lists."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT

from .collect import library_id_set
from .store import citation_counts, connect, example_citers, stats

OUT_DIR = PROJECT_ROOT / "results" / "citation_library"


def build_ranking(
    *,
    min_count: int = 2,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    lib = library_id_set()
    rows = citation_counts(conn)
    ranked: list[dict[str, Any]] = []
    for r in rows:
        cnt = int(r["library_cite_count"] or 0)
        if cnt < min_count:
            continue
        arxiv = (r["arxiv_id"] or "") or None
        if arxiv:
            arxiv = str(arxiv).strip()
        in_lib = bool(arxiv and arxiv in lib)
        ranked.append(
            {
                "library_cite_count": cnt,
                "title": r["title"] or "",
                "arxiv_id": arxiv or "",
                "year": r["year"],
                "doi": r["doi"] or "",
                "s2_id": r["s2_id"] or "",
                "canonical_id": r["canonical_id"],
                "global_citation_count": r["global_citation_count"],
                "in_library": in_lib,
                "example_citers": example_citers(conn, r["canonical_id"], limit=10),
            }
        )
    st = stats(conn)
    conn.close()
    return ranked


def write_outputs(
    ranked: list[dict[str, Any]],
    *,
    min_count: int,
    top: int = 50,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "citation_counts.csv"
    json_path = out_dir / "citation_counts.json"
    md_path = out_dir / "summary.md"
    candidates_path = out_dir / "candidates_not_in_library.csv"
    ids_path = out_dir / "candidates_arxiv_ids.txt"
    corpus_path = out_dir / "importance_corpus.json"

    fields = [
        "library_cite_count",
        "in_library",
        "title",
        "arxiv_id",
        "year",
        "doi",
        "canonical_id",
        "global_citation_count",
        "example_citers",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in ranked:
            r = dict(row)
            r["example_citers"] = ";".join(r.get("example_citers") or [])
            w.writerow(r)

    # candidates: not in library, has arxiv id, sorted by count
    candidates = [
        r
        for r in ranked
        if not r.get("in_library") and r.get("arxiv_id")
    ]
    with open(candidates_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in candidates:
            r = dict(row)
            r["example_citers"] = ";".join(r.get("example_citers") or [])
            w.writerow(r)

    ids_path.write_text(
        "\n".join(r["arxiv_id"] for r in candidates if r.get("arxiv_id")) + "\n",
        encoding="utf-8",
    )

    # corpus-shaped JSON for merge into library later
    importance_papers = []
    for r in candidates:
        importance_papers.append(
            {
                "arxiv_id": r["arxiv_id"],
                "title": r.get("title") or "",
                "year": r.get("year"),
                "doi": r.get("doi") or "",
                "library_cite_count": r["library_cite_count"],
                "source": "citation_library",
            }
        )
    corpus_path.write_text(
        json.dumps(importance_papers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = {
        "min_count": min_count,
        "n_ranked": len(ranked),
        "n_candidates_not_in_library": len(candidates),
        "refs": ranked[:5000],  # cap json size
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# In-library citation analysis",
        "",
        "Each **library_cite_count** is how many papers *in your library* list this work as a reference.",
        "",
        f"- Min count: **{min_count}**",
        f"- Ranked works: **{len(ranked)}**",
        f"- Candidates not in library (have arXiv id): **{len(candidates)}**",
        "",
        "## Top overall",
        "",
        "| # | Count | In lib | arXiv | Title |",
        "|--:|------:|:------:|-------|-------|",
    ]
    for i, row in enumerate(ranked[:top], 1):
        title = (row.get("title") or "")[:70].replace("|", "/")
        lines.append(
            f"| {i} | {row['library_cite_count']} | "
            f"{'yes' if row.get('in_library') else ''} | "
            f"{row.get('arxiv_id') or '—'} | {title} |"
        )
    lines += [
        "",
        "## Top candidates to add (not in library)",
        "",
        "| # | Count | arXiv | Title |",
        "|--:|------:|-------|-------|",
    ]
    for i, row in enumerate(candidates[:top], 1):
        title = (row.get("title") or "")[:70].replace("|", "/")
        lines.append(
            f"| {i} | {row['library_cite_count']} | {row.get('arxiv_id')} | {title} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Full ranking: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- Candidates CSV: `{candidates_path.relative_to(PROJECT_ROOT)}`",
        f"- arXiv ids (for ingest): `{ids_path.relative_to(PROJECT_ROOT)}`",
        f"- Importance corpus JSON: `{corpus_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## Next steps",
        "",
        "```bash",
        "# Add top candidates into the main library",
        f"uv run python src/corpus_builder.py from-ids $(head -50 {ids_path.relative_to(PROJECT_ROOT)} | tr '\\n' ' ')",
        "# Or: merge importance_corpus.json manually / rebuild index",
        f"# Ingest full text for Q&A:",
        f"uv run python -m src.rag.cli ingest $(head -20 {ids_path.relative_to(PROJECT_ROOT)} | tr '\\n' ' ')",
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    paths = {
        "csv": csv_path,
        "json": json_path,
        "summary": md_path,
        "candidates_csv": candidates_path,
        "candidate_ids": ids_path,
        "importance_corpus": corpus_path,
    }
    for k, p in paths.items():
        print(f"[cite] wrote {p}")
    return paths
