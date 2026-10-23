"""Enrich missing titles / arXiv ids for highly cited works via S2 batch API."""

from __future__ import annotations

from typing import Any

from .store import connect, papers_missing_title, upsert_paper


def enrich_missing_titles(
    *,
    limit: int = 200,
    db_path=None,
) -> dict[str, Any]:
    """
    For top cited works with empty titles, batch-fetch S2 metadata and upsert.
    Safe to re-run; only touches rows still missing title.
    """
    try:
        from novelty.refs import fetch_s2_papers_batch, s2_node_to_record
        from novelty.ids import canonical_id
    except ImportError:
        from src.novelty.refs import fetch_s2_papers_batch, s2_node_to_record  # type: ignore
        from src.novelty.ids import canonical_id  # type: ignore

    conn = connect(db_path)
    rows = papers_missing_title(conn, limit=limit)
    # Map s2 paperId from canonical_id s2:... or papers.s2_id
    id_list: list[str] = []
    cid_by_s2: dict[str, str] = {}
    for r in rows:
        cid = r["canonical_id"]
        s2 = (r["s2_id"] or "").strip()
        if not s2 and cid and str(cid).startswith("s2:"):
            s2 = str(cid).split(":", 1)[1]
        if s2:
            id_list.append(s2)
            cid_by_s2[s2] = cid

    if not id_list:
        conn.close()
        return {"queried": 0, "updated": 0, "note": "nothing missing or no s2 ids"}

    print(f"[cite] enrich titles for {len(id_list)} cited works via S2 batch…", flush=True)
    updated = 0
    # S2 allows up to 500; chunk by 100 for safety
    for i in range(0, len(id_list), 100):
        chunk = id_list[i : i + 100]
        nodes = fetch_s2_papers_batch(chunk)
        for node in nodes:
            if not isinstance(node, dict) or not node.get("paperId"):
                continue
            rec = s2_node_to_record(node)
            if not rec:
                continue
            # Preserve original edge canonical_id if we already used s2:...
            s2 = node["paperId"]
            if s2 in cid_by_s2:
                rec["canonical_id"] = cid_by_s2[s2]
            upsert_paper(conn, rec)
            if rec.get("title"):
                updated += 1
        conn.commit()

    conn.close()
    print(f"[cite] enrich done: updated_titles≈{updated}", flush=True)
    return {"queried": len(id_list), "updated": updated}
