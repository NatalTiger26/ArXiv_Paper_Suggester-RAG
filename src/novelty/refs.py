"""Reference + abstract resolution: Semantic Scholar primary, OpenAlex backup."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from .ids import canonical_id, normalise_arxiv_id, strip_version

S2_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = "ArXivPaperSuggester-Novelty/0.1 (research; contact: local)"


def _s2_headers() -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if key:
        h["x-api-key"] = key.strip()
    return h


def _get(url: str, *, headers: dict | None = None, params: dict | None = None, timeout: int = 30) -> dict | list | None:
    try:
        r = requests.get(url, headers=headers or {"User-Agent": USER_AGENT}, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2.0)
            r = requests.get(url, headers=headers or {"User-Agent": USER_AGENT}, params=params, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def _external_ids_to_arxiv(ext: dict | None) -> str | None:
    if not ext:
        return None
    for key in ("ArXiv", "arXiv", "arxiv"):
        if ext.get(key):
            return normalise_arxiv_id(str(ext[key]))
    return None


def fetch_s2_paper(arxiv_id: str) -> dict[str, Any] | None:
    aid = strip_version(arxiv_id)
    fields = "paperId,externalIds,title,abstract,year,publicationDate,citationCount,fieldsOfStudy"
    data = _get(
        f"{S2_BASE}/paper/ARXIV:{aid}",
        headers=_s2_headers(),
        params={"fields": fields},
    )
    if not isinstance(data, dict) or data.get("paperId") is None:
        return None
    return data


def fetch_s2_references(arxiv_id: str, *, limit: int = 100) -> tuple[list[dict[str, Any]], str]:
    """Return (list of ref paper dicts, source_note)."""
    aid = strip_version(arxiv_id)
    fields = (
        "paperId,externalIds,title,abstract,year,citationCount,fieldsOfStudy"
    )
    # Try nested references on the paper endpoint first (one call when supported)
    data = _get(
        f"{S2_BASE}/paper/ARXIV:{aid}",
        headers=_s2_headers(),
        params={"fields": f"references.{fields}"},
    )
    refs: list[dict] = []
    if isinstance(data, dict) and data.get("references"):
        for item in data["references"]:
            # sometimes nested under citedPaper
            node = item.get("citedPaper") if isinstance(item, dict) and "citedPaper" in item else item
            if isinstance(node, dict) and (node.get("paperId") or node.get("title")):
                refs.append(node)
        if refs:
            return refs[:limit], "semanticscholar:nested"

    # Fallback: references endpoint
    data2 = _get(
        f"{S2_BASE}/paper/ARXIV:{aid}/references",
        headers=_s2_headers(),
        params={"fields": fields, "limit": min(limit, 100)},
    )
    if isinstance(data2, dict) and data2.get("data"):
        for item in data2["data"]:
            node = item.get("citedPaper") or item
            if isinstance(node, dict):
                refs.append(node)
        return refs[:limit], "semanticscholar:references"

    return [], "semanticscholar:empty"


def s2_node_to_record(node: dict[str, Any]) -> dict[str, Any] | None:
    ext = node.get("externalIds") or {}
    arxiv = _external_ids_to_arxiv(ext)
    s2_id = node.get("paperId")
    doi = ext.get("DOI") or ext.get("doi")
    cid = canonical_id(arxiv_id=arxiv, s2_id=s2_id, doi=doi)
    if not cid:
        return None
    title = (node.get("title") or "").strip() or None
    abstract = (node.get("abstract") or "").strip() or None
    year = node.get("year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None
    fos = node.get("fieldsOfStudy") or []
    if isinstance(fos, list):
        cats = [str(x) for x in fos if x]
    else:
        cats = []
    return {
        "canonical_id": cid,
        "arxiv_id": arxiv,
        "s2_id": s2_id,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "year": year,
        "categories": cats,
        "source_api": "semanticscholar",
        "has_abstract": bool(abstract),
    }


def fetch_openalex_by_arxiv(arxiv_id: str) -> dict[str, Any] | None:
    aid = strip_version(arxiv_id)
    data = _get(
        f"{OPENALEX_BASE}/works",
        params={
            "filter": f"ids.arxiv:{aid}",
            "per-page": 1,
            "mailto": os.getenv("OPENALEX_MAILTO", "novelty@localhost"),
        },
    )
    if not isinstance(data, dict):
        return None
    results = data.get("results") or []
    return results[0] if results else None


def _openalex_abstract(inv: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inv or not isinstance(inv, dict):
        return None
    try:
        max_pos = max(pos for positions in inv.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inv.items():
            for pos in positions:
                words[pos] = word
        text = " ".join(w for w in words if w).strip()
        return text or None
    except Exception:
        return None


def openalex_work_to_record(work: dict[str, Any]) -> dict[str, Any] | None:
    ids = work.get("ids") or {}
    arxiv = None
    if ids.get("arxiv"):
        arxiv = normalise_arxiv_id(str(ids["arxiv"]))
    # sometimes in locations
    doi = None
    if ids.get("doi"):
        doi = str(ids["doi"]).replace("https://doi.org/", "")
    oid = work.get("id")
    cid = canonical_id(arxiv_id=arxiv, openalex_id=oid, doi=doi)
    if not cid:
        return None
    abstract = _openalex_abstract(work.get("abstract_inverted_index"))
    title = (work.get("title") or work.get("display_name") or "").strip() or None
    year = work.get("publication_year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None
    concepts = work.get("concepts") or []
    cats = []
    for c in concepts[:8]:
        if isinstance(c, dict) and c.get("display_name"):
            cats.append(str(c["display_name"]))
    return {
        "canonical_id": cid,
        "arxiv_id": arxiv,
        "openalex_id": oid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "year": year,
        "categories": cats,
        "source_api": "openalex",
        "has_abstract": bool(abstract),
    }


def fetch_openalex_references(arxiv_id: str, *, limit: int = 100) -> tuple[list[dict[str, Any]], str]:
    work = fetch_openalex_by_arxiv(arxiv_id)
    if not work:
        return [], "openalex:no_work"
    # referenced_works is a list of OpenAlex URLs
    ref_urls = (work.get("referenced_works") or [])[:limit]
    out: list[dict] = []
    for url in ref_urls:
        wid = str(url).rstrip("/").split("/")[-1]
        data = _get(
            f"{OPENALEX_BASE}/works/{wid}",
            params={"mailto": os.getenv("OPENALEX_MAILTO", "novelty@localhost")},
        )
        if isinstance(data, dict):
            rec = openalex_work_to_record(data)
            if rec:
                out.append(rec)
        time.sleep(0.05)  # be polite
    return out, "openalex:referenced_works"


def resolve_focal_and_refs(
    arxiv_id: str,
    *,
    max_refs: int = 80,
) -> dict[str, Any]:
    """
    Resolve focal paper metadata + reference records.

    Returns dict with keys: focal, refs, source, stats
    """
    aid = strip_version(arxiv_id)
    focal: dict[str, Any] | None = None
    refs: list[dict[str, Any]] = []
    source = ""

    s2 = fetch_s2_paper(aid)
    if s2:
        focal = s2_node_to_record(s2)
        raw_refs, src = fetch_s2_references(aid, limit=max_refs)
        source = src
        for node in raw_refs:
            rec = s2_node_to_record(node)
            if rec:
                refs.append(rec)

    if focal is None or not refs:
        # OpenAlex backup for missing focal or empty refs
        oa = fetch_openalex_by_arxiv(aid)
        if oa and focal is None:
            focal = openalex_work_to_record(oa)
        if not refs:
            oa_refs, oa_src = fetch_openalex_references(aid, limit=max_refs)
            if oa_refs:
                refs = oa_refs
                source = oa_src if not source else source + "+" + oa_src

    if focal is None:
        # minimal focal from arxiv id alone
        focal = {
            "canonical_id": canonical_id(arxiv_id=aid),
            "arxiv_id": aid,
            "title": None,
            "abstract": None,
            "year": None,
            "categories": [],
            "source_api": "local",
            "has_abstract": False,
        }

    with_abs = sum(1 for r in refs if r.get("has_abstract"))
    return {
        "focal": focal,
        "refs": refs,
        "source": source or "none",
        "stats": {
            "n_refs": len(refs),
            "n_with_abstract": with_abs,
            "frac_with_abstract": (with_abs / len(refs)) if refs else 0.0,
            "arxiv_id": aid,
        },
    }
