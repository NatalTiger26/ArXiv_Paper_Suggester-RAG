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


# Shared adaptive pause after 429 so callers can space subsequent requests
_LAST_RATE_LIMIT_AT = 0.0
_RATE_LIMIT_STREAK = 0


def _get(url: str, *, headers: dict | None = None, params: dict | None = None, timeout: int = 30) -> dict | list | None:
    global _LAST_RATE_LIMIT_AT, _RATE_LIMIT_STREAK
    try:
        r = requests.get(url, headers=headers or {"User-Agent": USER_AGENT}, params=params, timeout=timeout)
        if r.status_code == 429:
            _RATE_LIMIT_STREAK += 1
            _LAST_RATE_LIMIT_AT = time.time()
            # Exponential backoff: 4s, 8s, 16s… capped at 45s; up to 3 retries
            wait = min(45.0, 4.0 * (2 ** min(_RATE_LIMIT_STREAK - 1, 3)))
            print(f"[refs] HTTP 429 — backoff {wait:.0f}s (streak={_RATE_LIMIT_STREAK})", flush=True)
            time.sleep(wait)
            r = requests.get(url, headers=headers or {"User-Agent": USER_AGENT}, params=params, timeout=timeout)
            if r.status_code == 429:
                wait2 = min(60.0, wait * 1.5)
                print(f"[refs] HTTP 429 again — backoff {wait2:.0f}s", flush=True)
                time.sleep(wait2)
                r = requests.get(url, headers=headers or {"User-Agent": USER_AGENT}, params=params, timeout=timeout)
        if r.status_code == 200:
            _RATE_LIMIT_STREAK = max(0, _RATE_LIMIT_STREAK - 1)
            return r.json()
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def suggested_inter_request_sleep(base: float = 0.35) -> float:
    """Recommend pause between papers; grows after recent rate limits."""
    global _LAST_RATE_LIMIT_AT, _RATE_LIMIT_STREAK
    extra = 0.0
    if _RATE_LIMIT_STREAK > 0:
        extra = min(8.0, 1.5 * _RATE_LIMIT_STREAK)
    if _LAST_RATE_LIMIT_AT and (time.time() - _LAST_RATE_LIMIT_AT) < 30:
        extra = max(extra, 2.0)
    # No free-tier key → be gentler by default
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if not key:
        base = max(base, 1.0)
    return base + extra


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


def fetch_s2_paper_with_refs(
    arxiv_id: str,
    *,
    limit: int = 100,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """
    One S2 call: focal metadata + nested references (halves API traffic vs two calls).
    """
    aid = strip_version(arxiv_id)
    ref_fields = "paperId,externalIds,title,year,citationCount"
    focal_fields = (
        f"paperId,externalIds,title,abstract,year,publicationDate,"
        f"citationCount,fieldsOfStudy,references.{ref_fields}"
    )
    keys = [
        f"ARXIV:{aid}",
        f"DOI:10.48550/arXiv.{aid}",
        f"DOI:10.48550/arxiv.{aid}",
    ]
    data = None
    used = ""
    for key in keys:
        data = _get(
            f"{S2_BASE}/paper/{key}",
            headers=_s2_headers(),
            params={"fields": focal_fields},
        )
        if isinstance(data, dict) and data.get("paperId"):
            used = key
            break
    if not isinstance(data, dict) or not data.get("paperId"):
        return None, [], "semanticscholar:empty"

    refs: list[dict] = []
    for item in data.get("references") or []:
        node = item.get("citedPaper") if isinstance(item, dict) and "citedPaper" in item else item
        if isinstance(node, dict) and (node.get("paperId") or node.get("title")):
            refs.append(node)
        if len(refs) >= limit:
            break

    # If nested refs missing, one extra call to /references
    if not refs:
        data2 = _get(
            f"{S2_BASE}/paper/{data['paperId']}/references",
            headers=_s2_headers(),
            params={
                "fields": "paperId,externalIds,title,year,citationCount",
                "limit": min(limit, 100),
            },
        )
        if isinstance(data2, dict) and data2.get("data"):
            for item in data2["data"]:
                node = item.get("citedPaper") or item
                if isinstance(node, dict):
                    refs.append(node)
            return data, refs[:limit], "semanticscholar:references"

    note = "semanticscholar:nested" if refs else "semanticscholar:focal_only"
    if used and used != f"ARXIV:{aid}":
        note += f":{used}"
    return data, refs[:limit], note


def fetch_s2_references(
    arxiv_id: str,
    *,
    limit: int = 100,
    s2_paper_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return (list of ref paper dicts, source_note). Prefer combined fetch when possible."""
    if not s2_paper_id:
        _, refs, note = fetch_s2_paper_with_refs(arxiv_id, limit=limit)
        return refs, note

    fields = "paperId,externalIds,title,year,citationCount"
    data = _get(
        f"{S2_BASE}/paper/{s2_paper_id}",
        headers=_s2_headers(),
        params={"fields": f"references.{fields}"},
    )
    refs: list[dict] = []
    if isinstance(data, dict) and data.get("references"):
        for item in data["references"]:
            node = item.get("citedPaper") if isinstance(item, dict) and "citedPaper" in item else item
            if isinstance(node, dict) and (node.get("paperId") or node.get("title")):
                refs.append(node)
        if refs:
            return refs[:limit], "semanticscholar:nested"

    data2 = _get(
        f"{S2_BASE}/paper/{s2_paper_id}/references",
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


def fetch_s2_papers_batch(paper_ids: list[str]) -> list[dict[str, Any]]:
    """Batch metadata for S2 paper ids (title, externalIds, year). Max 500 per request."""
    ids = [p for p in paper_ids if p][:500]
    if not ids:
        return []
    # S2 batch endpoint
    try:
        r = requests.post(
            f"{S2_BASE}/paper/batch",
            headers={**_s2_headers(), "Content-Type": "application/json"},
            params={"fields": "paperId,externalIds,title,year,citationCount,abstract"},
            json={"ids": ids},
            timeout=60,
        )
        if r.status_code == 429:
            print("[refs] batch 429 — sleep 8s", flush=True)
            time.sleep(8.0)
            r = requests.post(
                f"{S2_BASE}/paper/batch",
                headers={**_s2_headers(), "Content-Type": "application/json"},
                params={"fields": "paperId,externalIds,title,year,citationCount,abstract"},
                json={"ids": ids},
                timeout=60,
            )
        if r.status_code >= 400:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


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
    """Resolve OpenAlex work; always re-fetch by id so referenced_works is present."""
    aid = strip_version(arxiv_id)
    mailto = os.getenv("OPENALEX_MAILTO", "novelty@localhost")
    filters = [
        f"ids.arxiv:{aid}",
        f"doi:10.48550/arXiv.{aid}",
        f"doi:10.48550/arxiv.{aid}",
    ]
    work = None
    for filt in filters:
        data = _get(
            f"{OPENALEX_BASE}/works",
            params={"filter": filt, "per-page": 1, "mailto": mailto},
        )
        if isinstance(data, dict) and (data.get("results") or []):
            work = data["results"][0]
            break
    if work is None:
        # title-ish fallback for famous papers when id filters fail
        data = _get(
            f"{OPENALEX_BASE}/works",
            params={
                "search": aid,
                "filter": "type:article",
                "per-page": 5,
                "mailto": mailto,
            },
        )
        if isinstance(data, dict):
            for w in data.get("results") or []:
                ids = w.get("ids") or {}
                ax = str(ids.get("arxiv") or "")
                if aid in ax.replace("abs/", "/"):
                    work = w
                    break
    if not work:
        return None
    # List results often omit referenced_works — fetch full record
    oid = work.get("id")
    if oid:
        full = _get(str(oid), params={"mailto": mailto})
        if isinstance(full, dict) and full.get("id"):
            return full
    return work


def fetch_s2_paper_flexible(arxiv_id: str) -> dict[str, Any] | None:
    """Try several S2 id forms (ARXIV:, DOI, search)."""
    aid = strip_version(arxiv_id)
    for pid in (f"ARXIV:{aid}", f"DOI:10.48550/arXiv.{aid}", f"DOI:10.48550/arxiv.{aid}"):
        data = _get(
            f"{S2_BASE}/paper/{pid}",
            headers=_s2_headers(),
            params={"fields": "paperId,externalIds,title,abstract,year,publicationDate,citationCount,fieldsOfStudy"},
        )
        if isinstance(data, dict) and data.get("paperId"):
            return data
    # search fallback
    data = _get(
        f"{S2_BASE}/paper/search",
        headers=_s2_headers(),
        params={"query": f"arXiv:{aid}", "limit": 3, "fields": "paperId,externalIds,title,abstract,year"},
    )
    if isinstance(data, dict):
        for hit in data.get("data") or []:
            ext = hit.get("externalIds") or {}
            if _external_ids_to_arxiv(ext) == aid or aid in str(ext):
                return hit
            if hit.get("paperId"):
                return hit
    return None


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


def _enrich_ref_abstract(rec: dict[str, Any]) -> dict[str, Any]:
    """Fill missing abstract via S2 paper id or OpenAlex arXiv/DOI."""
    if rec.get("has_abstract") and (rec.get("abstract") or "").strip():
        return rec
    # Semantic Scholar by paper id
    s2_id = rec.get("s2_id")
    if s2_id:
        data = _get(
            f"{S2_BASE}/paper/{s2_id}",
            headers=_s2_headers(),
            params={"fields": "abstract,title,year,externalIds"},
        )
        if isinstance(data, dict):
            ab = (data.get("abstract") or "").strip()
            if ab:
                rec = dict(rec)
                rec["abstract"] = ab
                rec["has_abstract"] = True
                return rec
            # merge external ids for OA fallback
            ext = data.get("externalIds") or {}
            if not rec.get("arxiv_id"):
                rec = dict(rec)
                rec["arxiv_id"] = _external_ids_to_arxiv(ext)
            if not rec.get("doi"):
                rec = dict(rec)
                rec["doi"] = ext.get("DOI") or ext.get("doi")
    # OpenAlex by arXiv
    arxiv = rec.get("arxiv_id")
    if arxiv:
        oa = fetch_openalex_by_arxiv(str(arxiv))
        if oa:
            filled = openalex_work_to_record(oa)
            if filled and filled.get("has_abstract"):
                rec = dict(rec)
                rec["abstract"] = filled["abstract"]
                rec["has_abstract"] = True
                if not rec.get("title"):
                    rec["title"] = filled.get("title")
                return rec
    # OpenAlex by DOI
    doi = rec.get("doi")
    if doi:
        data = _get(
            f"{OPENALEX_BASE}/works/https://doi.org/{doi}",
            params={"mailto": os.getenv("OPENALEX_MAILTO", "novelty@localhost")},
        )
        if isinstance(data, dict):
            filled = openalex_work_to_record(data)
            if filled and filled.get("has_abstract"):
                rec = dict(rec)
                rec["abstract"] = filled["abstract"]
                rec["has_abstract"] = True
                return rec
    return rec


def enrich_refs_abstracts(
    refs: list[dict[str, Any]],
    *,
    max_enrich: int = 40,
    min_frac: float = 0.4,
) -> tuple[list[dict[str, Any]], int]:
    """Ensure enough refs have abstracts for embedding-based metrics."""
    if not refs:
        return refs, 0
    with_abs = sum(1 for r in refs if r.get("has_abstract"))
    if with_abs / len(refs) >= min_frac and with_abs >= 5:
        return refs, 0
    enriched = 0
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(refs):
        if i >= max_enrich and (sum(1 for r in out if r.get("has_abstract")) / max(1, len(out))) >= min_frac:
            out.extend(refs[i:])
            break
        if not rec.get("has_abstract"):
            rec = _enrich_ref_abstract(rec)
            if rec.get("has_abstract"):
                enriched += 1
            time.sleep(0.08)
        out.append(rec)
    return out, enriched


def resolve_focal_and_refs(
    arxiv_id: str,
    *,
    max_refs: int = 80,
    enrich_abstracts: bool = True,
) -> dict[str, Any]:
    """
    Resolve focal paper metadata + reference records.

    Returns dict with keys: focal, refs, source, stats

    enrich_abstracts:
      True  — novelty scoring (fills missing abstracts; slow, many API calls)
      False — citation-graph / edge collection only (fast; titles+ids enough)
    """
    aid = strip_version(arxiv_id)
    focal: dict[str, Any] | None = None
    refs: list[dict[str, Any]] = []
    source = ""

    # Prefer one combined S2 call (focal + refs) to cut rate-limit pressure
    s2_raw, raw_refs, src = fetch_s2_paper_with_refs(aid, limit=max_refs)
    if s2_raw:
        focal = s2_node_to_record(s2_raw)
        source = src
        for node in raw_refs:
            rec = s2_node_to_record(node)
            if rec:
                refs.append(rec)
        if enrich_abstracts and focal and not focal.get("has_abstract"):
            focal = _enrich_ref_abstract(focal)
    else:
        s2 = fetch_s2_paper(aid) or fetch_s2_paper_flexible(aid)
        if s2:
            focal = s2_node_to_record(s2)
            if enrich_abstracts and focal and not focal.get("has_abstract"):
                focal = _enrich_ref_abstract(focal)
            raw_refs2, src2 = fetch_s2_references(
                aid, limit=max_refs, s2_paper_id=s2.get("paperId")
            )
            source = src2
            for node in raw_refs2:
                rec = s2_node_to_record(node)
                if rec:
                    refs.append(rec)

    if focal is None or not refs:
        oa = fetch_openalex_by_arxiv(aid)
        if oa and focal is None:
            focal = openalex_work_to_record(oa)
        if not refs:
            # OpenAlex referenced_works is one HTTP per ref — only when needed
            if enrich_abstracts:
                oa_refs, oa_src = fetch_openalex_references(aid, limit=max_refs)
            else:
                # Lightweight: OpenAlex work record often includes referenced_works ids only
                oa_refs, oa_src = fetch_openalex_references_light(aid, limit=max_refs)
            if oa_refs:
                refs = oa_refs
                source = oa_src if not source else source + "+" + oa_src

    # Nested S2 refs often lack abstracts — enrich only for novelty path
    if enrich_abstracts:
        with_abs0 = sum(1 for r in refs if r.get("has_abstract"))
        if refs and (with_abs0 / len(refs) < 0.25 or with_abs0 < 3):
            refs, n_en = enrich_refs_abstracts(refs, max_enrich=min(50, max_refs))
            if n_en:
                source = (source or "s2") + f"+enrich({n_en})"
            with_abs1 = sum(1 for r in refs if r.get("has_abstract"))
            if with_abs1 < 3:
                oa_refs, oa_src = fetch_openalex_references(aid, limit=max_refs)
                if oa_refs and sum(1 for r in oa_refs if r.get("has_abstract")) > with_abs1:
                    refs = oa_refs
                    source = (source or "") + "+" + oa_src

    if focal is None:
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
    elif enrich_abstracts and not focal.get("has_abstract"):
        # last try: OpenAlex focal
        oa = fetch_openalex_by_arxiv(aid)
        if oa:
            filled = openalex_work_to_record(oa)
            if filled and filled.get("has_abstract"):
                focal = {**focal, "abstract": filled["abstract"], "has_abstract": True,
                         "title": focal.get("title") or filled.get("title")}

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


def fetch_openalex_references_light(
    arxiv_id: str,
    *,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], str]:
    """
    Citation-graph path: one OpenAlex work fetch; build minimal ref records from
    referenced_works URLs without a follow-up HTTP per cited work.
    """
    oa = fetch_openalex_by_arxiv(arxiv_id)
    if not isinstance(oa, dict):
        return [], "openalex:no_work"
    ref_urls = (oa.get("referenced_works") or [])[:limit]
    out: list[dict] = []
    for url in ref_urls:
        wid = str(url).rstrip("/").split("/")[-1]
        if not wid:
            continue
        out.append(
            {
                "canonical_id": f"openalex:{wid}",
                "arxiv_id": None,
                "s2_id": None,
                "doi": None,
                "title": None,
                "year": None,
                "abstract": None,
                "has_abstract": False,
                "citation_count": None,
                "source": "openalex:ref_id_only",
                "openalex_id": wid,
            }
        )
    return out, "openalex:referenced_works_ids_only"
