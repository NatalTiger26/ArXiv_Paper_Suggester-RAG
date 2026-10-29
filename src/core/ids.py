"""Unified paper identifier normalization and canonical ID schemes."""

from __future__ import annotations

import re
import urllib.parse

ARXIV_RE = re.compile(
    r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
OLD_ARXIV_RE = re.compile(
    r"(?:arXiv:)?([a-z\-]+/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(
    r"(?:doi(?::|\.org/)|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
    re.IGNORECASE,
)


def strip_version(arxiv_id: str | None) -> str:
    """Strip version suffix (e.g. v1, v2) and 'arxiv:' prefix from identifier."""
    s = (arxiv_id or "").strip()
    s = re.sub(r"^arxiv:", "", s, flags=re.I)
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def normalise_arxiv_id(raw: str | None) -> str | None:
    """Extract a canonical arXiv identifier from free text, IDs, or URLs."""
    if not raw:
        return None
    s = str(raw).strip()
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I).strip("/")
    m = ARXIV_RE.search(s)
    if m:
        return m.group(1)
    m = OLD_ARXIV_RE.search(s)
    if m:
        return m.group(1).lower()
    s2 = strip_version(s)
    if re.fullmatch(r"\d{4}\.\d{4,5}", s2):
        return s2
    if re.fullmatch(r"[a-z\-]+/\d{7}", s2, re.I):
        return s2.lower()
    return None


def normalise_doi(raw: str | None) -> str | None:
    """Extract a clean, unquoted DOI from URL, text, or DOI string."""
    if not raw:
        return None
    s = str(raw).strip()
    m = DOI_RE.search(s)
    if m:
        cleaned = urllib.parse.unquote(m.group(1)).rstrip(".,;)]}>")
        return cleaned.lower()
    return None


def canonical_id(
    raw: str | None = None,
    *,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    doi: str | None = None,
) -> str | None:
    """Resolve canonical identifier in order: arxiv:<id>, doi:<doi>, s2:<id>, openalex:<id>."""
    if raw:
        raw_s = str(raw).strip()
        if raw_s.startswith(("arxiv:", "doi:", "s2:", "openalex:")):
            return raw_s
        aid = normalise_arxiv_id(raw_s)
        if aid:
            return f"arxiv:{aid}"
        d = normalise_doi(raw_s)
        if d:
            return f"doi:{d}"
        return raw_s

    aid = normalise_arxiv_id(arxiv_id)
    if aid:
        return f"arxiv:{aid}"
    d = normalise_doi(doi)
    if d:
        return f"doi:{d}"
    if s2_id:
        return f"s2:{str(s2_id).strip()}"
    if openalex_id:
        oid = str(openalex_id).strip().rstrip("/").split("/")[-1]
        return f"openalex:{oid}"
    return None


def parse_canonical(cid: str) -> tuple[str, str]:
    """Split a canonical identifier into (scheme, raw_id)."""
    if ":" in cid:
        scheme, rest = cid.split(":", 1)
        return scheme.lower(), rest
    aid = normalise_arxiv_id(cid)
    if aid:
        return "arxiv", aid
    d = normalise_doi(cid)
    if d:
        return "doi", d
    return "raw", cid
