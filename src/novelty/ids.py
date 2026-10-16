"""Canonical paper identifiers for the novelty cache."""

from __future__ import annotations

import re

_ARXIV_RE = re.compile(
    r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_OLD_ARXIV_RE = re.compile(
    r"(?:arXiv:)?([a-z\-]+/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)


def strip_version(arxiv_id: str) -> str:
    s = (arxiv_id or "").strip()
    s = s.replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def normalise_arxiv_id(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    m = _ARXIV_RE.search(s)
    if m:
        return m.group(1)
    m = _OLD_ARXIV_RE.search(s)
    if m:
        return m.group(1).lower()
    # bare id?
    s2 = strip_version(s)
    if re.fullmatch(r"\d{4}\.\d{4,5}", s2):
        return s2
    if re.fullmatch(r"[a-z\-]+/\d{7}", s2, re.I):
        return s2.lower()
    return None


def canonical_id(
    *,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    doi: str | None = None,
) -> str | None:
    """Prefer arxiv:<id>, else s2:<id>, else openalex:<id>, else doi:<doi>."""
    aid = normalise_arxiv_id(arxiv_id)
    if aid:
        return f"arxiv:{aid}"
    if s2_id:
        return f"s2:{str(s2_id).strip()}"
    if openalex_id:
        oid = str(openalex_id).strip()
        if oid.startswith("https://"):
            oid = oid.rstrip("/").split("/")[-1]
        return f"openalex:{oid}"
    if doi:
        return f"doi:{str(doi).strip().lower()}"
    return None


def parse_canonical(cid: str) -> tuple[str, str]:
    if ":" in cid:
        scheme, rest = cid.split(":", 1)
        return scheme.lower(), rest
    aid = normalise_arxiv_id(cid)
    if aid:
        return "arxiv", aid
    return "raw", cid
