"""Canonical paper identifiers for the novelty cache. Re-exports from src.core.ids."""

from __future__ import annotations

from src.core.ids import (
    canonical_id,
    normalise_arxiv_id,
    normalise_doi,
    parse_canonical,
    strip_version,
)

__all__ = [
    "canonical_id",
    "normalise_arxiv_id",
    "normalise_doi",
    "parse_canonical",
    "strip_version",
]
