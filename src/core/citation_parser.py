"""Universal bibliographic citation parser for research literature.

Handles IEEE, APA, ACM, Nature, BibTeX, raw DOI URLs, and bare titles.
"""

from __future__ import annotations

import re
from typing import Any

from .ids import normalise_arxiv_id, normalise_doi


class ParsedCitation(dict):
    """Bibliographic citation metadata supporting attribute and dict access."""

    @property
    def doi(self) -> str | None:
        return self.get("doi")

    @property
    def arxiv_id(self) -> str | None:
        return self.get("arxiv_id")

    @property
    def title(self) -> str:
        return self.get("title") or ""

    @property
    def authors(self) -> list[str]:
        return self.get("authors") or []

    @property
    def raw_authors(self) -> str:
        return ", ".join(self.authors)

    @property
    def year(self) -> int | None:
        return self.get("year")

    @property
    def raw_text(self) -> str:
        return self.get("raw_text") or ""

    def to_synthetic_anchor(self) -> str:
        """Construct synthetic embedding text from title, authors, and year."""
        parts = [self.title]
        if self.authors:
            parts.append(f"Authors: {', '.join(self.authors)}")
        if self.year:
            parts.append(f"Year: {self.year}")
        return " - ".join(parts)


def parse_citation(raw_citation: str) -> ParsedCitation:
    """Parse unstructured citation string into structured bibliographic metadata."""
    text = re.sub(r"<[^>]+>", " ", str(raw_citation or "")).strip()
    text = " ".join(text.split())

    # Strip leading numerical index markers like [1], 1., (1)
    clean = re.sub(r"^\s*\[\s*\d+\s*\]\s*", "", text)
    clean = re.sub(r"^\s*\d+\.\s*", "", clean)
    clean = re.sub(r"^\s*\(\s*\d+\s*\)\s*", "", clean)

    doi = normalise_doi(clean)
    aid = normalise_arxiv_id(clean)

    # 1. Extract 4-digit publication year
    year = None
    y_match = re.search(r"\b(19\d\d|20\d\d)\b", clean)
    if y_match:
        try:
            year = int(y_match.group(1))
        except ValueError:
            year = None

    # 2. Extract title
    title = None

    # Style A: Quoted title (e.g. IEEE / ACM / Chicago styles: "Title of Paper")
    quote_match = re.search(r"[\"“\']([^\"“”\']{8,300})[\"”\']", clean)
    if quote_match:
        title = quote_match.group(1).strip().rstrip(".,")

    # Style B: BibTeX title field (title={...} or title="...")
    if not title:
        bibtex_match = re.search(r"title\s*=\s*[\"{]([^\"}]+)[\"}]", clean, re.I)
        if bibtex_match:
            title = bibtex_match.group(1).strip()

    # Style C: APA / Harvard format: Author, A. (Year). Title of article. In Periodical...
    if not title and year:
        apa_pattern = re.compile(rf"\({year}\)\.\s*([^.]+)\.", re.I)
        apa_match = apa_pattern.search(clean)
        if apa_match:
            title = apa_match.group(1).strip()

    # Style D: ACM style: Author. Year. Title. In Conference...
    if not title and year:
        acm_pattern = re.compile(rf"\b{year}\.\s*([^.]+)\.", re.I)
        acm_match = acm_pattern.search(clean)
        if acm_match:
            title = acm_match.group(1).strip()

    # Style E: Heuristic sentence split or bare title
    if not title:
        if not doi and not aid and len(clean.split()) <= 16:
            # Short string without identifiers is likely a bare title
            title = clean
        else:
            parts = [s.strip() for s in clean.split(".") if len(s.strip()) > 15]
            if len(parts) >= 2:
                # Typically the second element is the title after author list
                title = parts[1]
            elif parts:
                title = parts[0]

    if title:
        title = re.sub(r"^(?:in\s+|proceedings\s+of\s+)", "", title, flags=re.I)
        title = title.strip().rstrip(".,")
        if len(title) > 300:
            title = title[:300].rsplit(" ", 1)[0]

    # 3. Extract authors heuristic
    authors = []
    bib_author_match = re.search(r"author\s*=\s*[\"{]([^\"}]+)[\"}]", clean, re.I)
    if bib_author_match:
        raw_authors = bib_author_match.group(1).split(" and ")
        authors = [a.strip() for a in raw_authors if a.strip()]
    elif year:
        # Text preceding the year or title
        author_cand = clean.split(f"({year})")[0] if f"({year})" in clean else clean.split(str(year))[0]
        author_cand = author_cand.strip().rstrip(".,")
        if 4 < len(author_cand) < 120 and not any(k in author_cand.lower() for k in ["http", "doi", "arxiv"]):
            # Split by commas or 'and'
            raw_authors = re.split(r",\s*|\s+and\s+|&\s*", author_cand)
            authors = [a.strip() for a in raw_authors if len(a.strip()) > 2][:8]

    return ParsedCitation(
        {
            "doi": doi,
            "arxiv_id": aid,
            "title": title or clean[:240],
            "authors": authors,
            "year": year,
            "raw_text": text,
        }
    )
