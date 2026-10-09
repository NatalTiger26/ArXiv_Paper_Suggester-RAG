"""Structure-aware chunking of paper Markdown for RAG."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    section: str
    char_start: int
    char_end: int
    source_md: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": (self.title or "")[:300],
            "section": (self.section or "")[:200],
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_md": self.source_md,
        }


_BAD_TITLE_PATTERNS = [
    re.compile(r"^published as a conference", re.I),
    re.compile(r"^arXiv:", re.I),
    re.compile(r"^abstract$", re.I),
    re.compile(r"^untitled$", re.I),
]


def is_bad_title(title: str | None) -> bool:
    if not title or len(title.strip()) < 8:
        return True
    t = title.strip()
    return any(p.search(t) for p in _BAD_TITLE_PATTERNS)


def _extract_title(md: str) -> str:
    for line in md.splitlines()[:40]:
        line = line.strip()
        if line.startswith("# "):
            cand = line[2:].strip()
            if not is_bad_title(cand):
                return cand[:300]
        if line and not line.startswith("#") and len(line) > 15:
            if not is_bad_title(line):
                return line[:300]
    return "Untitled"


def _split_by_headings(md: str) -> list[tuple[str, str]]:
    """Return list of (section_heading, section_body)."""
    parts: list[tuple[str, str]] = []
    current_heading = "front_matter"
    current_lines: list[str] = []
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$")

    for line in md.splitlines():
        m = heading_re.match(line)
        if m:
            body = "\n".join(current_lines).strip()
            if body:
                parts.append((current_heading, body))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        parts.append((current_heading, body))
    return parts


def _char_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Sliding window over characters with boundary-aware cuts."""
    if len(text) <= size:
        return [(0, len(text), text)] if text.strip() else []

    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", "; ", " "):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            out.append((start, end, chunk))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_markdown(
    md_path: str | Path,
    paper_id: str,
    title: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Turn a Markdown file into overlapping chunks with section metadata."""
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    if title is None or is_bad_title(title):
        title = _extract_title(text)

    sections = _split_by_headings(text)
    chunks: list[Chunk] = []

    for section_name, body in sections:
        # Skip pure bibliography dumps that are huge and low-signal for Q&A
        sec_l = section_name.lower()
        if any(k in sec_l for k in ("references", "bibliography", "acknowledg")):
            # keep a short head only
            body = body[:1500]

        for start, end, slice_text in _char_chunks(body, chunk_size, chunk_overlap):
            raw = f"{paper_id}|{section_name}|{start}|{end}|{slice_text[:48]}"
            cid = hashlib.sha1(raw.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    paper_id=paper_id,
                    title=title,
                    text=slice_text,
                    section=section_name,
                    char_start=start,
                    char_end=end,
                    source_md=str(md_path.name),
                )
            )
    return chunks
