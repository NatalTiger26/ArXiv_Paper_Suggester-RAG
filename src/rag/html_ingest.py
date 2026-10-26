"""
arXiv Experimental HTML Parser and Ingestion Engine.
Downloads and converts arXiv's native HTML (LaTeXML / ar5iv) directly into
clean, section-aware Markdown and structured chunks.

Advantages over PDF parsing:
  • 10x-30x smaller transfer size (~400 KB HTML vs. 5-20 MB PDF).
  • Zero OCR or PDF text extraction distortion.
  • Exact mathematical equations extracted directly from LaTeX alttext attributes.
  • Explicit section boundaries (<section id="S1">, <h2>, <h3>) preserved.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .chunk import Chunk, is_bad_title
from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS

ARXIV_HTML_URL = "https://arxiv.org/html/{id}"
AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{id}"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (ArXivPaperSuggester/1.1; mailto:ajay.sharma.pg26@plaksha.edu.in)"
)


def fetch_arxiv_html(arxiv_id: str, timeout: int = 15) -> str | None:
    """
    Fetch native HTML for an arXiv ID.
    Returns decoded HTML text if HTTP 200, or None if HTML is unavailable (404/error).
    """
    aid = arxiv_id.strip().replace("arXiv:", "").replace("arxiv:", "")
    endpoints = [ARXIV_HTML_URL.format(id=aid), AR5IV_HTML_URL.format(id=aid)]

    for url in endpoints:
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    ctype = resp.headers.get("Content-Type", "")
                    if "text/html" in ctype:
                        content = resp.read().decode("utf-8", errors="ignore")
                        if "ltx_document" in content or "<section" in content or "ltx_title" in content:
                            return content
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            continue

    return None


class ArxivLatexHTMLParser(HTMLParser):
    """
    Robust stack-based parser for LaTeXML-generated arXiv HTML.
    Extracts sections, headings, math formulas, and paragraphs cleanly.
    """

    def __init__(self):
        super().__init__()
        self.skip_stack: list[str] = []
        self.sections: list[dict[str, Any]] = []
        self.current_sec_id: str = "front_matter"
        self.current_heading: str = "Abstract"
        self.current_body_lines: list[str] = []

        self.in_heading: bool = False
        self.heading_tag: str = ""
        self.heading_buf: list[str] = []
        self.in_math: bool = False
        self.in_display_math: bool = False
        self.extracted_title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        classes = attr_dict.get("class", "").split()

        # Tags whose subtree should be skipped entirely
        if tag in ("script", "style", "nav", "header", "footer", "aside") or any(
            c in ("ltx_page_header", "ltx_page_footer", "ltx_nav") for c in classes
        ):
            self.skip_stack.append(tag)
            return

        if self.skip_stack:
            return

        # Track display math containers
        if tag in ("div", "table") and any("displaymath" in c.lower() for c in classes):
            self.in_display_math = True

        # Section boundaries
        if tag == "section":
            sec_id = attr_dict.get("id", "")
            if sec_id:
                self._flush_section()
                self.current_sec_id = sec_id

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_heading = True
            self.heading_tag = tag
            self.heading_buf = []

        # Mathematical formulas
        if tag == "math":
            self.in_math = True
            alttext = attr_dict.get("alttext", "").strip()
            if alttext:
                is_display = (
                    self.in_display_math
                    or "ltx_displaymath" in classes
                    or "display" in attr_dict.get("display", "")
                )
                if is_display:
                    self.current_body_lines.append(f"\n\n$$\n{alttext}\n$$\n\n")
                else:
                    self.current_body_lines.append(f" ${alttext}$ ")

        # Block spacing
        if tag in ("p", "div", "li", "tr") and not self.in_math:
            self.current_body_lines.append("\n")

        # List items
        if tag == "li":
            self.current_body_lines.append("• ")

    def handle_endtag(self, tag: str):
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
            return

        if self.skip_stack:
            return

        if tag in ("div", "table") and self.in_display_math:
            self.in_display_math = False

        if tag == "math":
            self.in_math = False

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.in_heading:
            self.in_heading = False
            h_text = " ".join("".join(self.heading_buf).split()).strip()
            if h_text and len(h_text) > 2 and not is_bad_title(h_text):
                if not self.extracted_title and (tag == "h1" or "ltx_title_document" in self.heading_tag):
                    self.extracted_title = h_text
                
                # Only h1, h2, h3, h4 update the overarching section heading
                # h5, h6 (e.g. Proposition 1, Remark 2) are formatted in-line
                if tag in ("h1", "h2", "h3", "h4"):
                    self.current_heading = h_text
                    level = min(int(tag[1]), 4)
                    hashes = "#" * level
                    self.current_body_lines.append(f"\n\n{hashes} {h_text}\n\n")
                else:
                    self.current_body_lines.append(f"\n\n**{h_text}**\n\n")

    def handle_data(self, data: str):
        if self.skip_stack or self.in_math:
            return

        if self.in_heading:
            self.heading_buf.append(data)
        else:
            self.current_body_lines.append(data)

    def _flush_section(self):
        raw_text = "".join(self.current_body_lines).strip()
        if raw_text and len(raw_text) >= MIN_CHUNK_CHARS:
            cleaned = re.sub(r"\n{3,}", "\n\n", raw_text)
            self.sections.append({
                "section_id": self.current_sec_id,
                "heading": self.current_heading,
                "text": cleaned,
            })
        self.current_body_lines = []

    def close(self):
        super().close()
        self._flush_section()


def html_to_markdown_and_chunks(
    html: str,
    arxiv_id: str,
    title: str | None = None,
    source_name: str = "html",
) -> tuple[str, str, list[Chunk]]:
    """
    Parse arXiv LaTeXML HTML into clean Markdown and section-bounded Chunks.
    """
    parser = ArxivLatexHTMLParser()
    parser.feed(html)
    parser.close()

    resolved_title = title or parser.extracted_title or "Untitled"
    if is_bad_title(resolved_title):
        resolved_title = "Untitled"

    # Assemble Markdown text
    md_lines = [f"# {resolved_title}", f"arXiv: {arxiv_id}", ""]
    for s in parser.sections:
        md_lines.append(s["text"])
        md_lines.append("")
    markdown_content = "\n".join(md_lines)

    # Generate section-bounded chunks
    chunks: list[Chunk] = []
    chunk_index = 0

    for s in parser.sections:
        sec_text = s["text"]
        sec_heading = s["heading"]
        sec_id = s["section_id"]

        # If the entire section fits comfortably inside CHUNK_SIZE, keep it whole
        if len(sec_text) <= CHUNK_SIZE:
            if len(sec_text) >= MIN_CHUNK_CHARS:
                c = Chunk(
                    chunk_id=f"{arxiv_id}_{chunk_index:04d}",
                    paper_id=arxiv_id,
                    title=resolved_title,
                    text=sec_text,
                    section=sec_heading,
                    char_start=0,
                    char_end=len(sec_text),
                    source_md=f"{source_name}:{sec_id}",
                )
                chunks.append(c)
                chunk_index += 1
            continue

        # Otherwise sliding-window chunk *within* this section boundary only
        start = 0
        while start < len(sec_text):
            end = min(start + CHUNK_SIZE, len(sec_text))
            if end < len(sec_text):
                for sep in ("\n\n", "\n", ". ", "; ", " "):
                    pos = sec_text.rfind(sep, start + CHUNK_SIZE // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            sub_text = sec_text[start:end].strip()
            if len(sub_text) >= MIN_CHUNK_CHARS:
                c = Chunk(
                    chunk_id=f"{arxiv_id}_{chunk_index:04d}",
                    paper_id=arxiv_id,
                    title=resolved_title,
                    text=sub_text,
                    section=sec_heading,
                    char_start=start,
                    char_end=end,
                    source_md=f"{source_name}:{sec_id}",
                )
                chunks.append(c)
                chunk_index += 1

            if end >= len(sec_text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)

    return markdown_content, resolved_title, chunks
