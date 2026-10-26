#!/usr/bin/env python3
"""
Unit tests for arXiv Experimental HTML Ingest Engine.
Tests section extraction, equation parsing, and section-bounded chunking.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.rag.html_ingest import ArxivLatexHTMLParser, html_to_markdown_and_chunks


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Synthetic Paper: Deep Learning Geometry</title></head>
<body>
<nav class="html-header-nav"><a href="/">Home</a><input type="search"></nav>
<header class="arxiv-html-header"><h1>Banner</h1></header>

<div class="ltx_document">
  <h1 class="ltx_title ltx_title_document">Synthetic Paper: Deep Learning Geometry</h1>
  <div class="ltx_authors">Author One, Author Two</div>
  
  <section class="ltx_section" id="S1">
    <h2 class="ltx_title ltx_title_section"><span class="ltx_tag">1 </span>Introduction</h2>
    <div class="ltx_para">
      <p class="ltx_p">
        Neural networks operate in high-dimensional function spaces.
        We analyze the limiting dynamics under gradient flow with kernel <math alttext="\\Theta_\\infty"><mi>Θ</mi></math>.
      </p>
    </div>
  </section>

  <section class="ltx_section" id="S2">
    <h2 class="ltx_title ltx_title_section"><span class="ltx_tag">2 </span>Theoretical Framework</h2>
    <div class="ltx_para">
      <p class="ltx_p">
        Consider an L-layer network function <math alttext="f_\\theta(x)"><mi>f</mi></math>.
        The cost functional derivative satisfies:
      </p>
      <div class="ltx_equation ltx_displaymath">
        <math alttext="\\frac{d f_\\theta}{d t} = -\\Theta_t \\nabla \\mathcal{L}"><mi>eq</mi></math>
      </div>
      <h6 class="ltx_title">Theorem 1.</h6>
      <p class="ltx_p">In the infinite-width limit, the kernel remains invariant.</p>
    </div>
  </section>
</div>

<footer class="arxiv-html-footer">Footer text</footer>
</body>
</html>
"""


class TestHtmlIngest(unittest.TestCase):
    def test_parser_strips_nav_and_extracts_sections(self):
        parser = ArxivLatexHTMLParser()
        parser.feed(SAMPLE_HTML)
        parser.close()

        self.assertEqual(parser.extracted_title, "Synthetic Paper: Deep Learning Geometry")
        self.assertEqual(len(parser.sections), 3)

        front = parser.sections[0]
        self.assertEqual(front["section_id"], "front_matter")
        self.assertIn("Author One, Author Two", front["text"])

        sec1 = parser.sections[1]
        self.assertEqual(sec1["section_id"], "S1")
        self.assertIn("1 Introduction", sec1["heading"])
        self.assertIn("$\\Theta_\\infty$", sec1["text"])
        self.assertNotIn("Banner", sec1["text"])
        self.assertNotIn("Footer text", sec1["text"])

        sec2 = parser.sections[2]
        self.assertEqual(sec2["section_id"], "S2")
        self.assertIn("2 Theoretical Framework", sec2["heading"])
        self.assertIn("$f_\\theta(x)$", sec2["text"])
        self.assertIn("$$\n\\frac{d f_\\theta}{d t} = -\\Theta_t \\nabla \\mathcal{L}\n$$", sec2["text"])
        self.assertIn("**Theorem 1.**", sec2["text"])

    def test_markdown_and_chunk_generation(self):
        md, title, chunks = html_to_markdown_and_chunks(
            SAMPLE_HTML,
            arxiv_id="2501.99999",
            title=None,
        )

        self.assertEqual(title, "Synthetic Paper: Deep Learning Geometry")
        self.assertTrue(md.startswith("# Synthetic Paper: Deep Learning Geometry"))
        self.assertGreaterEqual(len(chunks), 3)

        # Chunks must maintain explicit section labels
        self.assertEqual(chunks[0].paper_id, "2501.99999")
        self.assertEqual(chunks[0].source_md, "html:front_matter")

        self.assertEqual(chunks[1].source_md, "html:S1")
        self.assertIn("1 Introduction", chunks[1].section)

        self.assertEqual(chunks[2].source_md, "html:S2")
        self.assertIn("2 Theoretical Framework", chunks[2].section)

    def test_empty_or_malformed_html(self):
        md, title, chunks = html_to_markdown_and_chunks("<html><body></body></html>", "9999.99999")
        self.assertEqual(title, "Untitled")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
