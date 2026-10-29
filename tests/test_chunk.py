"""Unit tests for semantic boundary chunking in src.rag.chunk."""

import unittest
from src.rag.chunk import _char_chunks, _find_protected_spans


class TestSemanticChunking(unittest.TestCase):
    def test_find_protected_spans(self):
        text = "Intro\n\n$$\nx^2 + y^2 = z^2\n$$\n\nCode:\n```python\nprint(1)\n```\nDone."
        spans = _find_protected_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertIn("x^2", text[spans[0][0]:spans[0][1]])
        self.assertIn("print(1)", text[spans[1][0]:spans[1][1]])

    def test_equation_integrity_under_tight_splits(self):
        text = (
            "Paragraph one is relatively long to fill up initial character allowance and ensure each chunk exceeds eighty characters minimum threshold without issues.\n\n"
            "$$\n"
            "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}\n"
            "$$\n\n"
            "Paragraph two immediately follows the equation with additional explanations and context that also comfortably exceeds eighty characters minimum."
        )
        chunks = _char_chunks(text, size=180, overlap=30)
        self.assertGreater(len(chunks), 1)
        for start, end, chunk in chunks:
            eq_count = chunk.count("$$")
            self.assertEqual(
                eq_count % 2,
                0,
                f"Chunk broke equation delimiter: {repr(chunk)}",
            )

    def test_code_block_integrity(self):
        text = (
            "Text before code block contains a substantial introductory paragraph explaining the algorithm and its motivation in depth and detail.\n\n"
            "```python\n"
            "def compute_total(values):\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v\n"
            "    return total\n"
            "```\n\n"
            "Text after code block continues the analysis with extensive evaluation remarks and discussion of complexity bounds across various input spaces."
        )
        chunks = _char_chunks(text, size=200, overlap=30)
        self.assertGreater(len(chunks), 1)
        for start, end, chunk in chunks:
            code_count = chunk.count("```")
            self.assertEqual(
                code_count % 2,
                0,
                f"Chunk broke code fence: {repr(chunk)}",
            )


if __name__ == "__main__":
    unittest.main()
