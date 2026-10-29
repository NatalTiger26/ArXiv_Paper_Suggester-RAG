"""Unit tests for src.core.ids identifier normalization utilities."""

import unittest
from src.core.ids import (
    canonical_id,
    normalise_arxiv_id,
    normalise_doi,
    parse_canonical,
    strip_version,
)


class TestCoreIDs(unittest.TestCase):
    def test_strip_version(self):
        self.assertEqual(strip_version("1706.03762v2"), "1706.03762")
        self.assertEqual(strip_version("hep-th/9711200v1"), "hep-th/9711200")
        self.assertEqual(strip_version("2206.04615"), "2206.04615")

    def test_normalise_arxiv_id(self):
        self.assertEqual(normalise_arxiv_id("1706.03762"), "1706.03762")
        self.assertEqual(normalise_arxiv_id("arXiv:1706.03762v5"), "1706.03762")
        self.assertEqual(normalise_arxiv_id("https://arxiv.org/abs/2206.04615v1"), "2206.04615")
        self.assertEqual(normalise_arxiv_id("https://arxiv.org/pdf/2206.04615.pdf"), "2206.04615")
        self.assertEqual(normalise_arxiv_id("hep-th/9711200"), "hep-th/9711200")
        self.assertIsNone(normalise_arxiv_id("not an arxiv id"))

    def test_normalise_doi(self):
        self.assertEqual(normalise_doi("10.1146/annurev-conmatphys-031119-050745"), "10.1146/annurev-conmatphys-031119-050745")
        self.assertEqual(normalise_doi("https://doi.org/10.1038/nature14539"), "10.1038/nature14539")
        self.assertEqual(normalise_doi("doi:10.1016/j.cell.2020.05.001."), "10.1016/j.cell.2020.05.001")
        self.assertIsNone(normalise_doi("invalid doi"))

    def test_canonical_id(self):
        self.assertEqual(canonical_id("1706.03762"), "arxiv:1706.03762")
        self.assertEqual(canonical_id("10.1038/nature14539"), "doi:10.1038/nature14539")
        self.assertEqual(canonical_id(arxiv_id="2206.04615"), "arxiv:2206.04615")
        self.assertEqual(canonical_id(doi="10.1146/annurev-conmatphys-031119-050745"), "doi:10.1146/annurev-conmatphys-031119-050745")
        self.assertEqual(canonical_id(s2_id="abc123def"), "s2:abc123def")
        self.assertEqual(canonical_id(openalex_id="https://openalex.org/W1234567"), "openalex:W1234567")

    def test_parse_canonical(self):
        self.assertEqual(parse_canonical("arxiv:1706.03762"), ("arxiv", "1706.03762"))
        self.assertEqual(parse_canonical("doi:10.1038/nature14539"), ("doi", "10.1038/nature14539"))
        self.assertEqual(parse_canonical("1706.03762"), ("arxiv", "1706.03762"))
        self.assertEqual(parse_canonical("10.1038/nature14539"), ("doi", "10.1038/nature14539"))


if __name__ == "__main__":
    unittest.main()
