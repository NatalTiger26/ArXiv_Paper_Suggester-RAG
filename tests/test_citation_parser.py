"""Unit tests for src.core.citation_parser multi-style reference parser."""

import unittest
from src.core.citation_parser import parse_citation
from src.novelty.refs import _merge_references, fetch_local_markdown_references


class TestCitationParser(unittest.TestCase):
    def test_parse_ieee(self):
        text = '[1] A. Vaswani et al., "Attention Is All You Need," in Advances in Neural Information Processing Systems, 2017.'
        parsed = parse_citation(text)
        self.assertEqual(parsed.title, "Attention Is All You Need")
        self.assertEqual(parsed.year, 2017)
        self.assertIn("Vaswani", parsed.raw_authors)

    def test_parse_apa(self):
        text = "Vaswani, A., Shazeer, N., Parmar, N. (2017). Attention is all you need. Advances in Neural Information Processing Systems."
        parsed = parse_citation(text)
        self.assertEqual(parsed.year, 2017)
        self.assertIn("Attention is all you need", parsed.title)
        self.assertIn("Vaswani", parsed.raw_authors)

    def test_parse_doi_in_text(self):
        text = "Detailed review on quantum materials. doi:10.1146/annurev-conmatphys-031119-050745"
        parsed = parse_citation(text)
        self.assertEqual(parsed.doi, "10.1146/annurev-conmatphys-031119-050745")

    def test_parse_arxiv_url(self):
        text = "Preprint available at https://arxiv.org/abs/2206.04615v1"
        parsed = parse_citation(text)
        self.assertEqual(parsed.arxiv_id, "2206.04615")

    def test_parse_bibtex(self):
        text = """@article{vaswani2017,
          title = {Attention Is All You Need},
          author = {Vaswani, Ashish and Shazeer, Noam},
          year = {2017},
          journal = {NeurIPS}
        }"""
        parsed = parse_citation(text)
        self.assertEqual(parsed.title, "Attention Is All You Need")
        self.assertEqual(parsed.year, 2017)
        self.assertIn("Vaswani", parsed.raw_authors)

    def test_synthetic_representation(self):
        text = '[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.'
        parsed = parse_citation(text)
        synth = parsed.to_synthetic_anchor()
        self.assertIn("BERT", synth)
        self.assertIn("2019", synth)

    def test_merge_references_deduplication(self):
        list1 = [
            {"canonical_id": "arxiv:1706.03762", "arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
            {"canonical_id": "arxiv:1810.04805", "arxiv_id": "1810.04805", "title": "BERT"},
        ]
        list2 = [
            {"canonical_id": "arxiv:1706.03762", "arxiv_id": "1706.03762", "title": "Attention Is All You Need Duplicate"},
            {"canonical_id": "arxiv:1512.03385", "arxiv_id": "1512.03385", "title": "Deep Residual Learning"},
        ]
        merged = _merge_references(list1, list2, limit=10)
        self.assertEqual(len(merged), 3)
        aids = [r["arxiv_id"] for r in merged]
        self.assertEqual(aids, ["1706.03762", "1810.04805", "1512.03385"])

    def test_fetch_local_markdown_references_existing(self):
        refs, src = fetch_local_markdown_references("0912.1128", limit=20)
        self.assertEqual(src, "local_markdown")
        self.assertTrue(len(refs) > 0)
        self.assertTrue(all(r.get("has_abstract") for r in refs))


if __name__ == "__main__":
    unittest.main()
