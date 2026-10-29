"""Unit tests for src.core.citation_parser multi-style reference parser."""

import unittest
from src.core.citation_parser import parse_citation


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


if __name__ == "__main__":
    unittest.main()
