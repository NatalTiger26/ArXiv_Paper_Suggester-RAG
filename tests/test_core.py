#!/usr/bin/env python3
"""
Unit test suite for ArXiv Paper Suggester core logic.
Run with:
  python3 -m unittest discover -s tests
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

from src.rag.download import normalise_arxiv_id
from src.final_eval import looks_like_refusal, looks_like_refusal_broader, citation_validity
from src.ranking import _validate_weights, combine_scores
from src.rag.hybrid import extract_query_identifiers, BM25Scorer


class TestIdNormalization(unittest.TestCase):
    def test_clean_id(self):
        self.assertEqual(normalise_arxiv_id("1806.07572"), "1806.07572")

    def test_version_stripped(self):
        self.assertEqual(normalise_arxiv_id("1806.07572v3"), "1806.07572")
        self.assertEqual(normalise_arxiv_id("2501.16496v1"), "2501.16496")

    def test_abs_url(self):
        self.assertEqual(
            normalise_arxiv_id("https://arxiv.org/abs/2404.14082"),
            "2404.14082",
        )
        self.assertEqual(
            normalise_arxiv_id("http://arxiv.org/abs/2304.14997v2"),
            "2304.14997",
        )

    def test_pdf_url(self):
        self.assertEqual(
            normalise_arxiv_id("https://arxiv.org/pdf/1806.07572.pdf"),
            "1806.07572",
        )

    def test_invalid_input(self):
        self.assertIsNone(normalise_arxiv_id(""))
        self.assertIsNone(normalise_arxiv_id(None))
        self.assertIsNone(normalise_arxiv_id("not-an-arxiv-id"))


class TestRefusalGraders(unittest.TestCase):
    def test_official_refusal_matches(self):
        text = "There is not enough information to answer this question from the ingested papers."
        self.assertTrue(looks_like_refusal(text))
        self.assertTrue(looks_like_refusal_broader(text))

    def test_positive_grounded_answer(self):
        text = "The Neural Tangent Kernel (NTK) characterizes the evolution of deep networks [1]."
        self.assertFalse(looks_like_refusal(text))
        self.assertFalse(looks_like_refusal_broader(text))

    def test_broader_refusal_distinction(self):
        # Conversational phrasing that official whitelist misses, but broader catches
        text = "There is no mention in the provided excerpts of quantum teleportation."
        self.assertFalse(looks_like_refusal(text))
        self.assertTrue(looks_like_refusal_broader(text))

    def test_case_insensitivity(self):
        self.assertTrue(looks_like_refusal("CANNOT ANSWER THIS QUESTION"))
        self.assertTrue(looks_like_refusal_broader("INSUFFICIENT INFORMATION IN THE CONTEXT"))


class TestCitationValidity(unittest.TestCase):
    def test_valid_citations(self):
        answer = "Circuit discovery isolates subgraphs [1] to explain network behavior [2]."
        res = citation_validity(answer, n_sources=3)
        self.assertTrue(res["has_citation"])
        self.assertTrue(res["valid"])
        self.assertEqual(res["invalid_indices"], [])
        self.assertEqual(res["n_citations"], 2)

    def test_invalid_out_of_bounds(self):
        answer = "Recent work [4] showed NTK convergence, while [1] proposed the architecture."
        res = citation_validity(answer, n_sources=2)
        self.assertTrue(res["has_citation"])
        self.assertFalse(res["valid"])
        self.assertEqual(res["invalid_indices"], [4])

    def test_no_citation_in_answer(self):
        answer = "Mechanistic interpretability reverse engineers neural representations."
        res = citation_validity(answer, n_sources=2)
        self.assertFalse(res["has_citation"])
        self.assertFalse(res["valid"])

    def test_refusal_without_citations_is_valid(self):
        answer = "There is not enough information in the provided context to answer."
        res = citation_validity(answer, n_sources=3)
        self.assertFalse(res["has_citation"])
        self.assertTrue(res["valid"])


class TestRankingWeights(unittest.TestCase):
    def test_valid_weights(self):
        w = {"embedding": 0.5, "recency": 0.2, "citation": 0.2, "author": 0.1}
        cleaned = _validate_weights(w)
        self.assertAlmostEqual(sum(cleaned.values()), 1.0)

    def test_negative_weight_raises(self):
        w = {"embedding": 0.8, "recency": -0.1, "citation": 0.2, "author": 0.1}
        with self.assertRaises(ValueError):
            _validate_weights(w)

    def test_zero_weights_raises(self):
        w = {"embedding": 0.0, "recency": 0.0, "citation": 0.0, "author": 0.0}
        with self.assertRaises(ValueError):
            _validate_weights(w)

    def test_combine_scores(self):
        signals = {"embedding": 0.9, "recency": 0.5, "citation": 0.8, "author": 0.2}
        weights = {"embedding": 0.5, "recency": 0.2, "citation": 0.2, "author": 0.1}
        score, contribs = combine_scores(signals, weights)
        expected = 0.9 * 0.5 + 0.5 * 0.2 + 0.8 * 0.2 + 0.2 * 0.1
        self.assertAlmostEqual(score, expected)
        self.assertEqual(len(contribs), 4)


class TestHybridQueryExtraction(unittest.TestCase):
    def test_extract_single_id(self):
        q = "What is the main contribution of arXiv paper 1806.07572?"
        ids = extract_query_identifiers(q)
        self.assertEqual(ids, ["1806.07572"])

    def test_extract_multiple_versioned_ids(self):
        q = "Compare findings in 2501.16496v2 against 2304.14997v1."
        ids = extract_query_identifiers(q)
        self.assertEqual(ids, ["2501.16496", "2304.14997"])

    def test_no_id_present(self):
        q = "What is circuit discovery in mechanistic interpretability?"
        ids = extract_query_identifiers(q)
        self.assertEqual(ids, [])

    def test_bm25_scoring(self):
        corpus = [
            {"paper_id": "1", "title": "Neural Tangent Kernel", "text": "infinite width limits"},
            {"paper_id": "2", "title": "Circuit Discovery", "text": "subgraphs and induction heads"},
        ]
        bm25 = BM25Scorer(corpus)
        scores_ntk = bm25.score("Neural Tangent")
        self.assertGreater(scores_ntk[0], scores_ntk[1])

        scores_circ = bm25.score("induction heads")
        self.assertGreater(scores_circ[1], scores_circ[0])


if __name__ == "__main__":
    unittest.main()
