"""Tests for catalog browsing, batch novelty classification, and bulk topic ingestion."""

import unittest
from pathlib import Path

from scripts.batch_novelty_library import classify_geometry
from scripts.bulk_topic_ingest import parse_topic_list
from src.catalog import browse_catalog, get_catalog_stats


class TestCatalogBrowsing(unittest.TestCase):
    def test_catalog_stats(self):
        stats = get_catalog_stats()
        self.assertIn("total_papers", stats)
        self.assertGreater(stats["total_papers"], 0)
        self.assertIn("min_year", stats)
        self.assertIn("top_categories", stats)

    def test_browse_catalog_pagination(self):
        papers, total = browse_catalog(limit=5, offset=0, query="")
        self.assertEqual(len(papers), 5)
        self.assertGreater(total, 100)

    def test_browse_catalog_search(self):
        papers, total = browse_catalog(limit=5, offset=0, query="neural")
        self.assertGreaterEqual(total, 1)
        for p in papers:
            self.assertIn("arxiv_id", p)
            self.assertIn("title", p)

    def test_browse_catalog_sorting(self):
        papers, total = browse_catalog(limit=5, offset=0, query="", sort_by="citations_desc")
        self.assertEqual(len(papers), 5)
        cites = [p.get("citation_count") or 0 for p in papers]
        # Ensure descending
        for i in range(len(cites) - 1):
            self.assertGreaterEqual(cites[i], cites[i + 1])


class TestScriptsLogic(unittest.TestCase):
    def test_classify_geometry(self):
        # High synergy: delta >= 0.05
        c_syn = classify_geometry(m=0.4, delta=0.08, r2_plus=0.6)
        self.assertEqual(c_syn, "High Synergy Cone Blend")

        # Linear extension: low delta, high m
        c_lin = classify_geometry(m=0.8, delta=0.01, r2_plus=0.65)
        self.assertEqual(c_lin, "Linear / Incremental Extension")

        # Paradigm shift / outlier: low r2_plus
        c_shift = classify_geometry(m=0.2, delta=0.01, r2_plus=0.3)
        self.assertEqual(c_shift, "Paradigm Shift / Subspace Outlier")

    def test_parse_topic_list(self):
        topics = parse_topic_list(["mechanistic interpretability", "diffusion, ssm"], None)
        self.assertIn("mechanistic interpretability", topics)
        self.assertIn("diffusion", topics)
        self.assertIn("ssm", topics)

        # Default fallback
        default_topics = parse_topic_list(None, None)
        self.assertGreater(len(default_topics), 0)


if __name__ == "__main__":
    unittest.main()
