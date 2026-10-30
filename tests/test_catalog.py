import tempfile
import unittest
from pathlib import Path

from src.catalog import (
    clean_arxiv_id,
    connect,
    get_paper,
    resolve_entities_in_query,
    search_catalog,
    update_citation,
)


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.db_path = Path(self.temp_file.name)
        self.temp_file.close()
        self.conn = connect(self.db_path)

        # Seed sample papers
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO papers (arxiv_id, title, abstract, year, citation_count, categories, doi, updated_at)
                VALUES
                ('1806.07572', 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks', 'We study gradient descent on wide neural networks...', 2018, 4383, 'cs.LG,stat.ML', '10.48550/arXiv.1806.07572', 1.0),
                ('1610.02391', 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization', 'We propose Visual Explanations for CNNs...', 2016, 30531, 'cs.CV', '10.48550/arXiv.1610.02391', 1.0)
                """
            )

    def tearDown(self):
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_clean_arxiv_id(self):
        self.assertEqual(clean_arxiv_id("arxiv:1806.07572v2"), "1806.07572")
        self.assertEqual(clean_arxiv_id("1806.07572"), "1806.07572")

    def test_get_paper(self):
        p = get_paper("1806.07572", conn=self.conn)
        self.assertIsNotNone(p)
        self.assertEqual(p["year"], 2018)
        self.assertEqual(p["citation_count"], 4383)

    def test_search_catalog(self):
        results = search_catalog("Neural Tangent Kernel", conn=self.conn)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["arxiv_id"], "1806.07572")

    def test_resolve_entities(self):
        entities = resolve_entities_in_query("What are the advantages of Grad-CAM?", conn=self.conn)
        self.assertIn("1610.02391", entities)

    def test_update_citation(self):
        update_citation("1806.07572", 5000, conn=self.conn)
        p = get_paper("1806.07572", conn=self.conn)
        self.assertEqual(p["citation_count"], 5000)

    def test_backfill_catalog_citations(self):
        from src.catalog import backfill_catalog_citations
        res = backfill_catalog_citations()
        self.assertIn("after_with_cites", res)
        self.assertTrue(res["after_with_cites"] > 0)


if __name__ == "__main__":
    unittest.main()
