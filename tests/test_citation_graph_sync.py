import sqlite3
import unittest
from pathlib import Path

from src.settings import PROJECT_ROOT
from src.citation_library import store


class TestCitationGraphSync(unittest.TestCase):
    def setUp(self):
        self.db_path = PROJECT_ROOT / "data" / "citation_library" / "graph.sqlite3"

    def test_landmark_papers_have_global_citations(self):
        if not self.db_path.exists():
            self.skipTest("Citation graph sqlite database not found.")
        conn = store.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL AND citation_count > 0"
        )
        count = cursor.fetchone()[0]
        conn.close()
        self.assertGreater(count, 0, "Expected at least some landmark papers to have global citation counts")

    def test_example_citers_bare_and_prefixed(self):
        if not self.db_path.exists():
            self.skipTest("Citation graph sqlite database not found.")
        conn = store.connect(self.db_path)
        # 1706.03762 (Attention is All You Need) is a well-known hub in the graph
        citers_bare = store.example_citers(conn, "1706.03762", limit=5)
        citers_prefixed = store.example_citers(conn, "arxiv:1706.03762", limit=5)
        conn.close()

        self.assertEqual(len(citers_bare), len(citers_prefixed))
        if citers_bare:
            self.assertIsInstance(citers_bare[0], str)


if __name__ == "__main__":
    unittest.main()
