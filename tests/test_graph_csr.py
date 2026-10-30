"""Unit tests for src.core.graph_csr Compressed Sparse Row citation graph."""

import unittest
from src.core.graph_csr import CitationGraphCSR, get_citation_csr


class TestCitationGraphCSR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.settings import DATA_DIR
        db_path = DATA_DIR / "citation_library" / "graph.sqlite3"
        if not db_path.exists():
            raise unittest.SkipTest("Local citation graph database not present on disk")

    def test_singleton_load(self):
        csr = get_citation_csr()
        self.assertIsNotNone(csr)
        self.assertGreater(csr.n_nodes, 0)
        self.assertGreater(csr.n_edges, 0)

    def test_2hop_neighbors(self):
        csr = get_citation_csr()
        neighbors = csr.get_2hop_neighbors("arxiv:1706.03762", max_results=20)
        self.assertIsInstance(neighbors, list)

    def test_pagerank_computation(self):
        csr = get_citation_csr()
        pr = csr.compute_pagerank(max_iter=5)
        self.assertEqual(len(pr), csr.n_nodes)
        self.assertGreater(float(pr.sum()), 0.0)

    def test_get_node_pagerank(self):
        csr = get_citation_csr()
        score = csr.get_node_pagerank("1706.03762")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hub_cache_accuracy(self):
        csr = get_citation_csr()
        self.assertGreater(len(csr._hub_cache), 0)
        for hub_id in list(csr._hub_cache.keys())[:5]:
            cached = csr.get_2hop_neighbors(hub_id, max_results=50)
            idx = csr.node_to_idx[hub_id]
            raw = csr._raw_2hop_traversal(idx, max_results=50)
            self.assertEqual(cached, raw)

    def test_personalized_pagerank(self):
        csr = get_citation_csr()
        pr_uniform = csr.compute_pagerank(max_iter=5)
        pr_default = csr.compute_pagerank(max_iter=5, seed_node=None)
        # Uniform and None must be identical
        self.assertTrue((pr_uniform == pr_default).all())

        # Test personalized to attention paper
        seed = "arxiv:1706.03762"
        if seed in csr.node_to_idx:
            pr_seeded = csr.compute_pagerank(max_iter=5, seed_node=seed)
            seed_idx = csr.node_to_idx[seed]
            self.assertGreater(pr_seeded[seed_idx], pr_uniform[seed_idx])


if __name__ == "__main__":
    unittest.main()
