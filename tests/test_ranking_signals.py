"""Unit tests and empirical verification for advanced multi-signal ranking."""

import unittest
import numpy as np
from src.ranking import (
    velocity_score,
    compute_bibliographic_coupling,
    project_contrastive_query,
    compute_cross_interaction_score,
    rerank_candidates,
)
from src.core.graph_csr import get_citation_csr


class TestRankingSignals(unittest.TestCase):
    def test_log_smoothed_temporal_velocity(self):
        # 2026 breakout preprint with 15 citations
        v_recent = velocity_score(cites=15, year=2026)
        # 2015 stagnant paper with identical 15 citations
        v_stagnant = velocity_score(cites=15, year=2015)
        # Landmark classic (e.g. Attention 2017) with 100,000 citations
        v_classic = velocity_score(cites=100000, year=2017)

        self.assertGreater(v_recent, v_stagnant)
        self.assertGreater(v_recent, 0.8)
        self.assertLess(v_stagnant, 0.1)
        self.assertEqual(v_classic, 1.0)

    def test_bibliographic_coupling_affinity(self):
        from src.settings import DATA_DIR
        if not (DATA_DIR / "citation_library" / "graph.sqlite3").exists():
            raise unittest.SkipTest("Citation graph database not found on disk")
        csr = get_citation_csr()
        # Seed with attention paper
        seed = ["arxiv:1706.03762"]
        # Disconnected or non-existent paper receives clean 0.0
        score_zero = compute_bibliographic_coupling(csr, "non_existent_paper", seed)
        self.assertEqual(score_zero, 0.0)

        # Same paper sharing its own references gets non-zero affinity
        score_self = compute_bibliographic_coupling(csr, "arxiv:1706.03762", seed)
        self.assertGreater(score_self, 0.0)

    def test_contrastive_negative_filtering_guard(self):
        np.random.seed(42)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 1. Orthogonal negative concept
        n_orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        q_proj = project_contrastive_query(q, n_orth)
        # Residual cosine with negative must be 0.0000
        self.assertAlmostEqual(float(np.dot(q_proj, n_orth)), 0.0, places=5)
        # Semantic energy retention with original query must be >= 99.5%
        self.assertGreaterEqual(float(np.dot(q_proj, q)), 0.995)

        # 2. Collinear negative concept (cos > 0.65) must trigger safety guard
        n_collinear = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            project_contrastive_query(q, n_collinear, max_collinear_cosine=0.65)

    def test_cross_encoder_reranker(self):
        query = "neural tangent kernel convergence"
        candidates = [
            {
                "paper_id": "p1",
                "title": "Random Matrix Theory and Overparameterized Models",
                "abstract": "General study of wide networks.",
                "final_score": 0.82,
            },
            {
                "paper_id": "p2",
                "title": "Neural Tangent Kernel Convergence Bounds",
                "abstract": "Exact convergence proofs for infinite width gradient flow.",
                "final_score": 0.80,
            },
        ]
        reranked = rerank_candidates(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        # Candidate 2 with exact phrase alignment in title should be elevated
        self.assertEqual(reranked[0]["paper_id"], "p2")
        self.assertIn("cross_encoder_score", reranked[0])


if __name__ == "__main__":
    unittest.main()
