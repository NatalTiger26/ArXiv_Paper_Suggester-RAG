"""Unit tests for src.core.eval_stats statistical metrics."""

import unittest
from src.core.eval_stats import (
    bootstrap_ci,
    jaccard_token_diversity,
    kendall_tau,
    ndcg_at_k,
    paired_bootstrap_delta,
    spearman_rank_correlation,
    wilson_ci,
)


class TestEvalStats(unittest.TestCase):
    def test_bootstrap_ci(self):
        vals = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91]
        lo, hi = bootstrap_ci(vals, n_boot=500)
        self.assertLessEqual(lo, hi)
        self.assertGreater(lo, 0.7)
        self.assertLess(hi, 1.0)

    def test_paired_bootstrap_delta(self):
        a = [0.5, 0.6, 0.7, 0.55]
        b = [0.6, 0.7, 0.8, 0.65]
        res = paired_bootstrap_delta(a, b, n_boot=500)
        self.assertAlmostEqual(res["delta"], 0.1, places=3)
        self.assertLess(res["p_value"], 0.05)

    def test_wilson_ci(self):
        lo, hi = wilson_ci(18, 20)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 0.9)
        self.assertGreater(hi, 0.9)

    def test_spearman_and_kendall(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.2, 2.1, 3.3, 4.2, 5.1]
        rho = spearman_rank_correlation(x, y)
        tau = kendall_tau(x, y)
        self.assertAlmostEqual(rho, 1.0, places=2)
        self.assertAlmostEqual(tau, 1.0, places=2)

    def test_ndcg_at_k(self):
        ranked = ["p1", "p2", "p3"]
        gt = {"p1": 5.0, "p2": 4.0, "p3": 2.0}
        score = ndcg_at_k(ranked, gt, k=3)
        self.assertEqual(score, 1.0)

    def test_jaccard_token_diversity(self):
        titles = [
            "Neural Tangent Kernel Convergence",
            "Quantum Materials and Topological States",
            "Mechanistic Interpretability of Transformers",
        ]
        div = jaccard_token_diversity(titles)
        self.assertGreater(div, 0.8)


if __name__ == "__main__":
    unittest.main()
