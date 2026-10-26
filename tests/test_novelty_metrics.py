"""Unit tests for novelty geometric metrics (NNLS cone projection, R2, Delta, attribution)."""

from __future__ import annotations

import unittest
import numpy as np

from src.novelty.metrics import compute_profile, nnls_cone_r2


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


class TestNoveltyMetrics(unittest.TestCase):
    def test_identity(self) -> None:
        d = 64
        y1 = np.zeros(d)
        y1[0] = 1.0
        Y = np.column_stack([y1, np.eye(d)[:, 1:4]])
        x = y1.copy()
        p = compute_profile(x, Y)
        self.assertAlmostEqual(p["m"], 1.0, places=5)
        self.assertAlmostEqual(p["R2_plus"], 1.0, places=4)
        self.assertAlmostEqual(p["Delta"], 0.0, places=4)

    def test_exact_blend(self) -> None:
        d = 64
        y1 = np.zeros(d)
        y1[0] = 1.0
        y2 = np.zeros(d)
        y2[1] = 1.0
        Y = np.column_stack([y1, y2])
        x = (y1 + y2) / np.sqrt(2.0)
        p = compute_profile(x, Y, ref_ids=["a", "b"])
        self.assertAlmostEqual(p["m"], 1.0 / np.sqrt(2.0), places=4)
        self.assertAlmostEqual(p["R2_plus"], 1.0, places=3)
        self.assertAlmostEqual(p["Delta"], 0.5, places=3)
        w = {a["ref_id"]: a["weight"] for a in p["attribution"]}
        self.assertAlmostEqual(w.get("a", 0), 0.5, delta=0.05)
        self.assertAlmostEqual(w.get("b", 0), 0.5, delta=0.05)

    def test_orthogonal(self) -> None:
        d = 64
        Y = np.eye(d)[:, :5]
        x = np.zeros(d)
        x[10] = 1.0
        p = compute_profile(x, Y)
        self.assertLess(p["R2_plus"], 1e-5)
        self.assertAlmostEqual(p["Delta"], 0.0, places=5)

    def test_opposite(self) -> None:
        d = 32
        y1 = np.zeros(d)
        y1[0] = 1.0
        x = -y1
        p = compute_profile(x, y1.reshape(-1, 1))
        self.assertAlmostEqual(p["m_plus_sq"], 0.0, places=8)
        self.assertLess(p["R2_plus"], 1e-5)

    def test_permutation_invariance(self) -> None:
        rng = np.random.default_rng(0)
        d, n = 40, 6
        Y = rng.normal(size=(d, n))
        Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
        x = _unit(Y[:, 0] * 0.7 + Y[:, 1] * 0.3)
        p1 = compute_profile(x, Y, ref_ids=[str(i) for i in range(n)])
        perm = rng.permutation(n)
        Y2 = Y[:, perm]
        ids2 = [str(i) for i in perm]
        p2 = compute_profile(x, Y2, ref_ids=ids2)
        for k in ("m", "a", "R2_plus", "Delta", "D_median"):
            self.assertAlmostEqual(p1[k] or 0.0, p2[k] or 0.0, places=6)

    def test_duplication(self) -> None:
        d = 32
        y1 = np.zeros(d)
        y1[0] = 1.0
        y2 = np.zeros(d)
        y2[1] = 1.0
        Y = np.column_stack([y1, y2])
        x = y1.copy()
        p1 = compute_profile(x, Y)
        Ydup = np.column_stack([y1, y1, y2])
        p2 = compute_profile(x, Ydup)
        self.assertAlmostEqual(p1["R2_plus"], p2["R2_plus"], places=4)

    def test_chance_level(self) -> None:
        rng = np.random.default_rng(1)
        d, n, trials = 64, 8, 30
        r2s = []
        for _ in range(trials):
            Y = rng.normal(size=(d, n))
            Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
            x = rng.normal(size=d)
            x = x / np.linalg.norm(x)
            p = compute_profile(x, Y)
            r2s.append(p["R2_plus"])
        mean_r2 = float(np.mean(r2s))
        bound = n / d
        self.assertLessEqual(mean_r2, bound + 0.15)

    def test_nnls_crosscheck(self) -> None:
        from scipy.optimize import nnls

        rng = np.random.default_rng(2)
        d, n = 50, 7
        Y = rng.normal(size=(d, n))
        x = np.abs(rng.normal(size=d))
        x = x / np.linalg.norm(x)
        Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
        r2_a, lam_a = nnls_cone_r2(x, Y)
        lam_b, _ = nnls(Y, x)
        proj = Y @ lam_b
        r2_b = float(np.dot(proj, proj))
        self.assertAlmostEqual(r2_a, r2_b, places=5)


if __name__ == "__main__":
    unittest.main()
