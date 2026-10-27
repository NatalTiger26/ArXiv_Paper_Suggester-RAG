import unittest
import numpy as np

from src.novelty.viz import project_subspace_2d


class TestNoveltyGeometry(unittest.TestCase):
    def test_cone_coordinates_math(self):
        # Construct synthetic orthogonal vectors in 4D
        d = 4
        # Dominant parent e1 = [1, 0, 0, 0]
        y_parent = np.array([1.0, 0.0, 0.0, 0.0])
        # Second parent y2 = [0, 1, 0, 0]
        y2 = np.array([0.0, 1.0, 0.0, 0.0])
        Y = np.column_stack([y_parent, y2])

        # Focal paper x has cosine m=0.8 with parent, 0.4 with y2, and residual in 3rd dim
        # x = 0.8 * e1 + 0.4 * e2 + sqrt(1 - 0.8^2 - 0.4^2) * e3
        # norm(x)^2 = 0.64 + 0.16 + 0.20 = 1.0
        x = np.array([0.8, 0.4, np.sqrt(0.20), 0.0])

        m = 0.8
        m_plus_sq = 0.64
        # NNLS projection lambda = [0.8, 0.4]
        # x_hat = [0.8, 0.4, 0, 0]
        # R2_plus = 0.64 + 0.16 = 0.80
        r2_plus = 0.80
        # Delta = R2_plus - m_plus_sq = 0.80 - 0.64 = 0.16
        delta = 0.16
        lam = np.array([0.8, 0.4])

        res = project_subspace_2d(
            x=x,
            Y=Y,
            ref_ids=["parent_1", "parent_2"],
            m=m,
            r2_plus=r2_plus,
            delta=delta,
            lam=lam,
        )

        self.assertTrue(res["ok"])
        metrics = res["metrics"]
        self.assertAlmostEqual(metrics["m"], 0.8, places=3)
        self.assertAlmostEqual(metrics["R2_plus"], 0.8, places=3)
        self.assertAlmostEqual(metrics["Delta"], 0.16, places=3)
        # Residual distance = sqrt(1 - 0.8) = sqrt(0.2) ≈ 0.4472
        self.assertAlmostEqual(metrics["residual_distance"], np.sqrt(0.2), places=3)

        points = {p["id"]: p for p in res["points"]}
        # Dominant parent should sit at (1, 0)
        parent_pt = points["parent_1"]
        self.assertAlmostEqual(parent_pt["x"], 1.0, places=3)
        self.assertAlmostEqual(parent_pt["y"], 0.0, places=3)

        # Cone projection x_hat should sit at (m, sqrt(Delta)) = (0.8, 0.4)
        proj_pt = points["reconstruction"]
        self.assertAlmostEqual(proj_pt["x"], 0.8, places=3)
        self.assertAlmostEqual(proj_pt["y"], 0.4, places=3)
        # Radial distance = sqrt(0.8^2 + 0.4^2) = sqrt(0.8) = sqrt(R2_plus)
        radius = np.sqrt(proj_pt["x"]**2 + proj_pt["y"]**2)
        self.assertAlmostEqual(radius, np.sqrt(0.8), places=3)

    def test_nan_and_zero_norm_numerical_stability(self):
        from src.novelty.metrics import _unit, prepare_matrix, nnls_cone_r2, compute_profile

        # Zero vector handling
        z = np.zeros(4)
        self.assertTrue(np.all(_unit(z) == 0.0))

        # NaN/Inf vector handling
        bad = np.array([np.nan, 1.0, np.inf, -np.inf])
        u_bad = _unit(bad)
        self.assertTrue(np.all(np.isfinite(u_bad)))

        # Matrix preparation with corrupted columns
        x = np.array([1.0, 0.0, 0.0, 0.0])
        Y_corrupt = np.array([
            [np.nan, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        x_clean, Y_clean = prepare_matrix(x, Y_corrupt)
        self.assertTrue(np.all(np.isfinite(x_clean)))
        self.assertTrue(np.all(np.isfinite(Y_clean)))

        # NNLS cone projection with zero/non-finite inputs
        r2, lam = nnls_cone_r2(x, Y_corrupt)
        self.assertTrue(np.isfinite(r2))
        self.assertTrue(np.all(np.isfinite(lam)))

        # Profile computation produces strictly finite metrics
        prof = compute_profile(x, Y_corrupt)
        self.assertTrue(prof["n_refs_used"] >= 0)
        if prof.get("m") is not None:
            self.assertTrue(np.isfinite(prof["m"]))
            self.assertTrue(np.isfinite(prof["Delta"]))
            self.assertTrue(np.isfinite(prof["R2_plus"]))

    def test_cache_vector_finiteness_validation(self):
        from src.novelty.cache import pack_vector, purge_corrupted_embeddings
        import sqlite3

        # Packing NaN vector should raise ValueError
        bad_vec = [1.0, np.nan, 3.0]
        with self.assertRaises(ValueError):
            pack_vector(bad_vec)

        # In-memory SQLite test of purge_corrupted_embeddings
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE embeddings (
                canonical_id TEXT, model TEXT, dim INTEGER, vector BLOB, embedded_at REAL,
                PRIMARY KEY (canonical_id, model, dim)
            )
        """)
        good_blob = np.array([1.0, 2.0, 3.0], dtype="float32").tobytes()
        bad_blob = np.array([1.0, np.nan, 3.0], dtype="float32").tobytes()

        conn.execute("INSERT INTO embeddings VALUES ('good', 'm', 3, ?, 0)", (good_blob,))
        conn.execute("INSERT INTO embeddings VALUES ('bad', 'm', 3, ?, 0)", (bad_blob,))
        conn.commit()

        purged = purge_corrupted_embeddings(conn)
        self.assertEqual(purged, 1)

        remaining = conn.execute("SELECT canonical_id FROM embeddings").fetchall()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["canonical_id"], "good")


if __name__ == "__main__":
    unittest.main()
