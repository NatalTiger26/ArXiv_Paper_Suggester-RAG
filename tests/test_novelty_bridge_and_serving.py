"""Unit tests for novelty bridge discovery and client-side vector serving."""

import unittest
import numpy as np
from src.novelty.bridge import find_conceptual_bridges
from starlette.testclient import TestClient
from src.api.server import app


class TestNoveltyBridgeAndServing(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        np.random.seed(42)
        self.corpus_vectors = np.random.randn(50, 256).astype(np.float32)
        norms = np.linalg.norm(self.corpus_vectors, axis=1, keepdims=True)
        self.corpus_vectors /= norms
        self.corpus_ids = [f"paper_{i}" for i in range(50)]

    def test_conceptual_bridge_discovery(self):
        # Domain A is unit vector along dimension 0
        va = np.zeros(256, dtype=np.float32)
        va[0] = 1.0
        # Domain B is unit vector along dimension 1
        vb = np.zeros(256, dtype=np.float32)
        vb[1] = 1.0

        # Construct a synthetic bridge paper that projects strongly onto both
        bridge_vec = np.zeros(256, dtype=np.float32)
        bridge_vec[0] = 0.707
        bridge_vec[1] = 0.707
        self.corpus_vectors[5] = bridge_vec
        self.corpus_ids[5] = "synthetic_bridge_paper"

        bridges = find_conceptual_bridges(
            va,
            vb,
            self.corpus_vectors,
            self.corpus_ids,
            top_k=5,
            min_affinity=0.1,
        )
        self.assertGreater(len(bridges), 0)
        # Synthetic bridge should be top ranked
        self.assertEqual(bridges[0]["paper_id"], "synthetic_bridge_paper")
        self.assertGreater(bridges[0]["bridge_score"], 0.6)

    def test_workspace_vectors_endpoint(self):
        response = self.client.get("/api/workspace/vectors?limit=100")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-encoding"), "gzip")
        # Compressed bytes should be well under 100 KB for 100 vectors
        self.assertLess(len(response.content), 100000)

    def test_novelty_bridge_endpoint(self):
        payload = {
            "domain_a": "deep learning generalization bounds",
            "domain_b": "neural tangent kernel infinite width",
            "top_k": 5,
        }
        response = self.client.post("/api/novelty/bridge", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("ok"))
        self.assertIn("bridges", data)
        self.assertLessEqual(data.get("latency_ms", 100), 50.0)


if __name__ == "__main__":
    unittest.main()
