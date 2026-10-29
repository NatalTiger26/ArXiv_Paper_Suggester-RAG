"""Unit tests and empirical fidelity verification for vector quantization."""

import unittest
import numpy as np
from src.core.quantization import (
    quantize_embeddings,
    dequantize_embeddings,
    two_stage_mips,
)


class TestVectorQuantization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 1000
        d = 256
        raw = np.random.randn(n, d).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.vecs_f32 = raw / norms

    def test_memory_compression_factor(self):
        vecs_i8, scales = quantize_embeddings(self.vecs_f32)
        bytes_f32 = self.vecs_f32.nbytes
        bytes_i8 = vecs_i8.nbytes + scales.nbytes
        ratio = bytes_f32 / bytes_i8
        self.assertGreaterEqual(ratio, 3.8)

    def test_reconstruction_error(self):
        vecs_i8, scales = quantize_embeddings(self.vecs_f32)
        recon = dequantize_embeddings(vecs_i8, scales)
        # Cosine similarity between reconstructed and original
        cos_sims = np.sum(recon * self.vecs_f32, axis=1) / (
            np.linalg.norm(recon, axis=1) * np.linalg.norm(self.vecs_f32, axis=1)
        )
        self.assertGreater(float(np.mean(cos_sims)), 0.999)

    def test_two_stage_mips_exact_fidelity(self):
        vecs_i8, scales = quantize_embeddings(self.vecs_f32)
        q = np.random.randn(256).astype(np.float32)
        q /= np.linalg.norm(q)

        # Baseline: Exact pure Float32 search
        scores_all = np.dot(self.vecs_f32, q)
        gold_top20 = np.argsort(scores_all)[::-1][:20]

        # Two-stage search
        res_indices, res_scores = two_stage_mips(
            q,
            vecs_i8,
            scales,
            self.vecs_f32,
            top_k=20,
            coarse_k=50,
        )

        # Invariant: 100% agreement on Top-20 ranking order and values
        self.assertEqual(list(res_indices), list(gold_top20))
        np.testing.assert_allclose(res_scores, scores_all[gold_top20], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
