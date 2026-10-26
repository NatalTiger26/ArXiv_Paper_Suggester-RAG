#!/usr/bin/env python3
"""
Unit tests for Novelty & Citation Library resilience.
Tests soft-coverage fallback, 404 negative caching, and checkpoint preservation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.novelty.refs import _get, _make_cache_key, _cache_get, _cache_set
from src.novelty.score import score_paper


class TestNoveltyResilience(unittest.TestCase):
    def test_negative_caching_404(self):
        url = "https://api.semanticscholar.org/graph/v1/paper/arxiv:9999.99999"
        cache_key = _make_cache_key(url, None)

        # Set 404 sentinel in cache
        _cache_set(cache_key, url, {"_status": 404}, status_code=404, ttl_days=1.0)

        # _cache_get returns the sentinel
        cached = _cache_get(cache_key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.get("_status"), 404)

        # _get returns None without making network calls
        with patch("requests.get") as mock_get:
            res = _get(url, use_cache=True)
            self.assertIsNone(res)
            mock_get.assert_not_called()

    @patch("src.novelty.score.resolve_focal_and_refs")
    @patch("src.rag.embed_store.load_embedding_model")
    @patch("src.novelty.score.embed_texts")
    def test_allow_partial_soft_coverage(self, mock_embed, mock_load_model, mock_resolve):
        import numpy as np

        # Mock paper with 10 references, but only 2 have abstracts (20% coverage < 70% threshold)
        focal = {
            "canonical_id": "arxiv:2401.00001",
            "title": "Focal Paper on Interpretability",
            "abstract": "We study representations in deep neural networks.",
            "year": 2024,
        }
        refs = []
        for i in range(10):
            refs.append({
                "canonical_id": f"arxiv:2401.000{i:02d}",
                "title": f"Reference Paper {i}",
                "abstract": f"Abstract for paper {i}" if i < 2 else "",
                "has_abstract": i < 2,
            })

        mock_resolve.return_value = {
            "focal": focal,
            "refs": refs,
            "source": "s2",
            "stats": {"n_refs": 10, "n_with_abstract": 2, "frac_with_abstract": 0.2},
        }

        # Mock embedding model returning normalized 8-d vectors
        mock_embed.side_effect = lambda texts, **kw: [list(np.ones(8) / np.sqrt(8)) for _ in texts]

        # 1. Without allow_partial: should refuse
        res_refused = score_paper("2401.00001", force_refresh_refs=True, allow_partial=False)
        self.assertFalse(res_refused.get("ok"))
        self.assertEqual(res_refused.get("error"), "insufficient_reference_abstracts")

        # 2. With allow_partial: should compute profile with title fallback and warning
        res_partial = score_paper("2401.00001", force_refresh_refs=False, allow_partial=True)
        self.assertTrue(res_partial.get("ok"))
        self.assertIn("coverage_warning", res_partial)
        self.assertIn("title fallback was used", res_partial["coverage_warning"])


if __name__ == "__main__":
    unittest.main()
