import unittest

from src.rag.rag_pipeline import _build_context, _chunk_rank_score, _dedupe_chunks


class TestContextOrdering(unittest.TestCase):
    def test_rrf_score_preferred_over_dense(self):
        # Chunk A has high dense score (0.95) but low RRF score (0.01)
        # Chunk B has lower dense score (0.70) but high RRF score (0.05) due to BM25 match
        chunk_a = {
            "chunk_id": "a",
            "paper_id": "paper_1",
            "section": "1 Introduction",
            "score": 0.95,
            "rrf_score": 0.01,
            "text": "Text A",
        }
        chunk_b = {
            "chunk_id": "b",
            "paper_id": "paper_2",
            "section": "1 Introduction",
            "score": 0.70,
            "rrf_score": 0.05,
            "text": "Text B",
        }

        self.assertGreater(_chunk_rank_score(chunk_b), _chunk_rank_score(chunk_a))

        context_str, used_chunks = _build_context([chunk_a, chunk_b], max_chars=1000)
        self.assertEqual(len(used_chunks), 2)
        # Chunk B should come FIRST because its RRF hybrid score is higher
        self.assertEqual(used_chunks[0]["chunk_id"], "b")
        self.assertEqual(used_chunks[1]["chunk_id"], "a")
        self.assertTrue(context_str.startswith("[1] paper_2"))

    def test_diversity_cap(self):
        chunks = [
            {"chunk_id": f"p1_{i}", "paper_id": "paper_1", "section": f"Sec {i}", "rrf_score": 0.10 - i * 0.01, "text": f"p1_{i}"}
            for i in range(5)
        ] + [
            {"chunk_id": "p2_0", "paper_id": "paper_2", "section": "Sec 0", "rrf_score": 0.03, "text": "p2_0"}
        ]
        deduped = _dedupe_chunks(chunks, max_per_paper=3)
        p1_count = sum(1 for c in deduped if c["paper_id"] == "paper_1")
        self.assertLessEqual(p1_count, 3)
        # paper_2 should be included despite lower scores than some p1 chunks because of diversity cap
        p2_present = any(c["paper_id"] == "paper_2" for c in deduped)
        self.assertTrue(p2_present)


if __name__ == "__main__":
    unittest.main()
