import tempfile
import unittest
from pathlib import Path

from src.telemetry import (
    calculate_cost_usd,
    get_telemetry_summary,
    record_call,
    reset_telemetry,
)


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.db_path = Path(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def test_cost_calculation(self):
        # 1M input tokens on Flash ($0.075) + 1M output ($0.30) = $0.375
        flash_cost = calculate_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000)
        self.assertAlmostEqual(flash_cost, 0.375, places=4)

        # 10k input, 1k output on Flash
        small_cost = calculate_cost_usd("gemini-2.5-flash", 10_000, 1_000)
        expected = (10_000 * 0.075 / 1e6) + (1_000 * 0.30 / 1e6)
        self.assertAlmostEqual(small_cost, round(expected, 6), places=6)

        # Pro pricing
        pro_cost = calculate_cost_usd("gemini-1.5-pro", 1_000_000, 1_000_000)
        self.assertAlmostEqual(pro_cost, 6.25, places=3)

    def test_record_and_summary(self):
        record_call(
            call_type="rag_ask",
            model="gemini-2.5-flash",
            prompt_tokens=500,
            candidate_tokens=100,
            latency_ms=250.0,
            db_path=self.db_path,
        )
        record_call(
            call_type="rag_ask",
            model="gemini-2.5-flash",
            prompt_tokens=400,
            candidate_tokens=150,
            latency_ms=180.0,
            cache_hit=True,
            db_path=self.db_path,
        )

        summary = get_telemetry_summary(db_path=self.db_path)
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_prompt_tokens"], 900)
        self.assertEqual(summary["total_candidate_tokens"], 250)
        self.assertEqual(summary["total_tokens"], 1150)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["cache_hit_rate_pct"], 50.0)
        self.assertGreater(summary["total_cost_usd"], 0.0)
        self.assertEqual(len(summary["recent_calls"]), 2)

    def test_reset(self):
        record_call(
            call_type="test",
            model="gemini-2.5-flash",
            prompt_tokens=100,
            candidate_tokens=50,
            db_path=self.db_path,
        )
        before = get_telemetry_summary(db_path=self.db_path)
        self.assertEqual(before["total_calls"], 1)

        reset_telemetry(db_path=self.db_path)
        after = get_telemetry_summary(db_path=self.db_path)
        self.assertEqual(after["total_calls"], 0)
        self.assertEqual(after["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
