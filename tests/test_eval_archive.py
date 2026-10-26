import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.final_eval import archive_previous_eval


class TestEvalArchive(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.eval_dir = self.temp_dir / "rag_eval"
        self.results_dir = self.temp_dir / "results"
        self.eval_dir.mkdir(parents=True)
        self.results_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_archive_when_missing(self):
        res = archive_previous_eval(self.eval_dir, self.results_dir)
        self.assertIsNone(res)

    def test_archive_creates_backup(self):
        final_json = self.eval_dir / "final_eval.json"
        data = {
            "summary": {"timestamp": "2026-10-20T12:00:00Z", "mrr": 0.55},
            "records": [{"q": 1}],
        }
        final_json.write_text(json.dumps(data), encoding="utf-8")

        rep_md = self.results_dir / "EVALUATION_REPORT.md"
        rep_md.write_text("# Report", encoding="utf-8")

        archived = archive_previous_eval(self.eval_dir, self.results_dir)
        self.assertIsNotNone(archived)
        self.assertTrue(archived.exists())
        self.assertTrue((archived / "final_eval.json").exists())
        self.assertTrue((archived / "EVALUATION_REPORT.md").exists())

        # If run again with same content, it should point to existing archive
        archived_same = archive_previous_eval(self.eval_dir, self.results_dir)
        self.assertEqual(archived, archived_same)

        # If final_eval.json has new content, it should create an incremented archive
        data["summary"]["mrr"] = 0.58
        final_json.write_text(json.dumps(data), encoding="utf-8")
        archived_next = archive_previous_eval(self.eval_dir, self.results_dir)
        self.assertNotEqual(archived, archived_next)
        self.assertTrue((archived_next / "final_eval.json").exists())


if __name__ == "__main__":
    unittest.main()
