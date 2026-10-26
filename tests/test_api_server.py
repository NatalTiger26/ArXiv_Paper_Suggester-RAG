import unittest
from starlette.testclient import TestClient

from src.api.server import app


class TestAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("catalog", data)
        self.assertIn("telemetry", data)

    def test_search_catalog(self):
        resp = self.client.get("/api/search?q=neural&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("results", data)
        self.assertIn("count", data)
        self.assertIn("latency_ms", data)

    def test_hubs_endpoint(self):
        resp = self.client.get("/api/hubs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("hubs", data)

    def test_novelty_endpoint(self):
        resp = self.client.get("/api/novelty?arxiv_id=1806.07572")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("ok"))
        self.assertIn("m", data)
        self.assertIn("Delta", data)
        self.assertIn("R2_plus", data)
        self.assertIn("viz_2d", data)

    def test_telemetry_endpoints(self):
        resp_get = self.client.get("/api/telemetry")
        self.assertEqual(resp_get.status_code, 200)
        resp_reset = self.client.post("/api/telemetry/reset")
        self.assertEqual(resp_reset.status_code, 200)

    def test_validation_errors(self):
        # Missing query in recommend
        resp_rec = self.client.post("/api/recommend", json={})
        self.assertEqual(resp_rec.status_code, 400)

        # Missing query in ask
        resp_ask = self.client.post("/api/ask", json={})
        self.assertEqual(resp_ask.status_code, 400)

        # Missing query parameter in ask stream
        resp_stream = self.client.get("/api/ask/stream")
        self.assertEqual(resp_stream.status_code, 400)

        # Missing arxiv_id in ingest
        resp_ing = self.client.post("/api/ingest", json={})
        self.assertEqual(resp_ing.status_code, 400)

    def test_static_web_studio_served(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("ArXiv Discovery & RAG Studio", resp.text)


if __name__ == "__main__":
    unittest.main()
