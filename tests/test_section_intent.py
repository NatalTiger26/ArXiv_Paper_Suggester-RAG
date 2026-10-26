import unittest

from src.rag.hybrid import detect_query_intent, section_intent_boost


class TestSectionIntent(unittest.TestCase):
    def test_detect_intent(self):
        self.assertEqual(detect_query_intent("What is the main contribution of the paper?"), "CONTRIBUTION_SUMMARY")
        self.assertEqual(detect_query_intent("What are the limitations and failure modes?"), "LIMITATIONS_FUTURE")
        self.assertEqual(detect_query_intent("What benchmark datasets were evaluated?"), "EMPIRICAL_EVAL")
        self.assertEqual(detect_query_intent("Proof of Theorem 2 convergence bound"), "THEORETICAL_TECH")

    def test_section_boost(self):
        # Contribution question boosts Abstract/Introduction
        boost_abs = section_intent_boost("Abstract", "CONTRIBUTION_SUMMARY")
        boost_app = section_intent_boost("Appendix B Proofs", "CONTRIBUTION_SUMMARY")
        self.assertGreater(boost_abs, boost_app)

        # Empirical question boosts Experiments
        boost_exp = section_intent_boost("4 Experiments and Results", "EMPIRICAL_EVAL")
        self.assertGreater(boost_exp, 0.0)


if __name__ == "__main__":
    unittest.main()
