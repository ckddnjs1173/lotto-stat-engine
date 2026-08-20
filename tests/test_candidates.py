import unittest

from lotto_engine.candidates import TOTAL_COMBINATION_COUNT, generate_candidates


class CandidateGenerationTests(unittest.TestCase):
    def test_total_combination_count_is_exact(self):
        self.assertEqual(TOTAL_COMBINATION_COUNT, 8_145_060)

    def test_sample_returns_exact_unique_requested_count(self):
        candidates = generate_candidates(2_000, seed=1237)
        self.assertEqual(len(candidates), 2_000)
        self.assertEqual(len({tuple(candidate) for candidate in candidates}), 2_000)

    def test_invalid_candidate_counts_fail_loudly(self):
        with self.assertRaises(ValueError):
            generate_candidates(0, seed=1)
        with self.assertRaises(ValueError):
            generate_candidates(TOTAL_COMBINATION_COUNT + 1, seed=1)


if __name__ == "__main__":
    unittest.main()
