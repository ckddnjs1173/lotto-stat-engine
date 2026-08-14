import random
import unittest
from unittest.mock import patch

from lotto_engine.v27_calibration import (
    _calibrated_scores,
    _calibration_buckets,
    _sample_unique_candidates,
    empirical_type_score,
)


class V27CalibrationTests(unittest.TestCase):
    def test_empirical_type_score_is_monotonic_midrank(self):
        reference = [10.0, 20.0, 20.0, 30.0]
        self.assertAlmostEqual(empirical_type_score(10.0, reference), 12.5)
        self.assertAlmostEqual(empirical_type_score(20.0, reference), 50.0)
        self.assertAlmostEqual(empirical_type_score(30.0, reference), 87.5)
        self.assertLess(
            empirical_type_score(15.0, reference),
            empirical_type_score(25.0, reference),
        )

    def test_calibration_uses_same_type_reference_scale(self):
        calibration = [
            {"pattern_type": "normal", "scores": {"base_only": 80.0}},
            {"pattern_type": "normal", "scores": {"base_only": 90.0}},
            {"pattern_type": "normal", "scores": {"base_only": 100.0}},
            {"pattern_type": "mixed", "scores": {"base_only": 0.0}},
            {"pattern_type": "mixed", "scores": {"base_only": 10.0}},
            {"pattern_type": "mixed", "scores": {"base_only": 20.0}},
            {"pattern_type": "outlier", "scores": {"base_only": 40.0}},
            {"pattern_type": "outlier", "scores": {"base_only": 50.0}},
            {"pattern_type": "outlier", "scores": {"base_only": 60.0}},
        ]
        buckets = _calibration_buckets(calibration, "base_only")
        candidates = [
            {"pattern_type": "normal", "scores": {"base_only": 90.0}},
            {"pattern_type": "mixed", "scores": {"base_only": 10.0}},
            {"pattern_type": "outlier", "scores": {"base_only": 50.0}},
        ]
        calibrated = _calibrated_scores(candidates, "base_only", buckets)
        self.assertEqual(calibrated, [50.0, 50.0, 50.0])

    def test_candidate_samples_are_unique_and_disjoint(self):
        sequence = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 7],
            [1, 2, 3, 4, 5, 8],
            [1, 2, 3, 4, 5, 9],
        ]
        with patch("lotto_engine.v27_calibration.random_combination", side_effect=sequence):
            first, seen = _sample_unique_candidates(random.Random(1), 2)
            second, _ = _sample_unique_candidates(
                random.Random(2), 2, forbidden=seen
            )
        self.assertEqual(first, [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]])
        self.assertEqual(second, [[1, 2, 3, 4, 5, 8], [1, 2, 3, 4, 5, 9]])
        self.assertTrue(set(map(tuple, first)).isdisjoint(set(map(tuple, second))))


if __name__ == "__main__":
    unittest.main()
