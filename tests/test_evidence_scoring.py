import unittest

from lotto_engine.evidence_scoring import (
    PAIR_MATRIX_WIDTH,
    reliability_from_brier_skill,
    score_unified_evidence,
)


class UnifiedEvidenceScoringTests(unittest.TestCase):
    @staticmethod
    def _profile(number_reliability=0.0, pair_reliability=0.0):
        return {
            "number_log_lifts": tuple([0.0] * 46),
            "pair_log_lifts": tuple([0.0] * (PAIR_MATRIX_WIDTH * PAIR_MATRIX_WIDTH)),
            "number_reliability": float(number_reliability),
            "pair_reliability": float(pair_reliability),
        }

    def test_neutral_evidence_maps_to_fifty(self):
        profile = self._profile(number_reliability=1.0, pair_reliability=1.0)
        scored = score_unified_evidence([1, 2, 3, 4, 5, 6], profile)
        self.assertAlmostEqual(scored["ranking_score"], 0.0)
        self.assertAlmostEqual(scored["prediction_score"], 50.0)

    def test_positive_number_and_pair_lift_raise_common_score(self):
        profile = self._profile(number_reliability=0.5, pair_reliability=0.5)
        number_lifts = list(profile["number_log_lifts"])
        pair_lifts = list(profile["pair_log_lifts"])
        numbers = [1, 2, 3, 4, 5, 6]
        for number in numbers:
            number_lifts[number] = 0.1
        for left_index, left in enumerate(numbers):
            for right in numbers[left_index + 1:]:
                pair_lifts[left * PAIR_MATRIX_WIDTH + right] = 0.2
                pair_lifts[right * PAIR_MATRIX_WIDTH + left] = 0.2
        profile["number_log_lifts"] = tuple(number_lifts)
        profile["pair_log_lifts"] = tuple(pair_lifts)

        scored = score_unified_evidence(numbers, profile)
        self.assertAlmostEqual(scored["score_breakdown"]["number_log_lift"], 0.1)
        self.assertAlmostEqual(scored["score_breakdown"]["pair_log_lift"], 0.2)
        self.assertAlmostEqual(scored["ranking_score"], 0.15)
        self.assertGreater(scored["prediction_score"], 50.0)

    def test_zero_reliability_neutralizes_raw_lift_without_inverting_it(self):
        profile = self._profile(number_reliability=0.0, pair_reliability=0.0)
        number_lifts = list(profile["number_log_lifts"])
        for number in [1, 2, 3, 4, 5, 6]:
            number_lifts[number] = 5.0
        profile["number_log_lifts"] = tuple(number_lifts)

        scored = score_unified_evidence([1, 2, 3, 4, 5, 6], profile)
        self.assertAlmostEqual(scored["ranking_score"], 0.0)
        self.assertAlmostEqual(scored["prediction_score"], 50.0)

    def test_reliability_is_continuous_and_requires_window_stability(self):
        summary = {
            "brier_skill_vs_uniform": 0.03,
            "windows": {
                "300": {"brier_skill_vs_uniform": 0.02},
                "100": {"brier_skill_vs_uniform": -0.01},
            },
        }
        # mean positive skill = (0.03 + 0.02 + 0) / 3; two of three windows positive.
        expected = ((0.03 + 0.02) / 3.0) * (2.0 / 3.0)
        self.assertAlmostEqual(reliability_from_brier_skill(summary), expected)

    def test_all_negative_brier_skill_has_zero_reliability(self):
        summary = {
            "brier_skill_vs_uniform": -0.01,
            "windows": {
                "300": {"brier_skill_vs_uniform": -0.02},
                "100": {"brier_skill_vs_uniform": -0.03},
            },
        }
        self.assertEqual(reliability_from_brier_skill(summary), 0.0)


if __name__ == "__main__":
    unittest.main()
