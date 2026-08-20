import unittest

import numpy as np

from lotto_engine.v31_final_directional_recommendation import FEATURE_NAMES
from lotto_engine.v31_joint_ablation_audit import (
    CLEAN3_REMOVED_FEATURES,
    CLEAN4_REMOVED_FEATURES,
    FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE,
    FAIR_EXPECTED_CONSECUTIVE_PAIRS,
    FAIR_EXPECTED_NUMBER_RANGE,
    FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP,
    build_joint_scenario_weights,
)


class V31JointAblationAuditTests(unittest.TestCase):
    def _stats(self):
        width = len(FEATURE_NAMES)
        return {
            "sum_outer": np.eye(width, dtype=float) * 20.0,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 20,
            "solved_targets": 5,
        }

    def test_clean3_direct_and_retrained_zero_removed_features(self):
        weights = build_joint_scenario_weights(self._stats(), ridge_lambda=2.0)
        for scenario in ("direct_without:clean3", "retrained_without:clean3"):
            vector = weights[scenario]
            for feature in CLEAN3_REMOVED_FEATURES:
                self.assertEqual(float(vector[FEATURE_NAMES.index(feature)]), 0.0)
            self.assertNotEqual(
                float(vector[FEATURE_NAMES.index("number_full_log_lift")]), 0.0
            )

    def test_clean4_also_removes_number_full_log_lift(self):
        weights = build_joint_scenario_weights(self._stats(), ridge_lambda=2.0)
        for scenario in ("direct_without:clean4", "retrained_without:clean4"):
            vector = weights[scenario]
            for feature in CLEAN4_REMOVED_FEATURES:
                self.assertEqual(float(vector[FEATURE_NAMES.index(feature)]), 0.0)

    def test_fair_reference_values_are_exact_6_of_45_expectations(self):
        self.assertAlmostEqual(FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP, 0.8)
        self.assertAlmostEqual(FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE, 0.5994353632754086)
        self.assertAlmostEqual(FAIR_EXPECTED_CONSECUTIVE_PAIRS, 2.0 / 3.0)
        self.assertAlmostEqual(FAIR_EXPECTED_NUMBER_RANGE, 230.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
