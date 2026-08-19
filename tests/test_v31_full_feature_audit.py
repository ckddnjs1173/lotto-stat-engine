import unittest

import numpy as np

from lotto_engine.v31_core import FEATURE_NAMES
from lotto_engine.v31_full_feature_audit import (
    FEATURE_GROUPS,
    _paired_delta,
    build_scenario_weight_matrix,
    scenario_removals,
    tie_safe_percentile,
)
from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC


class V31FullFeatureAuditTests(unittest.TestCase):
    def _stats(self):
        width = len(FEATURE_NAMES)
        second = np.eye(width, dtype=float) * 20.0
        second += np.full((width, width), 0.5, dtype=float)
        return {
            "sum_outer": second,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 20,
            "solved_targets": 8,
        }

    def test_scenarios_cover_every_feature_and_declared_group(self):
        removals = scenario_removals()
        self.assertIn("full", removals)
        for feature in FEATURE_NAMES:
            self.assertEqual(removals[f"without_feature:{feature}"], (feature,))
        for group, features in FEATURE_GROUPS.items():
            self.assertEqual(removals[f"without_group:{group}"], tuple(features))

    def test_weight_matrix_zeroes_removed_coordinates(self):
        bundle = build_scenario_weight_matrix(self._stats(), FULL11_BASELINE_SPEC)
        names = bundle["scenario_names"]
        matrix = bundle["scenario_weight_matrix"]
        for feature in FEATURE_NAMES:
            row = names.index(f"without_feature:{feature}")
            column = FEATURE_NAMES.index(feature)
            self.assertEqual(float(matrix[row, column]), 0.0)

    def test_tie_safe_percentile_gives_half_credit_for_equal_scores(self):
        baseline = np.asarray([1.0, 2.0, 2.0, 3.0])
        self.assertAlmostEqual(tie_safe_percentile(2.0, baseline), 50.0)

    def test_paired_delta_direction_is_inconclusive_when_ci_crosses_zero(self):
        result = _paired_delta(
            [49.0, 51.0, 50.0, 50.0],
            [50.0, 50.0, 50.0, 50.0],
            bootstrap_reps=100,
            seed=123,
        )
        self.assertEqual(result["direction"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
