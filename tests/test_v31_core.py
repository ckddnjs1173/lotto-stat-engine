import unittest

import numpy as np

from lotto_engine.v31_core import (
    FEATURE_NAMES,
    _tie_break,
    fit_subset_pairwise_ridge,
)
from lotto_engine.v31_model_spec import (
    CLEAN3_CANDIDATE_SPEC,
    FULL11_BASELINE_SPEC,
)
from lotto_engine.v31_scenario_recommendation import scenario_spec


class V31CoreTests(unittest.TestCase):
    def _stats(self):
        width = len(FEATURE_NAMES)
        return {
            "sum_outer": np.eye(width, dtype=float) * 20.0,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 20,
            "solved_targets": 7,
        }

    def test_exhaustive_tie_key_does_not_depend_on_seed(self):
        combo = (3, 5, 9, 16, 20, 23)
        self.assertEqual(_tie_break(combo, 0), _tie_break(combo, 999999))

    def test_lexicographically_smaller_combination_wins_equal_score_tie(self):
        left = (1, 2, 3, 4, 5, 6)
        right = (1, 2, 3, 4, 5, 7)
        self.assertGreater(_tie_break(left), _tie_break(right))

    def test_model_spec_hash_is_stable_and_scenario_specific(self):
        self.assertEqual(FULL11_BASELINE_SPEC.sha256(), FULL11_BASELINE_SPEC.sha256())
        self.assertNotEqual(FULL11_BASELINE_SPEC.sha256(), CLEAN3_CANDIDATE_SPEC.sha256())
        self.assertEqual(len(FULL11_BASELINE_SPEC.sha256()), 64)

    def test_no_v31_spec_claims_promoted_status(self):
        self.assertIn("not_promoted", FULL11_BASELINE_SPEC.status)
        self.assertIn("requires_joint_and_stability_audit", CLEAN3_CANDIDATE_SPEC.status)

    def test_scenario_selection_is_explicit(self):
        self.assertIs(scenario_spec("full11"), FULL11_BASELINE_SPEC)
        self.assertIs(scenario_spec("clean3"), CLEAN3_CANDIDATE_SPEC)
        with self.assertRaises(ValueError):
            scenario_spec("current")

    def test_subset_ridge_exposes_condition_diagnostics(self):
        model = fit_subset_pairwise_ridge(
            self._stats(),
            range(len(FEATURE_NAMES)),
            ridge_lambda=2.0,
        )
        self.assertTrue(np.isfinite(model["effective_weights"]).all())
        self.assertEqual(
            len(model["scaled_second_moment_eigenvalues"]),
            len(FEATURE_NAMES),
        )
        self.assertTrue(np.isfinite(model["scaled_second_moment_condition_number"]))


if __name__ == "__main__":
    unittest.main()
