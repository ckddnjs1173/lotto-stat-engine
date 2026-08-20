import unittest

import numpy as np

from lotto_engine.v31_component_influence_audit import (
    AUDIT_FEATURES,
    _model_bundle,
    _pool_summary,
    fit_subset_pairwise_ridge,
    score_scenarios_for_vector,
)
from lotto_engine.v31_final_directional_recommendation import FEATURE_NAMES


class V31ComponentInfluenceAuditTests(unittest.TestCase):
    def _stats(self):
        width = len(FEATURE_NAMES)
        return {
            "sum_outer": np.eye(width, dtype=float) * 20.0,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 20,
            "solved_targets": 7,
        }

    def test_subset_ridge_excludes_requested_feature(self):
        removed = FEATURE_NAMES.index("previous_draw_overlap")
        active = tuple(index for index in range(len(FEATURE_NAMES)) if index != removed)
        model = fit_subset_pairwise_ridge(self._stats(), active, ridge_lambda=2.0)
        self.assertNotIn(removed, model["active_indices"])
        self.assertEqual(len(model["effective_weights"]), len(FEATURE_NAMES) - 1)
        self.assertTrue(np.all(np.isfinite(model["effective_weights"])))

    def test_direct_without_is_exact_full_minus_named_contribution(self):
        models = _model_bundle(self._stats(), ridge_lambda=2.0)
        vector = np.linspace(-0.8, 0.8, len(FEATURE_NAMES))
        scores = score_scenarios_for_vector(vector, models)
        full_weights = np.asarray(models["full"]["effective_weights"], dtype=float)
        for feature in AUDIT_FEATURES:
            index = FEATURE_NAMES.index(feature)
            expected = scores["full"] - full_weights[index] * vector[index]
            self.assertAlmostEqual(scores[f"direct_without:{feature}"], expected)
            self.assertTrue(np.isfinite(scores[f"retrained_without:{feature}"]))

    def test_pool_summary_exposes_concentration_metrics(self):
        ranked = [
            (3.0, 3, (1, 2, 10, 20, 30, 40)),
            (2.0, 2, (1, 3, 11, 21, 31, 41)),
            (1.0, 1, (4, 5, 12, 22, 32, 42)),
        ]
        summary = _pool_summary(ranked, frozenset({1, 2, 3, 4, 5, 6}), top_k=2)
        self.assertEqual(summary["pool_size"], 3)
        self.assertAlmostEqual(summary["number_inclusion_rates"]["1"], 2 / 3)
        self.assertGreater(summary["mean_previous_draw_overlap"], 0.0)
        self.assertIn("any_consecutive_pair_rate", summary)
        self.assertIn("mean_number_range", summary)


if __name__ == "__main__":
    unittest.main()
