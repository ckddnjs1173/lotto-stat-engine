import unittest

import numpy as np

from lotto_engine.v31_clean3_recommendation import (
    ACTIVE_PRODUCTION_FEATURES,
    DISABLED_PRODUCTION_FEATURES,
    fit_clean3_pairwise_ridge,
)
from lotto_engine.v31_final_directional_recommendation import FEATURE_NAMES


class V31Clean3RecommendationTests(unittest.TestCase):
    def _stats(self):
        width = len(FEATURE_NAMES)
        return {
            "sum_outer": np.eye(width, dtype=float) * 10.0,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 10,
            "solved_targets": 3,
        }

    def test_clean3_has_eight_active_features(self):
        self.assertEqual(len(FEATURE_NAMES), 11)
        self.assertEqual(len(ACTIVE_PRODUCTION_FEATURES), 8)
        self.assertEqual(
            DISABLED_PRODUCTION_FEATURES,
            ("previous_draw_overlap", "number_range", "consecutive_pairs"),
        )

    def test_clean3_retrains_subset_and_zero_fills_disabled_weights(self):
        model = fit_clean3_pairwise_ridge(self._stats(), ridge_lambda=2.0)
        self.assertEqual(tuple(model["active_features"]), ACTIVE_PRODUCTION_FEATURES)
        self.assertEqual(tuple(model["disabled_features"]), DISABLED_PRODUCTION_FEATURES)
        for feature in DISABLED_PRODUCTION_FEATURES:
            index = FEATURE_NAMES.index(feature)
            self.assertEqual(float(model["effective_weights"][index]), 0.0)
            self.assertEqual(float(model["scaled_weights"][index]), 0.0)
        for feature in ACTIVE_PRODUCTION_FEATURES:
            index = FEATURE_NAMES.index(feature)
            self.assertNotEqual(float(model["effective_weights"][index]), 0.0)


if __name__ == "__main__":
    unittest.main()
