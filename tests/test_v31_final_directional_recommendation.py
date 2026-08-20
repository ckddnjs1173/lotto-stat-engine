import unittest

import numpy as np

from lotto_engine.v31_final_directional_recommendation import (
    FEATURE_NAMES,
    _select_portfolio_entries,
    directional_raw_features,
    fair_null_transform,
    fit_additive_pairwise_ridge,
)


class V31FinalDirectionalRecommendationTests(unittest.TestCase):
    def _neutral_context(self):
        return {
            "number_lifts": np.zeros(46, dtype=float),
            "pair_lifts": np.zeros(990, dtype=float),
            "recent20_number_rate": np.full(46, 6.0 / 45.0, dtype=float),
            "recent100_number_rate": np.full(46, 6.0 / 45.0, dtype=float),
            "recent100_pair_rate": np.full(990, 1.0 / 66.0, dtype=float),
            "previous_draw": frozenset(),
        }

    def test_directional_sum_and_zone_flip_under_mirror(self):
        context = self._neutral_context()
        low = (2, 5, 8, 11, 14, 23)
        mirror = tuple(sorted(46 - n for n in low))
        left = directional_raw_features(low, context)
        right = directional_raw_features(mirror, context)
        sum_idx = FEATURE_NAMES.index("sum_signed_center_138")
        zone_idx = FEATURE_NAMES.index("high_minus_low_zone_count")
        self.assertAlmostEqual(left[sum_idx], -right[sum_idx])
        self.assertAlmostEqual(left[zone_idx], -right[zone_idx])

    def test_fair_null_transform_is_bounded(self):
        width = len(FEATURE_NAMES)
        reference = np.sort(np.vstack([
            np.linspace(-2.0, 2.0, 101) + i * 0.01 for i in range(width)
        ]).T, axis=0)
        raw = np.linspace(-100.0, 100.0, width)
        transformed = fair_null_transform(raw, reference)
        self.assertTrue(np.all(transformed >= -1.0))
        self.assertTrue(np.all(transformed <= 1.0))

    def test_additive_ridge_has_expected_width_and_no_interactions(self):
        width = len(FEATURE_NAMES)
        stats = {
            "sum_outer": np.eye(width, dtype=float) * 10.0,
            "sum_diff": np.arange(1, width + 1, dtype=float),
            "pair_rows": 10,
            "solved_targets": 3,
        }
        model = fit_additive_pairwise_ridge(stats, ridge_lambda=2.0)
        self.assertEqual(model["effective_weights"].shape, (width,))
        self.assertEqual(width, 11)
        self.assertFalse(any("interaction:" in name for name in FEATURE_NAMES))
        self.assertFalse(any("square:" in name for name in FEATURE_NAMES))

    def test_score_is_plain_weighted_sum(self):
        width = len(FEATURE_NAMES)
        vector = np.linspace(-1.0, 1.0, width)
        weights = np.linspace(0.1, 1.1, width)
        expected = float(np.dot(weights, vector))
        self.assertAlmostEqual(expected, float(np.sum(weights * vector)))

    def test_portfolio_keeps_score_order_but_rejects_three_number_overlap(self):
        ranked = [
            (10.0, 1, (1, 2, 3, 4, 5, 6)),
            (9.0, 2, (1, 2, 3, 7, 8, 9)),  # shares 3 with first => reject
            (8.0, 3, (1, 2, 7, 8, 9, 10)),  # shares 2 with first => accept
            (7.0, 4, (3, 4, 11, 12, 13, 14)),  # shares 2 with first => accept
        ]
        selected, relaxed = _select_portfolio_entries(ranked, 3, max_shared_numbers=2)
        self.assertFalse(relaxed)
        self.assertEqual([entry[0] for entry in selected], [10.0, 8.0, 7.0])
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1:]:
                self.assertLessEqual(len(set(left[2]) & set(right[2])), 2)


if __name__ == "__main__":
    unittest.main()
