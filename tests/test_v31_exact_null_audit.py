import unittest

import numpy as np

from lotto_engine.v31_core import FEATURE_NAMES
from lotto_engine.v31_exact_null_audit import hybrid_exact_vector, sampled_vector
from lotto_engine.v31_exact_structural_null import (
    EXACT_STRUCTURAL_FEATURES,
    exact_distribution_for_feature,
    exact_midrank_z,
)


class V31ExactNullAuditTests(unittest.TestCase):
    def test_hybrid_vector_replaces_only_structural_coordinates(self):
        context = {
            "number_lifts": np.zeros(46, dtype=float),
            "pair_lifts": np.zeros(990, dtype=float),
            "recent20_number_rate": np.full(46, 6.0 / 45.0),
            "recent100_number_rate": np.full(46, 6.0 / 45.0),
            "recent100_pair_rate": np.full(990, 1.0 / 66.0),
            "previous_draw": frozenset({1, 2, 3, 4, 5, 6}),
        }
        reference = np.zeros((8, len(FEATURE_NAMES)), dtype=float)
        numbers = (1, 7, 13, 19, 25, 31)
        sampled = sampled_vector(numbers, context, reference)
        hybrid = hybrid_exact_vector(numbers, context, reference)

        structural = set(EXACT_STRUCTURAL_FEATURES)
        for index, feature in enumerate(FEATURE_NAMES):
            if feature in structural:
                raw_value = {
                    "previous_draw_overlap": 1,
                    "sum_signed_center_138": sum(numbers) - 138,
                    "high_minus_low_zone_count": 1 - 3,
                    "odd_count_signed_center_3": sum(n % 2 for n in numbers) - 3,
                    "number_range": numbers[-1] - numbers[0],
                    "consecutive_pairs": 0,
                }[feature]
                expected = exact_midrank_z(
                    raw_value, exact_distribution_for_feature(feature)
                )
                self.assertAlmostEqual(hybrid[index], expected)
            else:
                self.assertEqual(hybrid[index], sampled[index])


if __name__ == "__main__":
    unittest.main()
