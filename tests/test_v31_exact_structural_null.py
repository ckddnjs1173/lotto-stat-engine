import unittest

from lotto_engine.candidates import TOTAL_COMBINATION_COUNT
from lotto_engine.v31_exact_structural_null import (
    consecutive_pairs_distribution,
    exact_distribution_for_feature,
    exact_midrank_z,
    high_minus_low_distribution,
    number_range_distribution,
    odd_count_distribution,
    previous_draw_overlap_distribution,
    sum_distribution,
)


class V31ExactStructuralNullTests(unittest.TestCase):
    def test_every_exact_distribution_totals_full_combination_space(self):
        distributions = (
            previous_draw_overlap_distribution(),
            odd_count_distribution(),
            high_minus_low_distribution(),
            number_range_distribution(),
            consecutive_pairs_distribution(),
            sum_distribution(),
        )
        for counts in distributions:
            self.assertEqual(sum(counts.values()), TOTAL_COMBINATION_COUNT)

    def test_previous_draw_overlap_counts_match_hypergeometric(self):
        counts = previous_draw_overlap_distribution()
        self.assertEqual(counts[0], 3_262_623)
        self.assertEqual(counts[1], 3_454_542)
        self.assertEqual(counts[2], 1_233_765)
        self.assertEqual(counts[6], 1)

    def test_consecutive_pair_counts_match_run_combinatorics(self):
        counts = consecutive_pairs_distribution()
        self.assertEqual(
            counts,
            {
                0: 3_838_380,
                1: 3_290_040,
                2: 913_900,
                3: 98_800,
                4: 3_900,
                5: 40,
            },
        )

    def test_range_distribution_has_valid_support(self):
        counts = number_range_distribution()
        self.assertEqual(min(counts), 5)
        self.assertEqual(max(counts), 44)
        self.assertEqual(counts[5], 40)

    def test_sum_distribution_has_valid_support_and_symmetry(self):
        counts = sum_distribution()
        self.assertEqual(min(counts), 21)
        self.assertEqual(max(counts), 255)
        for total, count in counts.items():
            self.assertEqual(count, counts[276 - total])

    def test_exact_midrank_known_overlap_two_value(self):
        z = exact_midrank_z(2, previous_draw_overlap_distribution())
        self.assertAlmostEqual(z, 0.8008580622929704)

    def test_center_offsets_are_only_coordinate_labels(self):
        sum_counts = exact_distribution_for_feature("sum_signed_center_138")
        odd_counts = exact_distribution_for_feature("odd_count_signed_center_3")
        self.assertEqual(sum_counts[-117], sum_distribution()[21])
        self.assertEqual(sum_counts[117], sum_distribution()[255])
        self.assertEqual(odd_counts[-3], odd_count_distribution()[0])
        self.assertEqual(odd_counts[3], odd_count_distribution()[6])


if __name__ == "__main__":
    unittest.main()
