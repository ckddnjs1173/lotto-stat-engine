import math
import unittest

from lotto_engine.candidates import iter_all_combinations
from lotto_engine.config import BASE_WEIGHTS
from lotto_engine.features import extract_features, pattern_type
from lotto_engine.loader import load_lotto_data
from lotto_engine.profiles import build_profile
from lotto_engine.scoring import SCORE_WEIGHTS, score_candidate


class MixedStructureV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = build_profile(load_lotto_data())

    def test_pattern_features_include_requested_dimensions(self):
        features = extract_features([1, 2, 3, 11, 22, 45], previous_draw=[1, 7, 15, 31, 35, 43])
        for key in (
            "odd_even_extreme", "sum_low", "sum_high", "sum_extreme",
            "range_narrow", "range_wide", "max_gap_extreme",
            "consecutive_pairs", "ending_duplicate_count", "empty_section_count",
            "prime_count", "previous_draw_proximity",
        ):
            self.assertIn(key, features)
        self.assertEqual(features["previous_draw_proximity"], 1)

    def test_score_formula_and_breakdown(self):
        result = score_candidate([1, 15, 19, 31, 35, 43], self.profile, BASE_WEIGHTS)
        calculated = sum(
            SCORE_WEIGHTS[key] * result["score_breakdown"][key] for key in SCORE_WEIGHTS
        )
        self.assertTrue(math.isclose(result["prediction_score"], calculated, abs_tol=0.0001))
        self.assertIn(result["pattern_type"], {"normal", "mixed", "outlier"})
        self.assertEqual(SCORE_WEIGHTS["historical_pattern_score"], 0.0)
        self.assertEqual(SCORE_WEIGHTS["number_dynamics_score"], 0.0)

    def test_components_discriminate_within_the_same_pattern_type(self):
        first = score_candidate([1, 8, 15, 22, 29, 36], self.profile, BASE_WEIGHTS)
        second = score_candidate([4, 9, 17, 25, 34, 42], self.profile, BASE_WEIGHTS)
        self.assertEqual(first["pattern_type"], second["pattern_type"])
        for key in ("outlier_survival_score", "type_balance_score", "transition_score"):
            self.assertNotEqual(
                first["score_breakdown"][key], second["score_breakdown"][key], key
            )

    def test_profiles_cover_patterns_transitions_and_dynamics(self):
        self.assertEqual(set(self.profile["pattern_type_probs"]), {"normal", "mixed", "outlier"})
        self.assertEqual(set(self.profile["number_dynamics"]), set(range(1, 46)))
        self.assertIn("100", self.profile["recent_distributions"])
        self.assertIn("300", self.profile["recent_distributions"])

    def test_no_valid_structure_is_hard_filtered(self):
        from lotto_engine.filters import passes_hard_filter
        self.assertTrue(passes_hard_filter([1, 2, 3, 4, 5, 6]))
        self.assertTrue(passes_hard_filter([40, 41, 42, 43, 44, 45]))

    def test_exhaustive_iterator_boundaries(self):
        iterator = iter_all_combinations()
        self.assertEqual(next(iterator), [1, 2, 3, 4, 5, 6])
        self.assertEqual(math.comb(45, 6), 8_145_060)


if __name__ == "__main__":
    unittest.main()
