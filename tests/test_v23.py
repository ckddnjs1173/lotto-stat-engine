import math
import random
import unittest

from lotto_engine.mixed_backtest import tie_safe_percentile
from lotto_engine.mixed_scoring import (
    INTERACTIONS,
    MIXED_SCORE_WEIGHTS,
    SINGLE_FEATURES,
    _accumulate,
    _empty_distribution,
    build_mixed_profile,
    score_mixed_candidate,
    structure_record,
)
from lotto_engine.recommender import _mixed_diagnostics, _select_diverse_mixed


class MixedSlotV23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(23)
        cls.records = [
            structure_record(sorted(rng.sample(range(1, 46), 6)))
            for _ in range(1000)
        ]
        cls.baseline = _empty_distribution()
        for record in cls.records:
            _accumulate(cls.baseline, record)
        mixed = [record for record in cls.records if record["pattern_type"] == "mixed"]
        cls.profile = build_mixed_profile(mixed, cls.baseline)

    def test_structure_record_has_required_dimensions(self):
        record = structure_record([1, 2, 11, 22, 33, 45], "outlier")
        for key in (
            *SINGLE_FEATURES,
            "normal_backbone_bin",
            "pattern_type",
            "previous_pattern_type",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["previous_pattern_type"], "outlier")

    def test_mixed_score_formula_and_output(self):
        result = score_mixed_candidate([1, 4, 8, 12, 16, 18], self.profile)
        calculated = sum(
            MIXED_SCORE_WEIGHTS[key] * result[key] for key in MIXED_SCORE_WEIGHTS
        )
        self.assertTrue(
            math.isclose(result["mixed_slot_score"], calculated, abs_tol=0.0001)
        )
        for key in (
            "numbers",
            "mixed_lift_score",
            "mixed_interaction_score",
            "normal_backbone_score",
            "controlled_extreme_score",
            "recency_consistency_score",
            "extreme_count",
            "extreme_signature",
            "pattern_type",
        ):
            self.assertIn(key, result)

    def test_interaction_set_matches_v23_specification(self):
        self.assertEqual(len(INTERACTIONS), 6)
        self.assertIn(("extreme_count", "normal_backbone_bin"), INTERACTIONS)

    def test_tie_safe_percentile_uses_half_credit_for_equal(self):
        self.assertEqual(tie_safe_percentile(5.0, [4.0, 5.0, 5.0, 6.0]), 50.0)

    def test_component_scores_preserve_same_signature_variation(self):
        first = score_mixed_candidate([1, 2, 8, 41, 43, 45], self.profile)
        second = score_mixed_candidate([1, 5, 9, 41, 43, 45], self.profile)
        self.assertNotEqual(first["normal_backbone_score"], second["normal_backbone_score"])
        self.assertNotEqual(first["controlled_extreme_score"], second["controlled_extreme_score"])

    def test_mixed_diversity_is_soft_and_reports_diagnostics(self):
        candidates = []
        for score, numbers in ((99, [1, 2, 8, 41, 43, 45]), (98, [1, 5, 9, 41, 43, 45]), (97, [2, 12, 22, 32, 42, 44])):
            item = score_mixed_candidate(numbers, self.profile)
            item["mixed_slot_score"] = score
            candidates.append(item)
        selected = _select_diverse_mixed(candidates, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["mixed_slot_score"], 99)
        self.assertIn("mixed_diversity_penalty", selected[1])
        diagnostics = _mixed_diagnostics(selected)
        expected = any(item.get("mixed_diversity_penalty", 0) > 0 for item in selected)
        self.assertEqual(diagnostics["mixed_diversity_penalty_applied"], expected)


if __name__ == "__main__":
    unittest.main()
