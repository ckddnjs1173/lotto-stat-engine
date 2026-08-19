import unittest

from lotto_engine.v31_freeze_candidate_audit import sample_nested_training_negatives as audit_negatives
from lotto_engine.v31_frozen_core import (
    sample_nested_reference_candidates as frozen_reference_candidates,
    sample_nested_training_negatives as frozen_negatives,
)
from lotto_engine.v31_model_spec import FROZEN7_SPEC
from lotto_engine.v31_reference_stability_audit import sample_nested_reference_candidates as audit_reference_candidates


EXPECTED_ACTIVE = (
    "number_full_log_lift",
    "number_recent20_excess",
    "number_recent100_excess",
    "previous_draw_overlap",
    "high_minus_low_zone_count",
    "number_range",
    "consecutive_pairs",
)


class V31FrozenModelTests(unittest.TestCase):
    def test_frozen7_identity_is_exactly_audited_choice(self):
        self.assertEqual(FROZEN7_SPEC.active_features, EXPECTED_ACTIVE)
        self.assertEqual(FROZEN7_SPEC.fair_reference_samples_per_target, 4096)
        self.assertEqual(FROZEN7_SPEC.train_negatives_per_target, 256)
        self.assertEqual(FROZEN7_SPEC.ridge_lambda, 2.0)
        self.assertEqual(FROZEN7_SPEC.reference_stream_delta, 0)
        self.assertEqual(FROZEN7_SPEC.negative_stream_delta, 0)
        self.assertEqual(
            FROZEN7_SPEC.structural_null_policy,
            "exact_whole_universe_midrank_for_structural_features",
        )

    def test_production_reference_stream_matches_audited_stream(self):
        expected = audit_reference_candidates(
            1238, 256, FROZEN7_SPEC, stream_delta=0
        )
        actual = frozen_reference_candidates(1238, 256, FROZEN7_SPEC)
        self.assertEqual(actual, expected)

    def test_production_negative_stream_matches_audited_stream(self):
        actual_draw = (10, 20, 23, 34, 37, 40)
        expected = audit_negatives(
            1237, 256, actual_draw, FROZEN7_SPEC, stream_delta=0
        )
        actual = frozen_negatives(1237, 256, actual_draw, FROZEN7_SPEC)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
