from __future__ import annotations

import unittest

from lotto_engine.v31_freeze_candidate_audit import (
    CANDIDATE_FEATURE_SETS,
    FINALIST_NAMES,
    sample_nested_training_negatives,
)
from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC


class V31FreezeCandidateAuditTests(unittest.TestCase):
    def test_predeclared_candidates_are_nested(self):
        names = (
            "exact_full11",
            "exact_no_pair9",
            "exact_no_pair_sum8",
            "exact_no_pair_sum_odd7",
        )
        self.assertEqual(tuple(CANDIDATE_FEATURE_SETS), names)
        widths = [len(CANDIDATE_FEATURE_SETS[name]) for name in names]
        self.assertEqual(widths, [11, 9, 8, 7])
        for left, right in zip(names, names[1:]):
            self.assertTrue(
                set(CANDIDATE_FEATURE_SETS[right]).issubset(
                    CANDIDATE_FEATURE_SETS[left]
                )
            )
        self.assertEqual(
            FINALIST_NAMES,
            ("exact_no_pair_sum8", "exact_no_pair_sum_odd7"),
        )

    def test_nested_negative_stream_preserves_prefix(self):
        actual = (1, 2, 3, 4, 5, 6)
        short = sample_nested_training_negatives(
            1238, 64, actual, FULL11_BASELINE_SPEC
        )
        long = sample_nested_training_negatives(
            1238, 256, actual, FULL11_BASELINE_SPEC
        )
        self.assertEqual(short, long[:64])
        self.assertEqual(len(long), len(set(long)))
        self.assertNotIn(actual, long)

    def test_nested_negative_stream_is_deterministic(self):
        actual = (10, 20, 23, 34, 37, 40)
        left = sample_nested_training_negatives(
            1238, 128, actual, FULL11_BASELINE_SPEC, stream_delta=1
        )
        right = sample_nested_training_negatives(
            1238, 128, actual, FULL11_BASELINE_SPEC, stream_delta=1
        )
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
