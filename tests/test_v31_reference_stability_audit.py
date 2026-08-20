import unittest

import numpy as np

from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC
from lotto_engine.v31_reference_stability_audit import (
    _average_ranks,
    _compare_scores,
    sample_nested_reference_candidates,
)


class V31ReferenceStabilityAuditTests(unittest.TestCase):
    def test_nested_reference_stream_preserves_prefix_when_count_grows(self):
        small = sample_nested_reference_candidates(
            round_no=1238,
            count=32,
            spec=FULL11_BASELINE_SPEC,
        )
        large = sample_nested_reference_candidates(
            round_no=1238,
            count=128,
            spec=FULL11_BASELINE_SPEC,
        )
        self.assertEqual(small, large[: len(small)])
        self.assertEqual(len(set(large)), len(large))

    def test_average_ranks_are_tie_safe(self):
        ranks = _average_ranks(np.asarray([5.0, 1.0, 1.0, 3.0]))
        np.testing.assert_allclose(ranks, np.asarray([3.0, 0.5, 0.5, 2.0]))

    def test_identical_scores_have_perfect_stability(self):
        scores = np.asarray([0.1, 0.4, 0.2, 0.2, 0.9])
        result = _compare_scores(scores, scores.copy())
        self.assertAlmostEqual(result["pearson_score_correlation"], 1.0)
        self.assertAlmostEqual(result["spearman_rank_correlation"], 1.0)
        self.assertAlmostEqual(result["mean_absolute_score_delta"], 0.0)
        self.assertAlmostEqual(result["max_absolute_score_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
