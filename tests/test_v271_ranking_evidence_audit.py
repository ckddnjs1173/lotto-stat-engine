import random
import unittest

import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v271_ranking_evidence_audit import (
    _raw_scores,
    _sample_fair_candidates,
    ranking_reliability_from_summary,
    run_v271_ranking_evidence_audit,
)


class V271RankingEvidenceAuditTests(unittest.TestCase):
    @staticmethod
    def _synthetic_draws(count: int) -> pd.DataFrame:
        rows = []
        for round_no in range(1, count + 1):
            start = (round_no * 7) % 45
            numbers = sorted(((start + offset) % 45) + 1 for offset in range(6))
            row = {ROUND_COLUMN: round_no}
            row.update({column: number for column, number in zip(NUMBER_COLUMNS, numbers)})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_fair_sampling_is_deterministic_unique_and_excludes_actual(self):
        actual = (1, 2, 3, 4, 5, 6)
        left = _sample_fair_candidates(random.Random(123), 50, actual)
        right = _sample_fair_candidates(random.Random(123), 50, actual)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 50)
        self.assertEqual(len(set(left)), 50)
        self.assertNotIn(actual, left)

    def test_equal_family_fusion_is_predeclared_half_and_half(self):
        number_lifts = [0.0] * 46
        pair_lifts = [0.0] * 990
        for number in range(1, 7):
            number_lifts[number] = 0.2
        from lotto_engine.pair_evidence import PAIR_TO_INDEX
        from itertools import combinations
        for pair in combinations((1, 2, 3, 4, 5, 6), 2):
            pair_lifts[PAIR_TO_INDEX[pair]] = 0.4
        scores = _raw_scores((1, 2, 3, 4, 5, 6), number_lifts, pair_lifts)
        self.assertAlmostEqual(scores["number"], 0.2)
        self.assertAlmostEqual(scores["pair"], 0.4)
        self.assertAlmostEqual(scores["equal_family_fusion"], 0.3)

    def test_rank_reliability_is_zero_when_all_windows_are_below_random(self):
        summary = {
            "mean_percentile": 49.0,
            "recent_300_mean_percentile": 48.0,
            "recent_100_mean_percentile": 47.0,
        }
        self.assertEqual(ranking_reliability_from_summary(summary), 0.0)

    def test_future_append_does_not_change_existing_target_rank(self):
        first = self._synthetic_draws(101)
        extended = self._synthetic_draws(102)
        left = run_v271_ranking_evidence_audit(
            first,
            start_index=100,
            baseline_samples=20,
            bootstrap_reps=20,
            include_rows=True,
        )
        right = run_v271_ranking_evidence_audit(
            extended,
            start_index=100,
            baseline_samples=20,
            bootstrap_reps=20,
            include_rows=True,
        )
        self.assertEqual(left["rows"][0], right["rows"][0])


if __name__ == "__main__":
    unittest.main()
