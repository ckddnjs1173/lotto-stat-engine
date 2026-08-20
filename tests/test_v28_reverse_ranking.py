import unittest

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v28_reverse_ranking import (
    FEATURE_NAMES,
    _add_solved_target,
    _empty_training_stats,
    fit_pairwise_ridge_ranker,
    run_v28_reverse_ranking_audit,
)


def _synthetic_df(draws: int) -> pd.DataFrame:
    rows = []
    for index in range(draws):
        start = (index * 7) % 45
        numbers = sorted((((start + step * 6) % 45) + 1) for step in range(6))
        # Extremely rare modular collisions are impossible here because gcd(6,45)=3,
        # so use a second deterministic construction if uniqueness is lost.
        if len(set(numbers)) != 6:
            numbers = sorted((((index * 5 + step * 7) % 45) + 1) for step in range(6))
        row = {ROUND_COLUMN: index + 1}
        row.update({column: numbers[pos] for pos, column in enumerate(NUMBER_COLUMNS)})
        rows.append(row)
    return pd.DataFrame(rows)


class V28ReverseRankingTests(unittest.TestCase):
    def test_pairwise_ridge_learns_direction_from_prior_differences(self):
        stats = _empty_training_stats()
        actual = np.zeros(len(FEATURE_NAMES), dtype=float)
        actual[0] = 2.0
        negatives = []
        for value in (0.0, 0.5, 1.0, 1.5):
            negative = np.zeros(len(FEATURE_NAMES), dtype=float)
            negative[0] = value
            negatives.append(negative)
        _add_solved_target(stats, actual, negatives)
        model = fit_pairwise_ridge_ranker(stats, ridge_lambda=2.0)
        self.assertGreater(model["effective_weights"][0], 0.0)
        self.assertEqual(model["solved_targets"], 1)
        self.assertEqual(model["pair_rows"], 4)

    def test_outer_evaluation_starts_only_after_meta_training_targets(self):
        df = _synthetic_df(18)
        result = run_v28_reverse_ranking_audit(
            df,
            start_index=5,
            min_meta_train_targets=3,
            train_negatives_per_target=4,
            baseline_samples=8,
            bootstrap_reps=20,
            include_rows=True,
        )
        self.assertEqual(result["outer_summary"]["tests"], 10)
        first = result["rows"][0]
        self.assertEqual(first["target_index"], 8)
        self.assertEqual(first["prior_solved_targets"], 3)
        self.assertEqual(first["prior_pair_rows"], 12)

    def test_future_append_does_not_change_existing_outer_target(self):
        base = _synthetic_df(18)
        extended = _synthetic_df(19)
        kwargs = dict(
            start_index=5,
            min_meta_train_targets=3,
            train_negatives_per_target=4,
            baseline_samples=8,
            bootstrap_reps=20,
            include_rows=True,
        )
        base_result = run_v28_reverse_ranking_audit(base, **kwargs)
        extended_result = run_v28_reverse_ranking_audit(extended, **kwargs)
        base_rows = {row["target_round"]: row for row in base_result["rows"]}
        extended_rows = {row["target_round"]: row for row in extended_result["rows"]}
        for target_round, row in base_rows.items():
            self.assertIn(target_round, extended_rows)
            other = extended_rows[target_round]
            self.assertAlmostEqual(row["percentile"], other["percentile"])
            self.assertAlmostEqual(row["actual_score"], other["actual_score"])
            self.assertEqual(row["prior_solved_targets"], other["prior_solved_targets"])

    def test_result_declares_no_pattern_type_branch(self):
        result = run_v28_reverse_ranking_audit(
            _synthetic_df(12),
            start_index=5,
            min_meta_train_targets=2,
            train_negatives_per_target=3,
            baseline_samples=6,
            bootstrap_reps=20,
        )
        self.assertEqual(result["notes"]["pattern_type"], "not used")
        self.assertTrue(result["strict_nested_walk_forward"])
        self.assertEqual(result["feature_names"], list(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
