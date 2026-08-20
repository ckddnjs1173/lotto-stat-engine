import unittest

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v28_reverse_ranking import FEATURE_NAMES as BASE_FEATURE_NAMES
from lotto_engine.v29_quadratic_reverse_ranking import (
    FEATURE_NAMES,
    quadratic_basis_from_base,
    run_v29_quadratic_reverse_ranking_audit,
)


def _synthetic_df(draws: int) -> pd.DataFrame:
    rows = []
    for index in range(draws):
        numbers = sorted((((index * 5 + step * 7) % 45) + 1) for step in range(6))
        if len(set(numbers)) != 6:
            numbers = sorted((((index * 11 + step * 8) % 45) + 1) for step in range(6))
        row = {ROUND_COLUMN: index + 1}
        row.update({column: numbers[pos] for pos, column in enumerate(NUMBER_COLUMNS)})
        rows.append(row)
    return pd.DataFrame(rows)


class V29QuadraticReverseRankingTests(unittest.TestCase):
    def test_quadratic_basis_has_complete_fixed_width(self):
        base = np.arange(1, len(BASE_FEATURE_NAMES) + 1, dtype=float)
        expanded = quadratic_basis_from_base(base)
        self.assertEqual(len(BASE_FEATURE_NAMES), 10)
        self.assertEqual(len(FEATURE_NAMES), 65)
        self.assertEqual(expanded.shape, (65,))
        np.testing.assert_allclose(expanded[:10], base)
        np.testing.assert_allclose(expanded[10:20], base * base)

    def test_quadratic_interaction_order_is_deterministic(self):
        base = np.zeros(10, dtype=float)
        base[0] = 2.0
        base[1] = 3.0
        expanded = quadratic_basis_from_base(base)
        first_interaction_index = 20
        self.assertEqual(
            FEATURE_NAMES[first_interaction_index],
            f"interaction:{BASE_FEATURE_NAMES[0]}*{BASE_FEATURE_NAMES[1]}",
        )
        self.assertAlmostEqual(expanded[first_interaction_index], 6.0)

    def test_outer_evaluation_starts_after_same_meta_training_count(self):
        result = run_v29_quadratic_reverse_ranking_audit(
            _synthetic_df(18),
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
        base_result = run_v29_quadratic_reverse_ranking_audit(base, **kwargs)
        extended_result = run_v29_quadratic_reverse_ranking_audit(extended, **kwargs)
        base_rows = {row["target_round"]: row for row in base_result["rows"]}
        extended_rows = {row["target_round"]: row for row in extended_result["rows"]}
        for target_round, row in base_rows.items():
            self.assertIn(target_round, extended_rows)
            other = extended_rows[target_round]
            self.assertAlmostEqual(row["percentile"], other["percentile"])
            self.assertAlmostEqual(row["actual_score"], other["actual_score"])
            self.assertEqual(row["prior_solved_targets"], other["prior_solved_targets"])

    def test_result_declares_exploratory_status_and_no_pattern_branch(self):
        result = run_v29_quadratic_reverse_ranking_audit(
            _synthetic_df(12),
            start_index=5,
            min_meta_train_targets=2,
            train_negatives_per_target=3,
            baseline_samples=6,
            bootstrap_reps=20,
        )
        self.assertEqual(result["basis_width"], 65)
        self.assertEqual(result["notes"]["pattern_type"], "not used")
        self.assertTrue(result["strict_nested_walk_forward"])
        self.assertTrue(result["notes"]["same_sampling_as_v28"])
        self.assertEqual(
            result["research_status"],
            "exploratory_screen_after_v28_failure",
        )


if __name__ == "__main__":
    unittest.main()
