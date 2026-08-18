import unittest

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v30_null_bounded_reverse_ranking import (
    FEATURE_NAMES,
    bounded_quadratic_basis,
    null_midrank_transform,
    run_v30_null_bounded_reverse_ranking_audit,
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


class V30NullBoundedReverseRankingTests(unittest.TestCase):
    def test_midrank_transform_is_bounded_and_tie_safe(self):
        reference = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        raw = np.full(10, 5.0, dtype=float)
        transformed = null_midrank_transform(raw, reference)
        self.assertTrue(np.all(transformed >= -1.0))
        self.assertTrue(np.all(transformed <= 1.0))
        # Five values are lower and one is tied: midrank CDF=(5+6)/(2*10)=0.55.
        np.testing.assert_allclose(transformed, np.full(10, 0.10))

    def test_values_outside_reference_support_saturate(self):
        reference = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        high = null_midrank_transform(np.full(10, 100.0), reference)
        low = null_midrank_transform(np.full(10, -100.0), reference)
        np.testing.assert_allclose(high, np.ones(10))
        np.testing.assert_allclose(low, -np.ones(10))

    def test_quadratic_basis_cannot_exceed_unit_magnitude(self):
        bounded = np.linspace(-1.0, 1.0, 10)
        basis = bounded_quadratic_basis(bounded)
        self.assertEqual(len(FEATURE_NAMES), 65)
        self.assertEqual(basis.shape, (65,))
        self.assertLessEqual(float(np.max(np.abs(basis))), 1.0)

    def test_future_append_does_not_change_existing_outer_targets(self):
        base = _synthetic_df(16)
        extended = _synthetic_df(17)
        kwargs = dict(
            start_index=5,
            min_meta_train_targets=3,
            train_negatives_per_target=4,
            baseline_samples=8,
            reference_samples=16,
            bootstrap_reps=20,
            include_rows=True,
        )
        base_result = run_v30_null_bounded_reverse_ranking_audit(base, **kwargs)
        extended_result = run_v30_null_bounded_reverse_ranking_audit(extended, **kwargs)
        base_rows = {row["target_round"]: row for row in base_result["rows"]}
        extended_rows = {row["target_round"]: row for row in extended_result["rows"]}
        for target_round, row in base_rows.items():
            self.assertIn(target_round, extended_rows)
            other = extended_rows[target_round]
            self.assertAlmostEqual(row["percentile"], other["percentile"])
            self.assertAlmostEqual(row["actual_score"], other["actual_score"])
            np.testing.assert_allclose(
                row["actual_bounded_features"], other["actual_bounded_features"]
            )

    def test_result_declares_boundary_correction_and_no_pattern_branch(self):
        result = run_v30_null_bounded_reverse_ranking_audit(
            _synthetic_df(13),
            start_index=5,
            min_meta_train_targets=2,
            train_negatives_per_target=3,
            baseline_samples=6,
            reference_samples=12,
            bootstrap_reps=20,
        )
        self.assertEqual(result["basis_width"], 65)
        self.assertEqual(result["notes"]["pattern_type"], "not used")
        self.assertTrue(result["strict_nested_walk_forward"])
        self.assertIn("[-1,1]", result["notes"]["quadratic_bound"])
        self.assertEqual(
            result["research_status"],
            "exploratory_boundary_correction_after_v29_diagnostic",
        )


if __name__ == "__main__":
    unittest.main()
