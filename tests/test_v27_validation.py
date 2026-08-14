import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v27_validation import (
    MODEL_NAMES,
    _block_bootstrap_mean_ci,
    _compose,
    _renormalized_legacy_base,
    run_v27_score_audit,
    tie_safe_percentile,
)


class V27ValidationTests(unittest.TestCase):
    def test_tie_safe_percentile_uses_half_credit(self):
        self.assertAlmostEqual(tie_safe_percentile(2.0, [1.0, 2.0, 3.0]), 50.0)
        self.assertAlmostEqual(tie_safe_percentile(3.0, [1.0, 2.0, 3.0]), 83.3333333333)

    def test_primary_composition_matches_predeclared_formulas(self):
        scores = _compose(70.0, 80.0, 60.0)
        self.assertAlmostEqual(scores["base_only"], 70.0)
        self.assertAlmostEqual(scores["base_plus_transition"], 71.5)
        self.assertAlmostEqual(scores["base_plus_momentum"], 68.5)
        self.assertAlmostEqual(scores["v26_full"], 70.0)

    def test_legacy_base_ablation_renormalizes_remaining_terms(self):
        breakdown = {
            "outlier_survival_score": 80.0,
            "type_balance_score": 20.0,
            "transition_score": 40.0,
            "normal_structure_score": 60.0,
            "historical_pattern_score": 999.0,
            "number_dynamics_score": 999.0,
        }
        no_rarity = _renormalized_legacy_base(
            breakdown, frozenset({"type_balance_score"})
        )
        expected = (0.45 * 80.0 + 0.15 * 40.0 + 0.15 * 60.0) / 0.75
        self.assertAlmostEqual(no_rarity, expected)

        clean = _renormalized_legacy_base(
            breakdown, frozenset({"type_balance_score", "transition_score"})
        )
        expected_clean = (0.45 * 80.0 + 0.15 * 60.0) / 0.60
        self.assertAlmostEqual(clean, expected_clean)

    def test_block_bootstrap_is_deterministic(self):
        values = [float(index) for index in range(1, 41)]
        first = _block_bootstrap_mean_ci(values, reps=100, block_size=5, seed=123)
        second = _block_bootstrap_mean_ci(values, reps=100, block_size=5, seed=123)
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])

    def test_walk_forward_builds_profiles_from_past_rows_only(self):
        rows = []
        for draw in range(1, 6):
            row = {ROUND_COLUMN: draw}
            for index, column in enumerate(NUMBER_COLUMNS, 1):
                row[column] = index
            rows.append(row)
        df = pd.DataFrame(rows)
        observed_history_lengths = []

        def fake_build_profile(history):
            observed_history_lengths.append(len(history))
            return {"latest_pattern_type": "normal"}

        def fake_score(numbers, profile, mixed_profile, weights=None):
            value = float(sum(numbers))
            return {
                "pattern_type": "normal",
                "scores": {name: value for name in MODEL_NAMES},
                "components": {},
            }

        with (
            patch("lotto_engine.v27_validation.build_all_combination_baseline", return_value={}),
            patch("lotto_engine.v27_validation.build_profile", side_effect=fake_build_profile),
            patch("lotto_engine.v27_validation.build_draw_structure_records", return_value=[]),
            patch(
                "lotto_engine.v27_validation.build_mixed_profile",
                return_value={"dynamic_markov_decay": {}},
            ),
            patch("lotto_engine.v27_validation.score_v27_audit_candidate", side_effect=fake_score),
            patch("lotto_engine.v27_validation.random_combination", return_value=[1, 2, 3, 4, 5, 7]),
        ):
            payload = run_v27_score_audit(
                df,
                start_index=2,
                baseline_samples=1,
                bootstrap_reps=10,
            )

        self.assertEqual(observed_history_lengths, [2, 3, 4])
        self.assertEqual(payload["total_tests"], 3)
        self.assertTrue(payload["strict_walk_forward"])
        self.assertEqual(payload["baseline_samples_per_target"], 1)


if __name__ == "__main__":
    unittest.main()
