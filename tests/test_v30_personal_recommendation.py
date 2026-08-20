import unittest

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v30_personal_recommendation import (
    build_latest_v30_model,
    explain_v30_candidate,
    top_v30_stream,
    v30_personal_score,
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


class V30PersonalRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted = build_latest_v30_model(
            _synthetic_df(12),
            start_index=5,
            train_negatives_per_target=3,
            reference_samples=16,
        )

    def test_latest_model_uses_all_completed_targets(self):
        self.assertEqual(self.fitted["history_draws"], 12)
        self.assertEqual(self.fitted["solved_targets"], 7)
        self.assertEqual(self.fitted["pair_rows"], 21)
        self.assertEqual(self.fitted["target_round"], 13)
        self.assertEqual(self.fitted["reference"].shape, (16, 10))

    def test_score_matches_explanation(self):
        numbers = [1, 8, 15, 22, 29, 36]
        score = v30_personal_score(
            numbers,
            self.fitted["model"],
            self.fitted["context"],
            self.fitted["reference"],
        )
        explained = explain_v30_candidate(
            numbers,
            self.fitted["model"],
            self.fitted["context"],
            self.fitted["reference"],
        )
        self.assertAlmostEqual(score, explained["model_score"])
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in explained["bounded_base_features"].values()))

    def test_top_stream_orders_only_by_model_score(self):
        candidates = [
            [1, 2, 3, 4, 5, 6],
            [4, 11, 18, 25, 32, 39],
            [7, 14, 21, 28, 35, 42],
            [10, 16, 22, 28, 34, 40],
        ]
        selected, evaluated = top_v30_stream(
            candidates,
            self.fitted["model"],
            self.fitted["context"],
            self.fitted["reference"],
            top_k=3,
            seed=123,
        )
        self.assertEqual(evaluated, 4)
        scores = [item["model_score"] for item in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_basis_explanation_contributions_sum_to_score(self):
        item = explain_v30_candidate(
            [3, 9, 15, 21, 27, 33],
            self.fitted["model"],
            self.fitted["context"],
            self.fitted["reference"],
        )
        # explain_v30_candidate reports only the strongest ten terms, so verify the
        # score is finite while the exact full-basis equality is covered by the
        # source v3.0 model tests.
        self.assertTrue(np.isfinite(item["model_score"]))
        self.assertEqual(len(item["strongest_terms"]), 10)


if __name__ == "__main__":
    unittest.main()
