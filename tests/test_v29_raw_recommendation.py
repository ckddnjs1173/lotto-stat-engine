import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v29_quadratic_reverse_ranking import (
    quadratic_candidate_vector,
    score_quadratic_vector,
)
from lotto_engine.v29_raw_recommendation import (
    build_latest_v29_model,
    explain_v29_candidate,
    load_validation_diagnostics,
    raw_v29_score,
    top_raw_v29_stream,
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


class V29RawRecommendationTests(unittest.TestCase):
    def test_latest_model_uses_all_completed_solved_targets(self):
        fitted = build_latest_v29_model(
            _synthetic_df(18),
            start_index=5,
            train_negatives_per_target=4,
        )
        self.assertEqual(fitted["history_draws"], 18)
        self.assertEqual(fitted["solved_targets"], 13)
        self.assertEqual(fitted["pair_rows"], 52)
        self.assertEqual(fitted["model"]["solved_targets"], 13)

    def test_raw_score_matches_frozen_v29_equation(self):
        fitted = build_latest_v29_model(
            _synthetic_df(14),
            start_index=5,
            train_negatives_per_target=3,
        )
        candidate = (1, 8, 15, 22, 29, 36)
        direct_vector = quadratic_candidate_vector(candidate, fitted["context"])
        direct = score_quadratic_vector(direct_vector, fitted["model"])
        wrapped = raw_v29_score(candidate, fitted["model"], fitted["context"])
        self.assertAlmostEqual(direct, wrapped)

    def test_explanation_sums_to_raw_score(self):
        fitted = build_latest_v29_model(
            _synthetic_df(14),
            start_index=5,
            train_negatives_per_target=3,
        )
        candidate = (2, 9, 16, 23, 30, 37)
        explained = explain_v29_candidate(candidate, fitted["model"], fitted["context"])
        expected = raw_v29_score(candidate, fitted["model"], fitted["context"])
        self.assertAlmostEqual(explained["raw_model_score"], expected)
        self.assertEqual(len(explained["base_features"]), 10)
        self.assertEqual(len(explained["strongest_quadratic_terms"]), 10)

    def test_top_stream_orders_only_by_raw_model_score(self):
        fitted = build_latest_v29_model(
            _synthetic_df(14),
            start_index=5,
            train_negatives_per_target=3,
        )
        candidates = [
            (1, 8, 15, 22, 29, 36),
            (2, 9, 16, 23, 30, 37),
            (3, 10, 17, 24, 31, 38),
            (4, 11, 18, 25, 32, 39),
        ]
        selected, evaluated = top_raw_v29_stream(
            candidates,
            fitted["model"],
            fitted["context"],
            top_k=3,
            seed=123,
        )
        self.assertEqual(evaluated, 4)
        scores = [item["raw_model_score"] for item in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_validation_file_is_diagnostic_only(self):
        payload = {
            "version": "test-v29",
            "outer_summary": {
                "mean_percentile": 49.1,
                "recent_300_mean_percentile": 47.2,
                "recent_100_mean_percentile": 47.4,
                "historical_gate_pass": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            diagnostics = load_validation_diagnostics(path)
        self.assertEqual(diagnostics["status"], "loaded")
        self.assertFalse(diagnostics["historical_gate_pass"])
        self.assertFalse(diagnostics["ranking_influenced"])
        self.assertEqual(diagnostics["role"], "diagnostic_only_not_used_in_ranking")


if __name__ == "__main__":
    unittest.main()
