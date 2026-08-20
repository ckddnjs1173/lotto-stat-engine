import unittest

import pandas as pd

from lotto_engine.number_evidence import (
    UNIFORM_NUMBER_PROBABILITY,
    number_posterior_probabilities,
    run_number_evidence_backtest,
)


def make_df(draws):
    rows = []
    for round_no, numbers in enumerate(draws, 1):
        rows.append({
            "회차": round_no,
            **{f"번호{index}": number for index, number in enumerate(numbers, 1)},
        })
    return pd.DataFrame(rows)


class NumberEvidenceTests(unittest.TestCase):
    def test_posterior_probabilities_sum_to_six(self):
        df = make_df([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ])
        probabilities = number_posterior_probabilities(df, prior_strength=24.0)
        self.assertAlmostEqual(sum(probabilities.values()), 6.0, places=12)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities.values()))

    def test_stronger_prior_shrinks_toward_uniform(self):
        df = make_df([[1, 2, 3, 4, 5, 6] for _ in range(20)])
        weak = number_posterior_probabilities(df, prior_strength=1.0)
        strong = number_posterior_probabilities(df, prior_strength=200.0)
        self.assertGreater(weak[1], strong[1])
        self.assertLess(abs(strong[1] - UNIFORM_NUMBER_PROBABILITY), abs(weak[1] - UNIFORM_NUMBER_PROBABILITY))
        self.assertLess(weak[45], strong[45])

    def test_future_append_does_not_change_existing_walk_forward_rows(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
            [37, 38, 39, 40, 41, 42],
            [1, 8, 15, 22, 29, 36],
        ]
        spec = ({"name": "test", "prior_strength": 24.0, "half_life": 25.0},)
        before = run_number_evidence_backtest(
            make_df(draws[:-1]), start_index=4, specs=spec, include_rows=True
        )
        after = run_number_evidence_backtest(
            make_df(draws), start_index=4, specs=spec, include_rows=True
        )
        self.assertEqual(before["rows"], after["rows"][:-1])

    def test_uniform_baseline_is_constant_for_six_of_45(self):
        df = make_df([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
        ])
        result = run_number_evidence_backtest(
            df,
            start_index=2,
            specs=({"name": "full", "prior_strength": 24.0, "half_life": None},),
        )
        expected = UNIFORM_NUMBER_PROBABILITY * (1.0 - UNIFORM_NUMBER_PROBABILITY)
        self.assertAlmostEqual(result["baseline"]["mean_brier"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
