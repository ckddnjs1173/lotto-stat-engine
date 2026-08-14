import unittest

import numpy as np
import pandas as pd

from lotto_engine.pair_evidence import (
    PAIR_COUNT,
    PAIRS_PER_DRAW,
    UNIFORM_PAIR_PROBABILITY,
    pair_posterior_probabilities,
    run_pair_evidence_backtest,
)


def make_df(rows: list[tuple[int, list[int]]]) -> pd.DataFrame:
    data = []
    for draw, numbers in rows:
        data.append({
            "회차": draw,
            **{f"번호{index + 1}": number for index, number in enumerate(sorted(numbers))},
        })
    return pd.DataFrame(data)


class PairEvidenceTests(unittest.TestCase):
    def test_uniform_pair_probability_is_one_over_66(self):
        self.assertEqual(PAIR_COUNT, 990)
        self.assertEqual(PAIRS_PER_DRAW, 15)
        self.assertAlmostEqual(UNIFORM_PAIR_PROBABILITY, 1.0 / 66.0, places=15)

    def test_posterior_probabilities_sum_to_fifteen(self):
        df = make_df([
            (1, [1, 2, 3, 4, 5, 6]),
            (2, [7, 8, 9, 10, 11, 12]),
            (3, [1, 8, 15, 22, 29, 36]),
        ])
        probabilities = pair_posterior_probabilities(df, prior_strength=66.0)
        self.assertEqual(len(probabilities), PAIR_COUNT)
        self.assertAlmostEqual(sum(probabilities.values()), 15.0, places=10)

    def test_stronger_prior_shrinks_toward_uniform(self):
        df = make_df([
            (draw, [1, 2, 3, 4, 5, 6]) for draw in range(1, 11)
        ])
        weak = pair_posterior_probabilities(df, prior_strength=1.0)
        strong = pair_posterior_probabilities(df, prior_strength=330.0)
        target = UNIFORM_PAIR_PROBABILITY
        self.assertGreater(abs(weak[(1, 2)] - target), abs(strong[(1, 2)] - target))

    def test_future_append_does_not_change_existing_walk_forward_rows(self):
        rows = []
        for draw in range(1, 9):
            start = ((draw - 1) * 5) % 40 + 1
            numbers = [((start + offset - 1) % 45) + 1 for offset in range(6)]
            rows.append((draw, numbers))
        before_df = make_df(rows)
        after_df = make_df(rows + [(9, [2, 9, 16, 23, 30, 37])])
        spec = ({"name": "test", "prior_strength": 66.0, "half_life": 25.0},)
        before = run_pair_evidence_backtest(before_df, start_index=3, specs=spec, include_rows=True)
        after = run_pair_evidence_backtest(after_df, start_index=3, specs=spec, include_rows=True)
        self.assertEqual(before["rows"], after["rows"][: len(before["rows"])])

    def test_uniform_baseline_is_constant_for_fifteen_of_990(self):
        q = UNIFORM_PAIR_PROBABILITY
        expected_brier = q * (1.0 - q)
        actual = np.zeros(PAIR_COUNT, dtype=float)
        actual[:PAIRS_PER_DRAW] = 1.0
        probabilities = np.full(PAIR_COUNT, q, dtype=float)
        observed_brier = float(np.mean((probabilities - actual) ** 2))
        self.assertAlmostEqual(observed_brier, expected_brier, places=15)


if __name__ == "__main__":
    unittest.main()
