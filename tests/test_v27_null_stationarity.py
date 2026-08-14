import unittest

from lotto_engine.v27_null_stationarity import (
    holm_adjust,
    lag_acf,
    pattern_transition_mutual_information,
    permutation_series_screen,
)


class V27NullStationarityTests(unittest.TestCase):
    def test_lag_acf_detects_strong_alternation(self):
        values = [0.0, 1.0] * 50
        acf = lag_acf(values, max_lag=3)
        self.assertLess(acf[0], -0.9)
        self.assertGreater(acf[1], 0.9)

    def test_permutation_series_screen_is_deterministic(self):
        values = [float(index % 7) for index in range(120)]
        first = permutation_series_screen(
            values, reps=25, max_lag=4, recent_window=30, seed=123
        )
        second = permutation_series_screen(
            values, reps=25, max_lag=4, recent_window=30, seed=123
        )
        self.assertEqual(first, second)

    def test_holm_adjustment_is_monotone_in_rank_order(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)

    def test_transition_mutual_information_distinguishes_dependence(self):
        alternating = ["normal", "mixed"] * 50
        constant = ["normal"] * 100
        self.assertGreater(pattern_transition_mutual_information(alternating), 0.5)
        self.assertAlmostEqual(pattern_transition_mutual_information(constant), 0.0)


if __name__ == "__main__":
    unittest.main()
