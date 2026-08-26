import unittest

from lotto_engine.final_backtest import _largest_remainder, _portfolio_metrics


class FinalWalkForwardValidationTests(unittest.TestCase):
    def test_largest_remainder_always_allocates_exact_ticket_count(self):
        allocation = _largest_remainder(
            {"normal": 0.21, "mixed": 0.51, "outlier": 0.28}, 10
        )
        self.assertEqual(sum(allocation.values()), 10)
        self.assertEqual(allocation, {"normal": 2, "mixed": 5, "outlier": 3})

    def test_portfolio_metrics_measure_best_hit_and_coverage(self):
        portfolio = [
            {"numbers": [1, 2, 3, 10, 11, 12]},
            {"numbers": [1, 4, 5, 20, 21, 22]},
        ]
        metrics = _portfolio_metrics(portfolio, [1, 2, 3, 4, 5, 6])
        self.assertEqual(metrics["best_hit"], 3)
        self.assertAlmostEqual(metrics["mean_hit"], 2.5)
        self.assertAlmostEqual(metrics["coverage"], 5 / 6)
        self.assertEqual(metrics["hit3"], 1)
        self.assertEqual(metrics["hit4"], 0)


if __name__ == "__main__":
    unittest.main()
