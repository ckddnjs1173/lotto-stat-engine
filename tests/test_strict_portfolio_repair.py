import unittest

from lotto_engine.strict_portfolio import select_strict_portfolio_entries


class StrictPortfolioRepairTests(unittest.TestCase):
    def _entry(self, score, numbers):
        return (float(score), 0, tuple(numbers))

    def test_minimum_removal_repair_can_complete_when_greedy_gets_stuck(self):
        ranked = [
            self._entry(10, (1, 2, 3, 4, 5, 6)),
            self._entry(9, (1, 7, 8, 9, 10, 11)),
            self._entry(8, (1, 12, 13, 14, 15, 16)),
            self._entry(7, (2, 3, 4, 17, 18, 19)),
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked,
            top_k=3,
            max_shared_numbers=2,
            max_number_exposure_rate=2.0 / 3.0,
        )
        self.assertTrue(meta["complete"])
        self.assertTrue(meta["repair_used"])
        self.assertEqual(meta["repair_removed_from_greedy"], 1)
        self.assertEqual([entry[2] for entry in selected], [
            ranked[1][2], ranked[2][2], ranked[3][2]
        ])
        self.assertFalse(meta["ranking_score_modified"])

    def test_incomplete_result_is_revalidated_against_actual_ticket_count(self):
        ranked = [
            self._entry(10, (1, 2, 3, 4, 5, 6)),
            self._entry(9, (1, 7, 8, 9, 10, 11)),
            self._entry(8, (12, 13, 14, 15, 16, 17)),
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked,
            top_k=4,
            max_shared_numbers=2,
            max_number_exposure_rate=0.5,
        )
        self.assertFalse(meta["complete"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(meta["max_number_ticket_count"], 1)
        self.assertLessEqual(meta["observed_max_number_exposure_rate"], 0.5)
        self.assertTrue(meta["actual_denominator_exposure_check"])
        self.assertTrue(meta["exposure_constraint_satisfied"])

    def test_complete_result_keeps_raw_rank_order_and_constraints(self):
        ranked = [
            self._entry(10 - index, tuple(range(1 + index * 6, 7 + index * 6)))
            for index in range(5)
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked,
            top_k=5,
            max_shared_numbers=2,
            max_number_exposure_rate=0.4,
        )
        self.assertTrue(meta["complete"])
        self.assertEqual(selected, ranked)
        self.assertTrue(meta["portfolio_entries_sorted_by_raw_rank"])
        self.assertTrue(meta["overlap_constraint_satisfied"])
        self.assertTrue(meta["exposure_constraint_satisfied"])


if __name__ == "__main__":
    unittest.main()
