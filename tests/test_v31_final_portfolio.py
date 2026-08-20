import unittest

from lotto_engine.v31_final_portfolio import select_strict_portfolio_entries


class V31FinalPortfolioTests(unittest.TestCase):
    def _entry(self, score, numbers):
        return (float(score), 0, tuple(numbers))

    def test_portfolio_never_relaxes_pairwise_overlap(self):
        ranked = [
            self._entry(10, (1, 2, 3, 4, 5, 6)),
            self._entry(9, (1, 2, 3, 7, 8, 9)),
            self._entry(8, (1, 2, 10, 11, 12, 13)),
            self._entry(7, (3, 4, 14, 15, 16, 17)),
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked, top_k=3, max_number_exposure_rate=1.0
        )
        combos = [set(entry[2]) for entry in selected]
        for i in range(len(combos)):
            for j in range(i):
                self.assertLessEqual(len(combos[i] & combos[j]), 2)
        self.assertFalse(meta["fallback_relaxed"])

    def test_portfolio_caps_individual_number_exposure(self):
        ranked = [
            self._entry(10 - i, (1, 2 + i * 5, 3 + i * 5, 4 + i * 5, 5 + i * 5, 6 + i * 5))
            for i in range(6)
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked, top_k=5, max_shared_numbers=2, max_number_exposure_rate=0.4
        )
        exposure_cap = meta["max_number_ticket_count"]
        number_one_count = sum(1 in entry[2] for entry in selected)
        self.assertLessEqual(number_one_count, exposure_cap)
        self.assertFalse(meta["fallback_relaxed"])

    def test_incomplete_pool_returns_fewer_instead_of_relaxing(self):
        ranked = [
            self._entry(10, (1, 2, 3, 4, 5, 6)),
            self._entry(9, (1, 2, 3, 7, 8, 9)),
        ]
        selected, meta = select_strict_portfolio_entries(
            ranked, top_k=3, max_number_exposure_rate=1.0
        )
        self.assertLess(len(selected), 3)
        self.assertFalse(meta["complete"])
        self.assertFalse(meta["fallback_relaxed"])


if __name__ == "__main__":
    unittest.main()
