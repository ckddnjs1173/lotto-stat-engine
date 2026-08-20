import unittest

from lotto_engine.mixed_scoring import structure_record
from lotto_engine.mixed_subtypes import build_dynamic_family_model
from lotto_engine.v27_momentum_confirmation import (
    momentum_lifts_for_history,
    momentum_lookup,
    summarize_momentum_confirmation,
)


class V27MomentumConfirmationTests(unittest.TestCase):
    def _records(self):
        combos = (
            [1, 2, 3, 10, 20, 30],
            [4, 9, 14, 19, 24, 29],
            [2, 11, 17, 28, 36, 45],
            [5, 6, 15, 16, 25, 35],
            [3, 13, 23, 33, 43, 44],
            [7, 8, 18, 27, 37, 42],
        )
        records = []
        for draw_no, combo in enumerate(combos, start=1):
            record = structure_record(combo)
            record["draw_no"] = draw_no
            records.append(record)
        return records

    def test_momentum_only_state_matches_production_dynamic_model(self):
        records = self._records()
        production = build_dynamic_family_model(records)["momentum_lifts"]
        confirmation = momentum_lifts_for_history(records)
        self.assertEqual(production, confirmation)

    def test_unseen_family_is_neutral_and_flagged(self):
        lift, seen = momentum_lookup("definitely_unseen_family", {"known": 1.2})
        self.assertEqual(lift, 1.0)
        self.assertFalse(seen)

    def test_confirmation_requires_seen_family_robustness(self):
        strong_full = [60.0] * 120
        weak_seen = [50.0] * 120
        summary = summarize_momentum_confirmation(
            strong_full, weak_seen, ["mixed"] * 120, [True] * 120,
            [0.8] * 120, [800] * 120, [0.0] * 120, bootstrap_reps=25,
        )
        self.assertFalse(summary["confirmed"])

        strong_seen = [60.0] * 120
        summary = summarize_momentum_confirmation(
            strong_full, strong_seen, ["mixed"] * 120, [True] * 120,
            [0.8] * 120, [800] * 120, [0.1] * 120, bootstrap_reps=25,
        )
        self.assertTrue(summary["confirmed"])


if __name__ == "__main__":
    unittest.main()
