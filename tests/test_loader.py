import unittest

import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.loader import LottoDataError, dataset_fingerprint, validate_lotto_data


def _frame(rounds):
    rows = []
    for round_no in rounds:
        base = ((round_no - 1) % 40) + 1
        nums = sorted({
            ((base + offset - 1) % 45) + 1
            for offset in (0, 3, 7, 12, 18, 25)
        })
        rows.append({
            ROUND_COLUMN: round_no,
            **{column: nums[index] for index, column in enumerate(NUMBER_COLUMNS)},
        })
    return pd.DataFrame(rows)


class LoaderValidationTests(unittest.TestCase):
    def test_continuous_history_from_draw_one_is_valid(self):
        validate_lotto_data(_frame([1, 2, 3]))

    def test_missing_round_is_rejected(self):
        with self.assertRaises(LottoDataError):
            validate_lotto_data(_frame([1, 3]))

    def test_history_not_starting_at_one_is_rejected(self):
        with self.assertRaises(LottoDataError):
            validate_lotto_data(_frame([2, 3]))

    def test_fractional_round_is_rejected_instead_of_truncated(self):
        df = _frame([1, 2, 3]).astype(float)
        df.loc[1, ROUND_COLUMN] = 2.5
        with self.assertRaises(LottoDataError):
            validate_lotto_data(df)

    def test_fractional_winning_number_is_rejected_instead_of_truncated(self):
        df = _frame([1, 2, 3]).astype(float)
        df.loc[1, NUMBER_COLUMNS[0]] = 12.5
        with self.assertRaises(LottoDataError):
            validate_lotto_data(df)

    def test_non_finite_winning_number_is_rejected(self):
        df = _frame([1, 2, 3]).astype(float)
        df.loc[1, NUMBER_COLUMNS[0]] = float("nan")
        with self.assertRaises(LottoDataError):
            validate_lotto_data(df)

    def test_fingerprint_is_stable_for_same_normalized_history(self):
        df = _frame([1, 2, 3])
        left = dataset_fingerprint(df)
        right = dataset_fingerprint(df.copy())
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_fingerprint_changes_when_a_winning_number_changes(self):
        df = _frame([1, 2, 3])
        changed = df.copy()
        changed.loc[2, NUMBER_COLUMNS[-1]] = 45
        self.assertNotEqual(dataset_fingerprint(df), dataset_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
