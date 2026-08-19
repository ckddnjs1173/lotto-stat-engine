import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v31_core import (
    build_latest_model,
    directional_raw_features,
)
from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC


def _context():
    return {
        "number_lifts": np.zeros(46, dtype=float),
        "pair_lifts": np.zeros(990, dtype=float),
        "recent20_number_rate": np.zeros(46, dtype=float),
        "recent100_number_rate": np.zeros(46, dtype=float),
        "recent100_pair_rate": np.zeros(990, dtype=float),
        "previous_draw": frozenset({1, 2, 3, 4, 5, 6}),
    }


def _synthetic_history(rounds: int = 10) -> pd.DataFrame:
    rows = []
    for round_no in range(1, rounds + 1):
        start = ((round_no - 1) * 4) % 45
        numbers = sorted({((start + offset) % 45) + 1 for offset in (0, 3, 8, 14, 21, 29)})
        if len(numbers) != 6:
            raise AssertionError("synthetic history must contain six unique numbers")
        rows.append(
            {
                ROUND_COLUMN: round_no,
                **{column: numbers[index] for index, column in enumerate(NUMBER_COLUMNS)},
            }
        )
    return pd.DataFrame(rows)


class V31ReproducibilityEligibilityTests(unittest.TestCase):
    def test_unusual_valid_combinations_are_all_eligible_for_raw_features(self):
        cases = (
            (1, 2, 3, 4, 5, 6),
            (40, 41, 42, 43, 44, 45),
            (1, 3, 5, 7, 9, 11),
            (2, 4, 6, 8, 10, 12),
            (1, 2, 40, 41, 44, 45),
        )
        for numbers in cases:
            vector = directional_raw_features(numbers, _context())
            self.assertEqual(vector.shape, (11,))
            self.assertTrue(np.all(np.isfinite(vector)))

    def test_same_data_and_spec_reproduce_identical_fitted_weights(self):
        spec = replace(
            FULL11_BASELINE_SPEC,
            name="test_reproducibility_spec",
            history_start_index=5,
            minimum_meta_train_targets=1,
            fair_reference_samples_per_target=32,
            train_negatives_per_target=4,
        )
        df = _synthetic_history(10)
        first = build_latest_model(df, spec)
        second = build_latest_model(df.copy(), spec)
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        self.assertTrue(
            np.array_equal(
                first["model"]["effective_weights"],
                second["model"]["effective_weights"],
            )
        )
        self.assertTrue(np.array_equal(first["reference"], second["reference"]))


if __name__ == "__main__":
    unittest.main()
