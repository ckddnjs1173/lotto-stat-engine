import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lotto_engine.config import NUMBER_COLUMNS, ROUND_COLUMN
from lotto_engine.v27_audit_artifact import dataframe_fingerprint, write_json_artifact


class V27AuditArtifactTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for draw, numbers in ((2, [2, 3, 4, 5, 6, 7]), (1, [1, 2, 3, 4, 5, 6])):
            row = {ROUND_COLUMN: draw}
            row.update(dict(zip(NUMBER_COLUMNS, numbers)))
            rows.append(row)
        return pd.DataFrame(rows)

    def test_fingerprint_is_stable_across_row_order(self):
        df = self._frame()
        first = dataframe_fingerprint(df)
        second = dataframe_fingerprint(df.iloc[::-1].reset_index(drop=True))
        self.assertEqual(first, second)
        self.assertEqual(first["row_count"], 2)
        self.assertEqual(first["first_draw"], 1)
        self.assertEqual(first["latest_draw"], 2)

    def test_fingerprint_changes_when_draw_data_changes(self):
        df = self._frame()
        changed = df.copy()
        changed.loc[changed[ROUND_COLUMN] == 2, NUMBER_COLUMNS[-1]] = 8
        self.assertNotEqual(
            dataframe_fingerprint(df)["sha256"],
            dataframe_fingerprint(changed)["sha256"],
        )

    def test_json_artifact_replaces_non_finite_values_with_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            write_json_artifact({"value": math.nan, "nested": [math.inf, 1.0]}, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(payload["value"])
        self.assertIsNone(payload["nested"][0])
        self.assertEqual(payload["nested"][1], 1.0)

    def test_json_artifact_converts_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            write_json_artifact(
                {
                    "tests": np.int64(640),
                    "score": np.float64(51.25),
                    "flag": np.bool_(True),
                    "missing": np.float64(np.nan),
                },
                path,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["tests"], 640)
        self.assertEqual(payload["score"], 51.25)
        self.assertIs(payload["flag"], True)
        self.assertIsNone(payload["missing"])


if __name__ == "__main__":
    unittest.main()
