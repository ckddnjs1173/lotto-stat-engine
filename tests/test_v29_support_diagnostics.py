import unittest

import numpy as np

from lotto_engine.v29_support_diagnostics import empirical_midrank_percentile


class V29SupportDiagnosticsTests(unittest.TestCase):
    def test_midrank_percentile_gives_half_credit_for_ties(self):
        reference = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=float)
        self.assertAlmostEqual(empirical_midrank_percentile(2.0, reference), 50.0)

    def test_midrank_percentile_handles_extremes(self):
        reference = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
        self.assertAlmostEqual(empirical_midrank_percentile(0.0, reference), 0.0)
        self.assertAlmostEqual(empirical_midrank_percentile(5.0, reference), 100.0)

    def test_midrank_percentile_requires_nonempty_vector(self):
        with self.assertRaises(ValueError):
            empirical_midrank_percentile(1.0, np.asarray([], dtype=float))
        with self.assertRaises(ValueError):
            empirical_midrank_percentile(1.0, np.zeros((2, 2), dtype=float))


if __name__ == "__main__":
    unittest.main()
