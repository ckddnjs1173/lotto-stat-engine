import unittest

import numpy as np

from lotto_engine.v31_core import FEATURE_NAMES
from lotto_engine.v31_dependency_audit import dependency_pairs_from_matrix


class V31DependencyAuditTests(unittest.TestCase):
    def test_dependency_pairs_are_sorted_by_absolute_strength(self):
        width = len(FEATURE_NAMES)
        matrix = np.eye(width, dtype=float)
        matrix[0, 1] = matrix[1, 0] = -0.9
        matrix[2, 3] = matrix[3, 2] = 0.7
        pairs = dependency_pairs_from_matrix(matrix)
        self.assertEqual({pairs[0]["left"], pairs[0]["right"]}, {FEATURE_NAMES[0], FEATURE_NAMES[1]})
        strengths = [item["absolute_second_moment_cosine"] for item in pairs]
        self.assertEqual(strengths, sorted(strengths, reverse=True))

    def test_dependency_pair_count_is_choose_two(self):
        width = len(FEATURE_NAMES)
        pairs = dependency_pairs_from_matrix(np.eye(width, dtype=float))
        self.assertEqual(len(pairs), width * (width - 1) // 2)


if __name__ == "__main__":
    unittest.main()
