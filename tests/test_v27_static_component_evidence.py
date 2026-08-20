import random
import unittest
from unittest.mock import patch

from lotto_engine.v27_static_component_evidence import (
    MIXED_COMPONENTS,
    NORMAL_OUTLIER_COMPONENTS,
    _passes_gate,
    _sample_same_type_records,
    mixed_static_values,
    normal_outlier_static_values,
)


class V27StaticComponentEvidenceTests(unittest.TestCase):
    def test_same_type_sampling_is_deterministic(self):
        first = _sample_same_type_records(random.Random(123), 8, "mixed")
        second = _sample_same_type_records(random.Random(123), 8, "mixed")
        self.assertEqual(
            [record["numbers"] for record in first],
            [record["numbers"] for record in second],
        )
        self.assertTrue(all(record["pattern_type"] == "mixed" for record in first))
        self.assertEqual(len({record["numbers"] for record in first}), 8)

    @patch("lotto_engine.v27_static_component_evidence.score_mixed_record")
    def test_mixed_static_values_removes_dynamic_context(self, mocked_score):
        mocked_score.return_value = {
            "mixed_slot_score": 51.0,
            **{name: 50.0 + index for index, name in enumerate(MIXED_COMPONENTS)},
        }
        profile = {"dynamic_markov_decay": {"should": "be removed"}, "x": 1}
        values = mixed_static_values({"numbers": (1, 2, 3, 4, 5, 6)}, profile)
        passed_profile = mocked_score.call_args.args[1]
        self.assertNotIn("dynamic_markov_decay", passed_profile)
        self.assertIn("dynamic_markov_decay", profile)
        self.assertEqual(values["current_branch_base"], 51.0)

    @patch("lotto_engine.v27_static_component_evidence.score_candidate")
    def test_normal_outlier_values_expose_predeclared_components(self, mocked_score):
        mocked_score.return_value = {
            "prediction_score": 53.0,
            "score_breakdown": {
                name: 40.0 + index
                for index, name in enumerate(NORMAL_OUTLIER_COMPONENTS)
            },
        }
        values = normal_outlier_static_values([1, 2, 3, 4, 5, 6], {"x": 1})
        self.assertEqual(values["current_branch_base"], 53.0)
        self.assertEqual(set(values) - {"current_branch_base"}, set(NORMAL_OUTLIER_COMPONENTS))

    def test_gate_requires_ci_and_both_recent_windows(self):
        passing = {
            "tests": 100,
            "mean_percentile": 52.0,
            "recent_300_mean_percentile": 51.0,
            "recent_100_mean_percentile": 50.1,
            "percentile_minus_50_block_bootstrap_95_ci": [0.2, 3.8],
        }
        self.assertTrue(_passes_gate(passing))
        failing = dict(passing)
        failing["recent_100_mean_percentile"] = 49.9
        self.assertFalse(_passes_gate(failing))


if __name__ == "__main__":
    unittest.main()
