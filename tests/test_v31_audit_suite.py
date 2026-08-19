import unittest

from scripts.run_v31_audit_suite import AUDIT_ORDER, _mode_settings


class V31AuditSuiteTests(unittest.TestCase):
    def test_audit_order_contains_all_required_diagnostics(self):
        self.assertEqual(
            set(AUDIT_ORDER),
            {
                "reference_fixed_weights",
                "reference_retrained",
                "focused_component",
                "joint_clean3_clean4",
                "full_feature_group",
                "pair_residualization",
                "exact_structural_null",
                "ridge_dependency",
            },
        )

    def test_full_mode_is_stricter_than_quick_mode(self):
        quick = _mode_settings("quick")
        full = _mode_settings("full")
        self.assertGreater(full["baseline_samples"], quick["baseline_samples"])
        self.assertGreater(full["bootstrap_reps"], quick["bootstrap_reps"])
        self.assertGreater(
            full["focused_latest_candidate_count"],
            quick["focused_latest_candidate_count"],
        )
        self.assertGreater(len(full["reference_counts"]), len(quick["reference_counts"]))


if __name__ == "__main__":
    unittest.main()
