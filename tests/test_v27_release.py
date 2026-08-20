import unittest

from lotto_engine.v27_release import (
    MODEL_VERSION,
    _seeded_fair_tiebreak,
    public_recommendation_payload,
    renormalized_static_score,
    without_dynamic_context,
)


class V27ReleaseTests(unittest.TestCase):
    def test_legacy_static_score_excludes_transition_for_reproducible_audit(self):
        breakdown = {
            "outlier_survival_score": 80.0,
            "type_balance_score": 60.0,
            "transition_score": 100.0,
            "normal_structure_score": 40.0,
            "historical_pattern_score": 20.0,
            "number_dynamics_score": 10.0,
        }
        score, active = renormalized_static_score(breakdown)
        expected = (0.45 * 80.0 + 0.25 * 60.0 + 0.15 * 40.0) / 0.85
        self.assertAlmostEqual(score, expected)
        self.assertNotIn("transition_score", active)
        self.assertEqual(set(active), {
            "outlier_survival_score",
            "type_balance_score",
            "normal_structure_score",
        })

    def test_without_dynamic_context_does_not_mutate_source(self):
        source = {"full": {"total": 10}, "dynamic_markov_decay": {"latest_draw": 10}}
        static = without_dynamic_context(source)
        self.assertIn("dynamic_markov_decay", source)
        self.assertNotIn("dynamic_markov_decay", static)
        self.assertEqual(static["full"], source["full"])

    def test_public_contract_exposes_unified_evidence(self):
        payload = {
            "meta": {
                "model_version": MODEL_VERSION,
                "release_status": "research_candidate",
                "generated_at_kst": "2026-08-18T11:00:00+09:00",
                "latest_draw": 1235,
                "target_draw": 1236,
                "recommendation_mode": "sampled_candidates",
                "evaluated_count": 100,
                "selection_strategy": "unified_bayesian_evidence_top_k",
                "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
                "dynamic_components": {
                    "transition": "disabled_after_v2.7_validation",
                    "momentum": "disabled_after_v2.7_validation",
                },
                "static_structure_components": "diagnostic_only_after_phase5a_no_survivors",
                "cross_type_calibration": "not_required_single_common_evidence_equation",
                "pattern_type_role": "metadata_only_not_used_for_ranking",
                "tie_break_policy": "seeded_structure_neutral_only_for_exact_evidence_ties",
                "evidence_models": {
                    "number": {"reliability": 0.01},
                    "pair": {"reliability": 0.02},
                },
                "score_disclaimer": "test",
            },
            "recommendations": [{
                "rank": 1,
                "numbers": [1, 2, 3, 4, 5, 6],
                "prediction_score": 50.125,
                "ranking_score": 0.0025,
                "pattern_type": "mixed",
                "score_origin": "unified_bayesian_evidence_v1",
                "score_breakdown": {
                    "number_log_lift": 0.1,
                    "pair_log_lift": 0.2,
                },
            }],
        }
        public = public_recommendation_payload(payload)
        item = public["recommendations"][0]
        self.assertEqual(public["model_version"], "v2.7.1")
        self.assertEqual(public["pattern_type_role"], "metadata_only_not_used_for_ranking")
        self.assertEqual(public["evidence_models"]["number"]["reliability"], 0.01)
        self.assertEqual(item["score"], 50.125)
        self.assertEqual(item["ranking_score"], 0.0025)
        self.assertEqual(item["score_origin"], "unified_bayesian_evidence_v1")
        self.assertNotIn("transition_lift_score", item["components"])
        self.assertNotIn("momentum_lift_score", item["components"])

    def test_release_contract_declares_dynamic_components_disabled(self):
        payload = {
            "meta": {
                "model_version": MODEL_VERSION,
                "release_status": "research_candidate",
                "generated_at_kst": "x",
                "latest_draw": 1,
                "target_draw": 2,
                "recommendation_mode": "x",
                "evaluated_count": 1,
                "selection_strategy": "unified_bayesian_evidence_top_k",
                "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
                "dynamic_components": {"transition": "disabled", "momentum": "disabled"},
                "score_disclaimer": "x",
            },
            "recommendations": [],
        }
        public = public_recommendation_payload(payload)
        self.assertTrue(public["dynamic_components"]["transition"].startswith("disabled"))
        self.assertTrue(public["dynamic_components"]["momentum"].startswith("disabled"))

    def test_exact_tie_break_is_deterministic_and_number_order_independent(self):
        first = _seeded_fair_tiebreak([1, 2, 3, 4, 5, 6], 1235)
        second = _seeded_fair_tiebreak([1, 2, 3, 4, 5, 6], 1235)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
