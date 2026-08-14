import unittest

from lotto_engine.v27_release import (
    MODEL_VERSION,
    public_recommendation_payload,
    renormalized_static_score,
    without_dynamic_context,
)


class V27ReleaseTests(unittest.TestCase):
    def test_static_score_excludes_legacy_transition_and_renormalizes(self):
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

    def test_public_contract_uses_static_score_only(self):
        payload = {
            "meta": {
                "model_version": MODEL_VERSION,
                "release_status": "release_candidate",
                "generated_at_kst": "2026-08-14T18:00:00+09:00",
                "latest_draw": 1235,
                "target_draw": 1236,
                "recommendation_mode": "sampled_candidates",
                "evaluated_count": 100,
                "selection_strategy": "pure_static_score_top_k",
                "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
                "dynamic_components": {
                    "transition": "disabled_after_v2.7_validation",
                    "momentum": "disabled_after_v2.7_validation",
                },
                "score_disclaimer": "test",
            },
            "recommendations": [{
                "rank": 1,
                "numbers": [1, 2, 3, 4, 5, 6],
                "prediction_score": 77.5,
                "pattern_type": "mixed",
                "score_origin": "mixed_static_base",
                "score_breakdown": {"mixed_lift_score": 70.0},
            }],
        }
        public = public_recommendation_payload(payload)
        item = public["recommendations"][0]
        self.assertEqual(public["model_version"], "v2.7")
        self.assertEqual(item["score"], 77.5)
        self.assertNotIn("transition_lift_score", item)
        self.assertNotIn("momentum_lift_score", item)

    def test_release_contract_declares_dynamic_components_disabled(self):
        payload = {
            "meta": {
                "model_version": "v2.7",
                "release_status": "release_candidate",
                "generated_at_kst": "x",
                "latest_draw": 1,
                "target_draw": 2,
                "recommendation_mode": "x",
                "evaluated_count": 1,
                "selection_strategy": "pure_static_score_top_k",
                "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
                "dynamic_components": {"transition": "disabled", "momentum": "disabled"},
                "score_disclaimer": "x",
            },
            "recommendations": [],
        }
        public = public_recommendation_payload(payload)
        self.assertTrue(public["dynamic_components"]["transition"].startswith("disabled"))
        self.assertTrue(public["dynamic_components"]["momentum"].startswith("disabled"))


if __name__ == "__main__":
    unittest.main()
