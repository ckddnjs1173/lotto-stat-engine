import math
import unittest
from collections import Counter

from lotto_engine.candidates import iter_all_combinations
from lotto_engine.mixed_subtypes import (
    build_dynamic_family_model,
    build_family_transition_matrix,
    calculate_decay_momentum,
    family_dynamic_scores,
    lift_to_score,
)
from lotto_engine.recommender import (
    PATTERN_TYPES,
    _flatten_type_heaps,
    _portfolio_allocation,
    _push_type_candidate,
    _type_pool_sizes,
)
from lotto_engine.scoring import FINAL_STATIC_WEIGHTS


def record(draw, pattern, family, coarse=None):
    return {
        "draw_no": draw,
        "pattern_type": pattern,
        "subtype_signature": family,
        "primary_subtype": coarse or family,
    }


class FinalDynamicModelTests(unittest.TestCase):
    def test_transitions_use_only_real_adjacent_draws(self):
        records = [
            record(1, "mixed", "a"),
            record(2, "normal", "bridge"),
            record(3, "mixed", "b"),
        ]
        model = build_dynamic_family_model(records)
        self.assertEqual(model["exact_counts"]["a"]["bridge"], 1)
        self.assertEqual(model["exact_counts"]["a"]["b"], 0)
        matrix = build_family_transition_matrix(records)
        self.assertGreater(matrix["a"]["bridge"], matrix["a"]["b"])

    def test_latest_draw_and_transition_state_change_with_append(self):
        before = [record(10, "normal", "a"), record(11, "mixed", "b")]
        after = before + [record(12, "outlier", "c")]
        first = build_dynamic_family_model(before)
        second = build_dynamic_family_model(after)
        self.assertEqual(first["latest_draw"], 11)
        self.assertEqual(second["latest_draw"], 12)
        self.assertNotEqual(first["latest_family"], second["latest_family"])

    def test_sparse_exact_transition_is_shrunk_but_strong_evidence_can_override(self):
        base_model = {
            "alpha": 0.1,
            "transition_prior_strength": 12.0,
            "families": ["x", "y"],
            "priors": {"x": 0.5, "y": 0.5},
            "type_counts": {"outlier": Counter({"y": 20})},
            "coarse_counts": {"rare_coarse": Counter({"y": 5})},
            "exact_counts": {"rare": Counter({"x": 1})},
            "latest_pattern_type": "outlier",
            "latest_coarse_family": "rare_coarse",
            "latest_family": "rare",
            "momentum_scores": {},
        }
        sparse_x, _ = family_dynamic_scores("x", base_model)
        sparse_y, _ = family_dynamic_scores("y", base_model)
        self.assertLess(sparse_x, 50.0)
        self.assertGreater(sparse_y, 50.0)

        strong_model = dict(base_model)
        strong_model["exact_counts"] = {"rare": Counter({"x": 50})}
        strong_x, _ = family_dynamic_scores("x", strong_model)
        self.assertGreater(strong_x, 50.0)
        self.assertGreater(strong_x, sparse_x)

    def test_momentum_uses_global_latest_draw_and_changes(self):
        records = [
            record(1, "mixed", "a"),
            record(2, "mixed", "b"),
            record(5, "normal", "c"),
        ]
        actual = calculate_decay_momentum(records, decay_rate=0.05, alpha=0.1)
        wa, wb = math.exp(-0.20), math.exp(-0.15)
        expected_a = ((wa + 0.1) / (wa + wb + 0.2)) / (1.1 / 2.2)
        self.assertAlmostEqual(actual["a"], expected_a, places=10)
        changed = calculate_decay_momentum(records + [record(6, "mixed", "a")])
        self.assertNotEqual(actual, changed)

    def test_production_momentum_uses_all_draw_types_and_recency(self):
        records = [
            record(1, "normal", "old_normal"),
            record(2, "mixed", "mixed_history"),
            record(98, "normal", "recent_normal"),
            record(99, "normal", "recent_normal"),
            record(100, "outlier", "recent_outlier"),
        ]
        model = build_dynamic_family_model(records, decay_rate=0.05)
        _, recent_score = family_dynamic_scores("recent_normal", model)
        _, old_score = family_dynamic_scores("old_normal", model)
        self.assertGreater(recent_score, 50.0)
        self.assertLess(old_score, 50.0)
        self.assertNotEqual(recent_score, old_score)

    def test_neutral_lifts_and_missing_family_are_safe(self):
        self.assertAlmostEqual(lift_to_score(1.0), 50.0)
        model = build_dynamic_family_model([
            record(1, "mixed", "a"), record(2, "mixed", "b")
        ])
        transition, momentum = family_dynamic_scores("missing", model)
        self.assertTrue(0.0 <= transition <= 100.0)
        self.assertEqual(momentum, 50.0)

    def test_type_partitioned_pool_prevents_global_score_monopoly(self):
        allocation = {"normal": 2, "mixed": 6, "outlier": 2}
        pool_sizes = _type_pool_sizes(allocation, 10)
        heaps = {candidate_type: [] for candidate_type in PATTERN_TYPES}
        serial = 0

        for index in range(3000):
            serial += 1
            item = {"pattern_type": "outlier", "numbers": [index]}
            _push_type_candidate(
                heaps, pool_sizes, "outlier", (1000.0 - index * 0.001, serial, item)
            )
        for candidate_type, base_score in (("normal", 20.0), ("mixed", 30.0)):
            for index in range(20):
                serial += 1
                item = {"pattern_type": candidate_type, "numbers": [index]}
                _push_type_candidate(
                    heaps, pool_sizes, candidate_type, (base_score - index * 0.01, serial, item)
                )

        retained = _flatten_type_heaps(heaps)
        retained_counts = Counter(item["pattern_type"] for item in retained)
        self.assertEqual(retained_counts["normal"], 20)
        self.assertEqual(retained_counts["mixed"], 20)
        self.assertEqual(retained_counts["outlier"], pool_sizes["outlier"])

    def test_zero_target_type_still_keeps_diagnostic_reserve(self):
        pool_sizes = _type_pool_sizes({"normal": 0, "mixed": 10, "outlier": 0}, 10)
        self.assertGreater(pool_sizes["normal"], 0)
        self.assertGreater(pool_sizes["outlier"], 0)

    def test_static_final_weights_exclude_type_and_transition_signals(self):
        for candidate_type, weights in FINAL_STATIC_WEIGHTS.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertNotIn("type_balance_score", weights)
            self.assertNotIn("transition_score", weights)
            self.assertNotIn("number_dynamics_score", weights)
            self.assertIn(candidate_type, {"normal", "outlier"})

    def test_dynamic_type_budget_is_exact(self):
        profile = {
            "latest_pattern_type": "outlier",
            "transition_probs": {
                "outlier": {"normal": 0.21, "mixed": 0.51, "outlier": 0.28}
            },
            "pattern_type_probs": {"normal": 0.2, "mixed": 0.55, "outlier": 0.25},
        }
        allocation = _portfolio_allocation(profile, 10)
        self.assertEqual(sum(allocation.values()), 10)
        self.assertEqual(allocation, {"normal": 2, "mixed": 5, "outlier": 3})

    def test_exhaustive_iterator_count_is_exact(self):
        self.assertEqual(sum(1 for _ in iter_all_combinations()), 8_145_060)


if __name__ == "__main__":
    unittest.main()
