import math
import unittest

from lotto_engine.candidates import iter_all_combinations
from lotto_engine.mixed_subtypes import (
    build_dynamic_family_model,
    build_family_transition_matrix,
    calculate_decay_momentum,
    family_dynamic_scores,
    lift_to_score,
)


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

    def test_neutral_lifts_and_missing_family_are_safe(self):
        self.assertAlmostEqual(lift_to_score(1.0), 50.0)
        model = build_dynamic_family_model([
            record(1, "mixed", "a"), record(2, "mixed", "b")
        ])
        transition, momentum = family_dynamic_scores("missing", model)
        self.assertTrue(0.0 <= transition <= 100.0)
        self.assertEqual(momentum, 50.0)

    def test_exhaustive_iterator_count_is_exact(self):
        self.assertEqual(sum(1 for _ in iter_all_combinations()), 8_145_060)


if __name__ == "__main__":
    unittest.main()
