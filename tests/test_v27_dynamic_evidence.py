import random
import unittest

from lotto_engine.mixed_scoring import structure_record
from lotto_engine.mixed_subtypes import (
    build_dynamic_family_model,
    family_dynamic_scores,
    lift_to_score,
    signature_family,
)
from lotto_engine.v27_dynamic_evidence import (
    _component_summary,
    _sample_unique_combinations,
    raw_dynamic_lifts,
)


class V27DynamicEvidenceTests(unittest.TestCase):
    def _model(self):
        combinations = (
            [1, 2, 3, 20, 34, 45],
            [4, 9, 10, 21, 33, 44],
            [2, 7, 15, 22, 31, 41],
            [3, 8, 16, 25, 32, 43],
            [5, 11, 17, 26, 35, 42],
            [1, 12, 18, 27, 36, 40],
        )
        records = []
        for draw_no, numbers in enumerate(combinations, 1):
            record = structure_record(numbers)
            record["draw_no"] = draw_no
            records.append(record)
        return records, build_dynamic_family_model(records)

    def test_raw_exact_transition_and_momentum_mirror_production_mapping(self):
        records, model = self._model()
        family = signature_family(records[-1]["subtype_signature"])
        raw = raw_dynamic_lifts(family, model)
        transition_score, momentum_score = family_dynamic_scores(family, model)
        self.assertAlmostEqual(lift_to_score(raw["transition_exact"]), transition_score)
        self.assertAlmostEqual(lift_to_score(raw["momentum"]), momentum_score)

    def test_raw_dynamic_lifts_are_finite_for_unseen_family(self):
        _, model = self._model()
        lifts = raw_dynamic_lifts("never_seen_family", model)
        self.assertEqual(set(lifts), {
            "transition_type", "transition_coarse", "transition_exact", "momentum"
        })
        for value in lifts.values():
            self.assertGreater(value, 0.0)

    def test_unique_candidate_sampling_is_deterministic(self):
        first = _sample_unique_combinations(random.Random(123), 30)
        second = _sample_unique_combinations(random.Random(123), 30)
        self.assertEqual(first, second)
        self.assertEqual(len({tuple(values) for values in first}), 30)

    def test_primary_candidate_gate_requires_positive_ci_and_recent_stability(self):
        result = _component_summary(
            [60.0] * 80,
            [0.1] * 80,
            ["mixed"] * 80,
            "transition_exact",
            bootstrap_reps=50,
        )
        self.assertTrue(result["screening_candidate"])
        diagnostic = _component_summary(
            [60.0] * 80,
            [0.1] * 80,
            ["mixed"] * 80,
            "transition_coarse",
            bootstrap_reps=50,
        )
        self.assertFalse(diagnostic["screening_candidate"])


if __name__ == "__main__":
    unittest.main()
