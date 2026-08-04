import unittest
from collections import Counter

from lotto_engine.filters import passes_hard_filter
from lotto_engine.loader import load_lotto_data
from lotto_engine.mixed_scoring import structure_record
from lotto_engine.mixed_subtypes import (
    build_exact_mixed_subtype_baseline,
    build_historical_subtype_records,
    latest_target_draw,
    signature_family,
    signature_family_diagnostics,
    subtype_information_diagnostics,
    suggest_mixed_signature_allocation,
)
from lotto_engine.recommender import _select_diverse_mixed


class MixedSubtypeAllocationV25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_lotto_data()
        cls.records = build_historical_subtype_records(cls.df)
        cls.baseline = build_exact_mixed_subtype_baseline()
        cls.allocation = suggest_mixed_signature_allocation(cls.records, baseline=cls.baseline)
        cls.subtype_diagnostics = subtype_information_diagnostics(cls.records, cls.baseline)
        cls.family_diagnostics = signature_family_diagnostics(cls.records, cls.baseline)

    def test_latest_target_and_family_allocation_are_dynamic(self):
        latest, target = latest_target_draw(self.df)
        self.assertEqual(target, latest + 1)
        self.assertEqual(sum(self.allocation.values()), 6)

    def test_soft_selection_tracks_generated_family_targets(self):
        pool = []
        for family, target in self.allocation.items():
            matches = [record for record in self.records if signature_family(record["subtype_signature"]) == family]
            self.assertGreaterEqual(len(matches), target)
            for historical in matches[:target]:
                structure = structure_record(historical["numbers"])
                pool.append({
                    **historical,
                    "numbers": list(historical["numbers"]),
                    "mixed_slot_score": 70.0,
                    "structure_record": structure,
                    "extreme_signature": list(structure["extreme_signature"]),
                })
        selected = _select_diverse_mixed(
            pool, 6, self.allocation, self.subtype_diagnostics, self.family_diagnostics
        )
        selected_counts = Counter(item["selected_family"] for item in selected)
        self.assertEqual(len(selected), 6)
        self.assertGreaterEqual(sum(min(selected_counts[key], value) for key, value in self.allocation.items()), 5)
        self.assertLess(selected_counts.get("section_hole", 0), 3)

    def test_no_hard_filters_are_introduced(self):
        self.assertTrue(passes_hard_filter([1, 2, 3, 4, 5, 6]))
        self.assertTrue(passes_hard_filter([40, 41, 42, 43, 44, 45]))


if __name__ == "__main__":
    unittest.main()
