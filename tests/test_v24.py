import unittest

import pandas as pd

from lotto_engine.candidates import iter_all_combinations
from lotto_engine.config import ROUND_COLUMN
from lotto_engine.filters import passes_hard_filter
from lotto_engine.mixed_scoring import structure_record
from lotto_engine.loader import load_lotto_data
from lotto_engine.mixed_subtypes import (
    build_exact_mixed_subtype_baseline,
    build_historical_subtype_records,
    latest_target_draw,
    subtype_record,
    suggest_mixed_signature_allocation,
    suggest_mixed_subtype_allocation,
)


class MixedSubtypeV24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = build_historical_subtype_records(load_lotto_data())
        cls.baseline = build_exact_mixed_subtype_baseline()

    def test_latest_and_target_draw_are_dynamic(self):
        df = pd.DataFrame({ROUND_COLUMN: [9, 12, 10]})
        self.assertEqual(latest_target_draw(df), (12, 13))

    def test_draw_1235_structural_tags(self):
        result = subtype_record([6, 7, 11, 15, 39, 43])
        for tag in ("parity_skew", "gap_bridge", "section_hole", "consecutive_anchor", "low_high_split"):
            self.assertIn(tag, result["subtype_tags"])

    def test_subtype_signature_is_deterministic(self):
        first = subtype_record([43, 6, 15, 7, 39, 11])
        second = subtype_record([6, 7, 11, 15, 39, 43])
        self.assertEqual(first["subtype_signature"], second["subtype_signature"])
        self.assertEqual(first["primary_subtype"], second["primary_subtype"])

    def test_candidate_structure_contains_v24_dimensions(self):
        record = structure_record(next(iter_all_combinations()))
        for key in ("pattern_type", "subtype_tags", "primary_subtype", "subtype_signature", "cluster_shape", "gap_shape"):
            self.assertIn(key, record)

    def test_no_hard_filters_are_introduced(self):
        self.assertTrue(passes_hard_filter([1, 2, 3, 4, 5, 6]))
        self.assertTrue(passes_hard_filter([40, 41, 42, 43, 44, 45]))

    def test_common_neutral_section_hole_does_not_dominate(self):
        allocation = suggest_mixed_subtype_allocation(self.records, baseline=self.baseline)
        self.assertLessEqual(allocation.get("section_hole", 0), 2)
        self.assertLess(allocation.get("section_hole", 0), sum(allocation.values()) / 2)

    def test_signature_allocation_prefers_supported_informative_families(self):
        allocation = suggest_mixed_signature_allocation(self.records, baseline=self.baseline)
        self.assertEqual(sum(allocation.values()), 6)
        self.assertTrue(all("+" in family for family in allocation))
        self.assertTrue(any(any(tag in family for tag in ("ending_duplicate", "high_cluster", "range_edge", "parity_extreme", "sum_edge")) for family in allocation))


if __name__ == "__main__":
    unittest.main()
