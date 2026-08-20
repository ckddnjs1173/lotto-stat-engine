import unittest

from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC
from lotto_engine.v31_retrained_reference_audit import _variant_labels


class V31RetrainedReferenceAuditTests(unittest.TestCase):
    def test_variant_labels_include_current_and_nested_streams(self):
        labels = _variant_labels(
            FULL11_BASELINE_SPEC,
            (1024, 4096),
            (0, 1),
        )
        self.assertEqual(labels[0], "current_policy:1024")
        self.assertEqual(len(labels), 5)
        self.assertIn("nested_count:1024:stream:0", labels)
        self.assertIn("nested_count:4096:stream:0", labels)
        self.assertIn("nested_count:1024:stream:1", labels)
        self.assertIn("nested_count:4096:stream:1", labels)
        self.assertEqual(len(set(labels)), len(labels))


if __name__ == "__main__":
    unittest.main()
