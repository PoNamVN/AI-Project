import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("validate_compact", Path(__file__).resolve().parents[1] / "scripts/validate_compact_runtime.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class CompactValidationTests(unittest.TestCase):
    def test_gate_rejects_regressed_correct_cases_rejection_and_mismatched_inputs(self):
        result = dict(tesla_correct=1, tesla_macro_f1=.5, tesla_top3_correct=1, other_returned_tesla=0,
                      results=[dict(source_path="synthetic/model_s/image.jpg", expected="model_s", top1="model_s")])
        self.assertTrue(validation.quality_passed(result, result))
        for changes in (dict(other_returned_tesla=1), dict(tesla_top3_correct=0), dict(tesla_macro_f1=.4),
                        dict(results=[dict(source_path="synthetic/model_s/image.jpg", expected="model_s", top1="model_y")]),
                        dict(results=[dict(source_path="different/model_s/image.jpg", expected="model_s", top1="model_s")])):
            with self.subTest(changes=changes):
                self.assertFalse(validation.quality_passed(result, {**result, **changes}))
