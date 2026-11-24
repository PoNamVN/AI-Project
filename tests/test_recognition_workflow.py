import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.recognition_workflow import compare_views, reference_preview, reviewed_result


class WorkflowTests(unittest.TestCase):
    def test_second_view_compares_without_fusing_scores_or_overriding_uncertainty(self):
        def result(label, status="tesla_candidate"):
            return dict(status=status, candidates=[dict(class_id=label, ranking_score=.71)])
        first = result("model_3")
        self.assertEqual(compare_views(first, result("model_3"))["status"], "agreement")
        self.assertEqual(compare_views(first, result("model_y"))["status"], "disagreement")
        self.assertEqual(compare_views(first, result("model_3", "uncertain_model"))["status"], "same_but_uncertain")
        for status in ("uncertain_brand", "outside_or_uncertain", "no_vehicle", "multiple_vehicles"):
            self.assertEqual(compare_views(first, result("model_3", status))["status"], "insufficient")
        self.assertEqual(first["candidates"][0]["ranking_score"], .71)
        self.assertNotIn("comparison", first)

    def setUp(self):
        stream = io.BytesIO()
        Image.new("RGB", (640, 400), "blue").save(stream, format="PNG")
        self.data = stream.getvalue()
        self.reference = dict(source_path="fixture-v1/model_3/car.png", class_id="model_3",
                              sha256=hashlib.sha256(self.data).hexdigest())

    def test_local_and_cloud_reference_are_bounded_and_checksum_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / self.reference["source_path"]
            photo.parent.mkdir(parents=True)
            photo.write_bytes(self.data)
            preview = reference_preview(self.reference, root)
            self.assertEqual(preview.width, 500)
            self.assertLessEqual(preview.height, 320)
            self.assertAlmostEqual(preview.width / preview.height, 640 / 400, delta=.01)
            photo.write_bytes(self.data + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                reference_preview(self.reference, root)
            with patch("src.recognition_workflow.CloudData") as cloud:
                cloud.return_value.reference_image.return_value = self.data
                self.assertEqual(reference_preview(self.reference, root, True).width, 500)
                cloud.return_value.reference_image.assert_called_once_with(self.reference["sha256"], "model_3", thumbnail=True)
            for source in ("../../outside.png", "/fixture/model_3/x.png", "fixture/model_y/x.png", "fixture/model_3/../x.png", "C:/model_3/x.png", None):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    reference_preview({**self.reference, "source_path": source}, root)

    def test_manual_review_keeps_prediction_and_never_becomes_a_verified_label(self):
        original = dict(candidates=[dict(class_id="model_3", model="Model 3")], year=None)
        corrected = reviewed_result(original, "model_y")
        self.assertEqual(corrected["candidates"], original["candidates"])
        self.assertNotIn("user_review", original)
        self.assertEqual(corrected["user_review"]["class_id"], "model_y")
        self.assertEqual(corrected["user_review"]["verification"], "user_choice_not_verified")
        self.assertIsNone(corrected["user_review"]["specifications"])
        self.assertEqual(reviewed_result(original, None)["user_review"]["decision"], "no_match")
        for result, label in ((original, "bmw"), ({"candidates": []}, "model_3")):
            with self.assertRaises(ValueError):
                reviewed_result(result, label)
