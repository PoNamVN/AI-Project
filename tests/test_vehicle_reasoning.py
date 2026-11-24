import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from src.data_audit import MODELS
from src.recognize_vehicle import Recognizer, occlusion_tiles
from src.vehicle_reasoning import (
    brand_margin,
    fit_ridge,
    probabilities,
    ridge_scores,
    summary,
)
from src.vision_features import DINO, SIGLIP, open_image, padded_box

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    detections = [{"box": [0, 0, 32, 24], "area": 768, "detector_score": 0.9}]
    other = False
    weak = False

    def detect(self, image):
        return self.detections

    def modes(self, image, crop, mode=None):
        return {
            "dino_crop": np.asarray([1.0, 0.0]),
            "siglip_crop": np.asarray(
                [0.0, 1.0, 0.0]
                if self.other
                else [0.99, 1.0, 0.0]
                if self.weak
                else [1.0, 0.0, 0.0]
            ),
        }

    def selected(self, images, mode):
        return np.tile([1.0, 0.0], (len(images), 1))


class ReasoningTests(unittest.TestCase):
    def test_ridge_class_balance_and_numeric_safety(self):
        x = np.repeat(np.eye(5), 3, axis=0)
        y = np.repeat(np.arange(5), 3)
        w = fit_ridge(x, y, 5, 0.1)
        s = ridge_scores(x, w)
        self.assertEqual(summary(y, s)["correct"], 15)
        self.assertEqual(summary(y, s)["top3_correct"], 15)
        for features, targets, reg in [
            (x, y, -1),
            (x, np.zeros(15, dtype=int), 1),
            (x[:3], y, 1),
            (x * np.nan, y, 1),
        ]:
            with self.assertRaises(ValueError):
                fit_ridge(features, targets, 5, reg)
        with self.assertRaises(ValueError):
            ridge_scores(x, w[:2])
        p = probabilities([[10000.0, -10000.0, 0.0]])
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(p.sum(), 1)

    def test_brand_margin_does_not_force_tesla(self):
        texts = np.eye(3)
        groups = ["tesla", "other_car", "not_car"]
        self.assertGreater(brand_margin([1, 0, 0], texts, groups)[0], 0)
        self.assertLess(brand_margin([0, 1, 0], texts, groups)[0], 0)

    def test_image_bounds_and_occlusion_preserve_input(self):
        original = Image.new("RGB", (32, 24), "blue")
        stream = io.BytesIO()
        original.save(stream, format="PNG")
        self.assertEqual(open_image(stream.getvalue()).size, (32, 24))
        with patch("src.vision_features.MAX_PIXELS", 10):
            with self.assertRaises(ValueError):
                open_image(stream.getvalue())
        self.assertEqual(padded_box([-1, -2, 30, 20], (32, 24)), (0, 0, 32, 22))
        for box in ([3, 2, 1, 4], [0, 0, float("nan"), 3], [100, 100, 200, 200]):
            with self.assertRaises(ValueError):
                padded_box(box, (32, 24))
        before = original.tobytes()
        tiles, boxes = occlusion_tiles(original)
        self.assertEqual(len(tiles), 9)
        self.assertEqual(len(boxes), 9)
        self.assertEqual(original.tobytes(), before)

    def test_recognition_rejection_multiple_choice_and_real_evidence_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/processed") as directory:
            run = Path(directory)
            labels = sorted(MODELS)
            records = [
                dict(
                    source_path=f"synthetic/{label}/fixture.png",
                    class_id=label,
                    source_url="https://example.org/test",
                    license="synthetic fixture",
                    artist_html="Test fixture only",
                    license_url="https://example.org/license",
                )
                for label in labels
            ]
            meta = dict(
                schema_version=1,
                class_ids=labels,
                feature_mode="dino_crop",
                dino=list(DINO),
                siglip=list(SIGLIP),
                classifier="ridge",
                brand_threshold=0.0,
                class_gap_threshold=0.08,
                train_records=records,
                brand_groups=["tesla", "other_car", "not_car"],
            )
            (run / "model.json").write_text(json.dumps(meta))
            weights = np.asarray(
                [[0.1, 0.9, 0.2, 0.1, 0.2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                dtype=np.float32,
            )
            np.savez_compressed(
                run / "classifier.npz",
                weights=weights,
                train_features=np.tile([1.0, 0.0], (5, 1)),
                brand_text=np.eye(3),
                class_text=np.ones((5, 3)),
                probe_text=np.ones((8, 3)),
            )
            engine = FakeEngine()
            recognizer = Recognizer(run, engine)
            stream = io.BytesIO()
            Image.new("RGB", (32, 24), "blue").save(stream, format="PNG")
            data = stream.getvalue()
            result, annotated, heatmap = recognizer.analyze(data)
            self.assertEqual(result["candidates"][0]["class_id"], "model_3")
            self.assertEqual(result["status"], "tesla_candidate")
            self.assertEqual([r["class_id"] for r in result["candidate_references"]],
                             [c["class_id"] for c in result["candidates"]])
            self.assertEqual(len({r["class_id"] for r in result["candidate_references"]}), 3)
            self.assertIsNone(result["year"])
            self.assertTrue(result["review_required"])
            self.assertIsNotNone(annotated)
            self.assertIsNotNone(heatmap)
            self.assertEqual(result["occlusion"]["score_drops"], [0] * 9)
            self.assertTrue(
                all(
                    p["status"].endswith("not_verified_part")
                    for p in result["visual_hypotheses"]
                )
            )
            engine.other = True
            result, _, _ = recognizer.analyze(data)
            self.assertEqual(result["status"], "outside_or_uncertain")
            self.assertEqual(result["candidates"], [])
            engine.other = False
            engine.weak = True
            recognizer.metadata["brand_threshold"] = -0.05
            result, _, _ = recognizer.analyze(data, explain=False)
            self.assertEqual(result["status"], "uncertain_brand")
            engine.weak = False
            engine.detections = []
            result, _, _ = recognizer.analyze(data)
            self.assertEqual(result["status"], "no_vehicle")
            engine.detections = [
                {"box": [0, 0, 32, 24], "area": 768, "detector_score": 0.9}
            ] * 2
            result, _, _ = recognizer.analyze(data)
            self.assertEqual(result["status"], "multiple_vehicles")
            result, _, _ = recognizer.analyze(data, 1, False)
            self.assertEqual(result["status"], "tesla_candidate")
            with self.assertRaises(ValueError):
                recognizer.analyze(data, 5)
            engine.detections = engine.detections[:1]
            recognizer.brand_weights = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            engine.other = True
            result, _, _ = recognizer.analyze(data, explain=False)
            self.assertEqual(result["brand_evidence"]["method"], "ridge")
            self.assertEqual(result["status"], "tesla_candidate")
            recognizer.brand_weights *= -1
            result, _, _ = recognizer.analyze(data, explain=False)
            self.assertEqual(result["status"], "outside_or_uncertain")


if __name__ == "__main__":
    unittest.main()
