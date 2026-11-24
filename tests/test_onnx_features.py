"""Synthetic runtime fixtures; no real images, weights or accuracy claims."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from src.onnx_features import OnnxFeatures, detector_pixels, file_hash, nms, preprocess
from src.vision_features import DINO, SIGLIP


class Session:
    def __init__(self, filename):
        self.detector = Path(filename).stem == "detector"
        self.bad = False

    def get_inputs(self):
        size = 320 if self.detector else 224
        return [SimpleNamespace(name="pixels", shape=[1, 3, size, size])]

    def run(self, outputs, inputs):
        if not self.detector:
            return [np.ones((1, 768), np.float32)]
        offsets = np.zeros((1, 2, 4), np.float32)
        logits = np.full((1, 2, 91), -100, np.float32)
        logits[0, 0, 3] = 100
        logits[0, 1, 0] = 100
        if self.bad:
            offsets[0, 0, 0] = np.nan
        return offsets, logits


class OnnxTests(unittest.TestCase):
    def fixture(self, root):
        model = root / "data/processed/model"
        runtime = root / "data/processed/runtime"
        model.mkdir(parents=True)
        runtime.mkdir(parents=True)
        for name in ("model.json", "classifier.npz"):
            (model / name).write_bytes(b"synthetic model marker")
        for name in ("dino.onnx", "siglip.onnx", "detector.onnx"):
            (runtime / name).write_bytes(b"synthetic graph marker")
        np.savez(runtime / "anchors.npz", anchors=np.asarray([[0, 0, 320, 320], [0, 0, 10, 10]], np.float32))
        config = dict(schema_version=1, source_dino=list(DINO), source_siglip=list(SIGLIP),
                      precision="weight_only_int8", box_weights=[10., 10., 5., 5.],
                      nms_thresh=.55, topk_candidates=300, detections_per_img=300,
                      bbox_xform_clip=4.135166556742356, model_metadata_sha256=file_hash(model / "model.json"),
                      classifier_sha256=file_hash(model / "classifier.npz"),
                      files={p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in runtime.iterdir()})
        (runtime / "runtime.json").write_text(json.dumps(config))
        (runtime / "validation.json").write_text(json.dumps(dict(passed=True, runtime_manifest_sha256=file_hash(runtime / "runtime.json"))))
        return model, runtime

    def test_preprocessing_and_nms_are_bounded_and_deterministic(self):
        image = Image.new("RGB", (640, 480), "white")
        for encoder in ("dino", "siglip"):
            array = preprocess(image, encoder)
            self.assertEqual(array.shape, (1, 3, 224, 224))
            self.assertEqual(array.dtype, np.float32)
            self.assertTrue(np.isfinite(array).all())
        np.testing.assert_allclose(preprocess(image, "siglip"), 1)
        with self.assertRaises(ValueError): preprocess(image, "other")
        for size in ((1, 1), (2, 3), (640, 480)):
            values = detector_pixels(Image.new("RGB", size, "white"))
            self.assertEqual(values.shape, (1, 3, 320, 320))
            np.testing.assert_allclose(values, 1)
        boxes = np.asarray([[0, 0, 10, 10], [0, 0, 9, 9], [20, 20, 30, 30]], np.float32)
        np.testing.assert_array_equal(nms(boxes, np.asarray([.9, .8, .7]), .55), [0, 2])

    def test_checksums_model_coupling_and_validation_fail_before_loading_graphs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model, runtime = self.fixture(root)
            fake = SimpleNamespace(SessionOptions=Mock, InferenceSession=Mock(side_effect=lambda filename, **kw: Session(filename)))
            with patch("src.onnx_features.ROOT", root), patch.dict("sys.modules", {"onnxruntime": fake}):
                engine = OnnxFeatures(model, runtime)
                values = engine.encode([Image.new("RGB", (20, 20))], "dino")
                self.assertEqual(values.shape, (1, 768))
                self.assertAlmostEqual(float(np.linalg.norm(values)), 1, places=6)
                detections = engine.detect(Image.new("RGB", (640, 480)))
                self.assertEqual(detections[0]["box"], [0, 0, 640, 480])
                engine.sessions["detector"].bad = True
                with self.assertRaises(ValueError): engine.detect(Image.new("RGB", (640, 480)))
                fake.InferenceSession.reset_mock()
                (runtime / "dino.onnx").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"): OnnxFeatures(model, runtime)
                fake.InferenceSession.assert_not_called()
                (runtime / "dino.onnx").write_bytes(b"synthetic graph marker")
                (model / "classifier.npz").write_bytes(b"different classifier")
                with self.assertRaisesRegex(ValueError, "cùng phiên bản"): OnnxFeatures(model, runtime)
                (runtime / "validation.json").write_text('{"passed":false}')
                with self.assertRaisesRegex(ValueError, "validation"): OnnxFeatures(model, runtime)
                fake.InferenceSession.assert_not_called()
                with self.assertRaises(ValueError): OnnxFeatures(model, root / "outside")
