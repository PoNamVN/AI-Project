import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import hashlib
import subprocess
import sys
import io
import json
import shutil
from PIL import Image

from src.evaluate_field import evaluate


class FieldTests(unittest.TestCase):
    def test_validation_only_needs_no_model_weights_and_produces_no_accuracy(self):
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'src').mkdir()
            for filename in ('__init__.py', 'evaluate_field.py', 'data_audit.py', 'vision_features.py'):
                shutil.copy2(source_root / 'src' / filename, root / 'src' / filename)
            base, raw = root / 'data/processed/fixture', root / 'data/raw/fixture'
            base.mkdir(parents=True)
            raw.mkdir(parents=True)
            model = base / "model"
            model.mkdir()
            (model / "model.json").write_text(json.dumps({"train_records": [], "manifest_sha256": "test"}))
            photo = raw / "model_3" / "test.png"
            photo.parent.mkdir()
            Image.new("RGB", (29, 23), "purple").save(photo)
            manifest = base / "batch.csv"
            manifest.write_text("source_path,class_id,vehicle_id,source_url,license\n" + f"{raw.name}/model_3/test.png,model_3,test-car,synthetic,fixture\n")
            output = base / "checked"
            command = [sys.executable, "-m", "src.evaluate_field", str(manifest), "--model", str(model), "--output", str(output), "--labels-reviewed", "--validate-only"]
            result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            for name in ('tesla-expanded-v2/splits.csv', 'week4-newcheck-v1/review.csv',
                         'non-tesla-gate-review-v1/review.csv', 'non-tesla-brand-training-v3/review.csv'):
                history = root / 'data/processed' / name
                history.parent.mkdir(parents=True)
                history.write_text('sha256\n', encoding='utf-8')
            result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertFalse(report["model_inference_performed"])
            self.assertEqual(report["images"], 1)
            self.assertNotIn("accuracy", report)
            self.assertFalse((output / "metrics.json").exists())

    def test_cli_refuses_overwrite_before_loading_model(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "data/processed") as directory:
            output = Path(directory)
            marker = output / "keep.txt"
            marker.write_text("keep")
            manifest = output / "manifest.csv"
            manifest.write_text("source_path,class_id,vehicle_id,source_url,license\n")
            result = subprocess.run([sys.executable, "-m", "src.evaluate_field", str(manifest), "--output", str(output), "--labels-reviewed"],
                                    cwd=root, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), "keep")
            self.assertFalse((output / "metrics.json").exists())

    def test_unresolved_inputs_stay_in_denominator_and_other_cars_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, label in enumerate(("model_3", "model_y", "other_car")):
                source = f"batch/{label}/{i}.png"
                path = root / source
                path.parent.mkdir(parents=True)
                buffer = io.BytesIO()
                Image.new("RGB", (18, 16), (i * 80, 20, 30)).save(buffer, format="PNG")
                path.write_bytes(buffer.getvalue())
                rows.append(dict(source_path=source, class_id=label, vehicle_id=str(i), source_url="self-captured", license="owner-authorized"))
            model = Mock()
            model.analyze.side_effect = [
                ({"status": "tesla_candidate", "candidates": [{"class_id": "model_3"}]}, None, None),
                ({"status": "no_vehicle", "candidates": []}, None, None),
                ({"status": "outside_or_uncertain", "candidates": []}, None, None),
            ]
            report = evaluate(rows, model, root, set())
            self.assertEqual(report["tesla_top1_over_all_inputs"], .5)
            self.assertEqual(report["other_returned_tesla"], 0)
            self.assertEqual(report["vehicles_declared"], 3)
            model.reset_mock()
            for altered, known in [
                ([rows[0], rows[0]], set()),
                ([{**rows[0], "source_path": "../outside.png"}], set()),
                ([rows[0]], {hashlib.sha256((root / rows[0]["source_path"]).read_bytes()).hexdigest()}),
                ([{**rows[0], "license": ""}], set()),
            ]:
                with self.assertRaises(ValueError):
                    evaluate(altered, model, root, known)
                model.analyze.assert_not_called()
            (root / rows[0]["source_path"]).write_bytes(b"not an image")
            with self.assertRaises(OSError):
                evaluate([rows[0]], model, root, set())
            model.analyze.assert_not_called()
