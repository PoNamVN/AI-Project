import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from src.data_audit import audit_dataset, report_csv


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, color="red", size=(32, 24)):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    def test_empty_and_missing_directory(self):
        self.assertEqual(audit_dataset(self.root)["summary"]["images"], 0)
        with self.assertRaises(ValueError):
            audit_dataset(self.root / "missing")

    def test_bad_files_labels_and_pixel_duplicates(self):
        original = self.image("Model 3/a.png")
        duplicate = self.root / "model_3/copy.png"
        duplicate.parent.mkdir()
        duplicate.write_bytes(original.read_bytes())
        self.image("model_y/same-pixels.bmp")
        self.image("unrecognised/other.png", "blue")
        self.image("loose.png", "green")
        (original.parent / "broken.jpg").write_bytes(b"not an image")
        report = audit_dataset(self.root)
        self.assertEqual(report["summary"]["images"], 6)
        self.assertEqual(report["summary"]["readable"], 5)
        self.assertEqual(report["summary"]["duplicate_images"], 3)
        self.assertEqual(report["summary"]["issues"]["missing_label"], 1)
        self.assertEqual(report["summary"]["issues"]["unknown_label"], 1)
        self.assertEqual(report["summary"]["issues"]["conflicting_labels"], 3)

    def test_split_leakage_by_pixels_and_vehicle(self):
        self.image("train/model_s/a.png")
        self.image("test/model_s/same.png")
        self.image("train/model_y/front.png", "blue")
        self.image("val/model_y/back.png", "green")
        (self.root / "provenance.csv").write_text(
            "relative_path,source_url,license,vehicle_id\n"
            "train/model_y/front.png,https://example.org/front,CC-BY,same-car\n"
            "val/model_y/back.png,https://example.org/back,CC-BY,same-car\n", encoding="utf-8")
        report = audit_dataset(self.root)
        self.assertEqual(report["summary"]["leakage_images"], 4)
        self.assertTrue(any(group["kind"] == "vehicle_id" for group in report["leakage_groups"]))

    def test_invalid_provenance_and_csv_formula(self):
        self.image("model_x/a.png")
        manifest = self.root / "provenance.csv"
        manifest.write_text("relative_path,source_url,license,vehicle_id\n../escape.png,x,y,z\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit_dataset(self.root)
        manifest.write_text("relative_path,source_url,license,vehicle_id\nmodel_x/a.png,=BAD(),unknown,\n", encoding="utf-8")
        report = audit_dataset(self.root)
        exported = list(csv.DictReader(io.StringIO(report_csv(report).decode("utf-8-sig"))))
        self.assertEqual(exported[0]["source_url"], "'=BAD()")

    def test_decoded_memory_limit(self):
        self.image("model_s/large.png")
        with patch("src.data_audit.MAX_PIXELS", 100):
            report = audit_dataset(self.root)
        self.assertEqual(report["summary"]["readable"], 0)
        self.assertEqual(report["summary"]["issues"]["unreadable"], 1)

    def test_cli_report_and_input_protection(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        path = dataset / "model_3"
        path.mkdir()
        Image.new("RGB", (8, 8), "white").save(path / "test.png")
        command = [sys.executable, "-m", "src.data_audit", str(dataset)]
        result = subprocess.run(command + ["--output", str(self.root / "reports")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = next((self.root / "reports").glob("*/audit.json"))
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["summary"]["images"], 1)
        self.assertTrue(output.with_name("images.csv").exists())
        result = subprocess.run(command + ["--output", str(dataset / "reports")], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
