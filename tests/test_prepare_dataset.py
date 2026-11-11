import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from src.data_audit import audit_dataset
from src.prepare_dataset import REVIEW_FIELDS, create_review, plan_splits, prepare, write_csv


class PrepareDatasetTests(unittest.TestCase):
    def test_review_gate_and_grouped_preprocessing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            provenance = []
            for class_index, label in enumerate(("model_3", "model_y")):
                (root / label).mkdir(parents=True)
                for index in range(4):
                    relative = f"{label}/{index}.png"
                    Image.new("RGB", (30, 18), (class_index, index, 0)).save(root / relative)
                    provenance.append(dict(relative_path=relative, source_url="https://example.org/photo",
                                           license="CC BY 4.0", vehicle_id=f"{label}-{min(index, 2)}"))
            write_csv(root / "provenance.csv", tuple(provenance[0]), provenance)
            images = audit_dataset(root)["images"]
            review = Path(temporary) / "review.csv"
            create_review(images, review)
            with self.assertRaisesRegex(ValueError, "keep hoặc exclude"):
                plan_splits(images, review, 42)
            with review.open(encoding="utf-8-sig", newline="") as handle:
                approved = [{**row, "decision": "keep"} for row in csv.DictReader(handle)]
            approved_path = Path(temporary) / "approved.csv"
            write_csv(approved_path, REVIEW_FIELDS, approved)
            rows = plan_splits(images, approved_path, 42)
            self.assertEqual(rows, plan_splits(images, approved_path, 42))
            for label in ("model_3", "model_y"):
                self.assertEqual({row["split"] for row in rows if row["class_id"] == label},
                                 {"train", "validation", "test"})
                self.assertEqual(len({row["split"] for row in rows if row["vehicle_id"] == f"{label}-2"}), 1)
            output = Path(temporary) / "processed"
            prepare(root, output, rows)
            self.assertEqual(len(list(output.rglob("*.png"))), 8)
            self.assertTrue((output / "splits.csv").is_file())
            with Image.open(output / rows[0]["relative_path"]) as image:
                self.assertEqual((image.mode, image.size), ("RGB", (128, 128)))
            self.assertEqual(audit_dataset(root)["images"], images)

            approved[0]["sha256"] = "changed"
            stale_path = Path(temporary) / "stale.csv"
            write_csv(stale_path, REVIEW_FIELDS, approved)
            with self.assertRaisesRegex(ValueError, "không khớp"):
                plan_splits(images, stale_path, 42)
