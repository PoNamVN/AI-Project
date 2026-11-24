"""Frozen split validation shared by the TensorFlow and PyTorch experiments."""

from collections import defaultdict
import csv
from src.data_audit import MODELS
from src.prepare_dataset import SPLITS


def load_manifest(root):
    rows = defaultdict(list)
    vehicles, hashes, paths = {}, set(), set()
    with (root / "splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "class_id", "vehicle_id", "sha256", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("splits.csv is missing required columns")
        for row in reader:
            split, label = row["split"], row["class_id"]
            relative = row["relative_path"] or ""
            path = (root / relative).resolve()
            if (
                split not in SPLITS
                or label not in MODELS
                or not path.is_relative_to(root)
                or path.parent != root / split / label
                or not path.is_file()
                or path in paths
            ):
                raise ValueError(f"Invalid or repeated split entry: {relative}")
            vehicle, digest = row["vehicle_id"], row["sha256"]
            if (
                not vehicle
                or not digest
                or vehicles.setdefault(vehicle, (split, label)) != (split, label)
            ):
                raise ValueError(
                    f"Vehicle appears in multiple splits or classes: {relative}"
                )
            if digest in hashes:
                raise ValueError(f"Repeated original image: {relative}")
            hashes.add(digest)
            paths.add(path)
            rows[split].append((relative, path, label))
    labels = sorted({label for _, _, label in rows["train"]})
    if len(labels) < 2 or any(
        {label for _, _, label in rows[split]} != set(labels) for split in SPLITS
    ):
        raise ValueError("Each split must contain the same two or more classes")
    return rows, labels
