"""Read-only Tesla image audit, shared by the CLI and Streamlit page."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


MODELS = {
    "model_s": "Model S", "model_3": "Model 3", "model_x": "Model X",
    "model_y": "Model Y", "cybertruck": "Cybertruck",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
# Bound decoded memory; review larger originals separately before training.
MAX_PIXELS = 25_000_000


def model_label(folder):
    normalized = folder.lower().replace(" ", "").replace("_", "").replace("-", "")
    return next((key for key in MODELS if key.replace("_", "") == normalized), "")


def load_provenance(root):
    """Optional per-image records; malformed manifests fail instead of disappearing."""
    path = root / "provenance.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "source_url", "license", "vehicle_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("provenance.csv cần các cột: " + ", ".join(sorted(required)))
        records = {}
        for number, row in enumerate(reader, 2):
            relative = (row.get("relative_path") or "").strip().replace("\\", "/")
            target = (root / relative).resolve()
            if not relative or not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"provenance.csv dòng {number}: đường dẫn ảnh không hợp lệ.")
            relative = target.relative_to(root).as_posix()
            if relative in records:
                raise ValueError(f"provenance.csv dòng {number}: ảnh đã có bản ghi.")
            records[relative] = {key: (row.get(key) or "").strip() for key in required}
        return records


def audit_dataset(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Thư mục dữ liệu không tồn tại.")
    provenance = load_provenance(root)
    rows, skipped = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        # Do not follow symlinks that escape the selected dataset.
        if not path.resolve().is_relative_to(root):
            skipped += 1
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        split = SPLITS.get(parts[0].lower(), "unsplit")
        folder_index = 0 if split == "unsplit" else 1
        folder = parts[folder_index] if len(parts) > folder_index + 1 else ""
        record = provenance.get(relative.as_posix(), {})
        row = {
            "relative_path": relative.as_posix(), "folder_label": folder,
            "class_id": model_label(folder), "split": split, "readable": False,
            "width": None, "height": None, "sha256": "", "pixel_sha256": "",
            "source_url": record.get("source_url", ""),
            "license": record.get("license", ""), "vehicle_id": record.get("vehicle_id", ""),
            "issues": [],
        }
        if not folder:
            row["issues"].append("missing_label")
        elif not row["class_id"]:
            row["issues"].append("unknown_label")
        if not row["source_url"] or not row["license"]:
            row["issues"].append("missing_provenance")
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            row["sha256"] = digest.hexdigest()
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as original:
                    if original.width * original.height > MAX_PIXELS:
                        raise ValueError("Image exceeds the audit memory limit")
                    original.verify()
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert("RGBA")
                    image.load()
                    row["width"], row["height"] = image.size
                    pixel_hash = hashlib.sha256(str(image.size).encode())
                    pixel_hash.update(image.tobytes())
                    row["pixel_sha256"] = pixel_hash.hexdigest()
            row["readable"] = True
        except (OSError, ValueError, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            row["issues"].append("unreadable")
            row["error"] = type(error).__name__
        rows.append(row)

    duplicate_groups, leakage_groups = [], []
    for field in ("sha256", "pixel_sha256", "vehicle_id"):
        groups = defaultdict(list)
        for row in rows:
            if row[field]:
                groups[row[field]].append(row)
        for key, members in groups.items():
            if len(members) < 2:
                continue
            paths = [row["relative_path"] for row in members]
            if field != "vehicle_id":
                duplicate_groups.append({"kind": field, "files": paths})
                for row in members:
                    if "duplicate" not in row["issues"]:
                        row["issues"].append("duplicate")
                if len({row["class_id"] for row in members}) > 1:
                    for row in members:
                        if "conflicting_labels" not in row["issues"]:
                            row["issues"].append("conflicting_labels")
            assigned = {row["split"] for row in members} - {"unsplit"}
            if len(assigned) > 1:
                leakage_groups.append({"kind": field, "files": paths, "splits": sorted(assigned)})
                for row in members:
                    if "split_leakage" not in row["issues"]:
                        row["issues"].append("split_leakage")

    counts = Counter(row["class_id"] for row in rows if row["class_id"] and row["readable"])
    issues = Counter(issue for row in rows for issue in row["issues"])
    return {
        "dataset_name": root.name,
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "images": len(rows), "readable": sum(row["readable"] for row in rows),
            "recognized_classes": len(counts), "issues": dict(issues),
            "duplicate_images": issues["duplicate"], "leakage_images": issues["split_leakage"],
            "skipped_external_links": skipped,
        },
        "classes": [{"class_id": key, "model": name, "readable_images": counts[key]} for key, name in MODELS.items()],
        "images": rows, "duplicate_groups": duplicate_groups, "leakage_groups": leakage_groups,
        "limitations": [
            "Folder names are candidate labels, not verified vehicle identities.",
            "Source/license entries are recorded, not legally or factually verified.",
            "Duplicate checks cover identical bytes or decoded pixels; resized/cropped images need visual review.",
            "Same-vehicle leakage can only be checked when vehicle_id is provided.",
            "No model training, automatic splitting or source-image modification is performed.",
        ],
    }


def report_csv(report):
    buffer = io.StringIO(newline="")
    fields = ["relative_path", "folder_label", "class_id", "split", "readable", "width", "height",
              "sha256", "pixel_sha256", "source_url", "license", "vehicle_id", "issues"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in report["images"]:
        exported = {**row, "issues": ";".join(row["issues"])}
        # Spreadsheet programs must not interpret user-supplied metadata as formulas.
        for key, value in exported.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                exported[key] = "'" + value
        writer.writerow(exported)
    return buffer.getvalue().encode("utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Audit Tesla image folders without modifying them.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/audit"))
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.directory.resolve()):
            raise ValueError("Đặt thư mục báo cáo bên ngoài thư mục ảnh gốc.")
        report = audit_dataset(args.directory)
        args.output.mkdir(parents=True, exist_ok=True)
        # A separate run directory preserves reports from previous scans.
        output = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir()
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "images.csv").write_bytes(report_csv(report))
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"Report: {output.resolve()}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
