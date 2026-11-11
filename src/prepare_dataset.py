"""Prepare reviewed Tesla images and freeze a reproducible split."""

import argparse
from collections import Counter, defaultdict
import csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from src.data_audit import MAX_PIXELS, MODELS, audit_dataset


REVIEW_FIELDS = ("relative_path", "sha256", "class_id", "decision", "vehicle_id")
SPLITS = ("train", "validation", "test")
IMAGE_SIZE = (128, 128)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in fields}
            # Spreadsheet programs must not execute formulas in imported text.
            for key, value in values.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    values[key] = "'" + value
            writer.writerow(values)


def create_review(images, destination):
    fields = (*REVIEW_FIELDS, "issues", "source_url", "license")
    write_csv(destination, fields, (
        {**row, "decision": "pending", "issues": ", ".join(row["issues"])} for row in images
    ))


def plan_splits(images, review, seed):
    candidates = {row["relative_path"]: row for row in images}
    groups = defaultdict(lambda: defaultdict(list))
    vehicle_labels, hashes, reviewed = {}, set(), set()
    with review.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("CSV thiếu cột. Hãy dùng file do lệnh review tạo.")
        for entry in reader:
            relative = (entry["relative_path"] or "").strip()
            if relative not in candidates and relative.startswith("'"):
                relative = relative[1:]
            row = candidates.get(relative)
            if row is None or relative in reviewed or row["sha256"] != entry["sha256"]:
                raise ValueError(f"Ảnh không khớp bản quét hoặc có dòng lặp: {relative}")
            reviewed.add(relative)
            decision = (entry["decision"] or "").strip().lower()
            if decision == "exclude":
                continue
            if decision != "keep":
                raise ValueError(f"Hãy chọn keep hoặc exclude cho ảnh: {relative}")
            label = (entry["class_id"] or "").strip()
            vehicle = (entry["vehicle_id"] or "").strip()
            if label not in MODELS or not vehicle or not row["readable"] or not row["source_url"] or not row["license"]:
                raise ValueError(f"Ảnh giữ lại cần đọc được, có nhãn hợp lệ, nguồn, giấy phép và mã xe: {relative}")
            if vehicle_labels.setdefault(vehicle, label) != label:
                raise ValueError(f"Mã xe dùng cho nhiều dòng: {vehicle}")
            if row["pixel_sha256"] in hashes:
                raise ValueError(f"Ảnh trùng chính xác. Chỉ giữ một bản: {relative}")
            hashes.add(row["pixel_sha256"])
            groups[label][vehicle].append({**row, "class_id": label, "vehicle_id": vehicle})
    if reviewed != candidates.keys():
        raise ValueError("CSV còn thiếu ảnh. Tạo lại bản rà soát cho đúng bộ dữ liệu.")
    if len(groups) < 2:
        raise ValueError("Cần giữ ít nhất hai dòng xe để phân loại.")

    rng = random.Random(seed)
    rows = []
    for label, vehicles in sorted(groups.items()):
        ids = sorted(vehicles)
        if len(ids) < 3:
            raise ValueError(f"{MODELS[label]} cần ít nhất ba nhóm xe để có mặt ở cả ba tập.")
        rng.shuffle(ids)
        holdout = max(1, round(len(ids) * 0.15))
        for index, vehicle in enumerate(ids):
            split = "test" if index < holdout else "validation" if index < 2 * holdout else "train"
            for row in sorted(vehicles[vehicle], key=lambda item: item["relative_path"]):
                rows.append({**row, "split": split})
    return rows


def prepare(root, output, rows):
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows, 1):
        relative = Path(row["split"], row["class_id"], f"{index:05}.png")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = (root / row["relative_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Ảnh nằm ngoài thư mục nguồn: {row['relative_path']}")
        with Image.open(source) as original:
            if original.width * original.height > MAX_PIXELS:
                raise ValueError(f"Ảnh vượt giới hạn kích thước: {row['relative_path']}")
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            ImageOps.pad(rgb, IMAGE_SIZE, method=Image.Resampling.LANCZOS, color="black").save(destination)
        row["source_path"] = row["relative_path"]
        row["relative_path"] = relative.as_posix()
    write_csv(output / "splits.csv", (
        "relative_path", "source_path", "class_id", "vehicle_id", "split",
        "sha256", "source_url", "license",
    ), rows)
    counts = Counter((row["class_id"], row["split"]) for row in rows)
    write_csv(output / "summary.csv", ("class_id", *SPLITS), (
        {"class_id": label, **{split: counts[label, split] for split in SPLITS}}
        for label in sorted({row["class_id"] for row in rows})
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Tạo CSV rà ảnh.")
    split = commands.add_parser("split", help="Chia tập và chuẩn hóa ảnh đã duyệt.")
    for command in (review, split):
        command.add_argument("dataset", type=Path)
        command.add_argument("--output", required=True, type=Path)
    split.add_argument("--review", required=True, type=Path)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        root, output = args.dataset.resolve(), args.output.resolve()
        processed = (Path(__file__).resolve().parents[1] / "data" / "processed").resolve()
        if output == processed or not output.is_relative_to(processed) or output.is_relative_to(root):
            raise ValueError("Đặt đầu ra bên trong data/processed, ngoài thư mục ảnh nguồn.")
        if output.exists():
            raise ValueError("Đầu ra đã tồn tại. Chọn tên mới để giữ kết quả cũ.")
        report = audit_dataset(root)
        if report["summary"]["skipped_external_links"]:
            raise ValueError("Dataset chứa liên kết ảnh ra ngoài thư mục nguồn.")
        images = report["images"]
        if not images:
            raise ValueError("Không tìm thấy ảnh để xử lý.")
        if args.command == "review":
            create_review(images, output)
            print(f"Review CSV ({len(images)} images): {output}")
        else:
            rows = plan_splits(images, args.review, args.seed)
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
                staging = Path(temporary) / "dataset"
                prepare(root, staging, rows)
                staging.rename(output)
            print(f"Prepared {len(rows)} images (seed={args.seed}): {output}")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        parser.exit(2, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
