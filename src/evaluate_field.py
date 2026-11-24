"""Evaluate a frozen, manually labeled field batch without fitting the model."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import sys

from src.data_audit import MODELS

ROOT = Path(__file__).resolve().parents[1]


def validate_rows(rows, raw_root, known_hashes):
    if not 1 <= len(rows) <= 1000:
        raise ValueError("Cần từ 1 đến 1000 ảnh có nhãn đã rà.")
    root = Path(raw_root).resolve()
    checked, hashes, paths = [], set(), set()
    for row in rows:
        source, label = row.get("source_path", ""), row.get("class_id", "")
        if not isinstance(source, str) or not isinstance(label, str) or label not in {*MODELS, "other_car"}:
            raise ValueError("Nhãn hoặc đường dẫn ảnh không hợp lệ.")
        parts = PurePosixPath(source).parts
        if (len(parts) != 3 or parts[1] != label or ".." in parts
                or "\\" in source or ":" in source or source in paths
                or not row.get("vehicle_id") or not row.get("source_url") or not row.get("license")):
            raise ValueError("Thiếu nguồn/quyền ảnh/mã xe hoặc đường dẫn bị trùng, không hợp lệ.")
        path = (root / source).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Ảnh kiểm tra nằm ngoài data/raw.")
        with path.open("rb") as handle:
            data = handle.read(30_000_001)
        if len(data) > 30_000_000:
            raise ValueError("Ảnh kiểm tra vượt 30 MB.")
        digest = hashlib.sha256(data).hexdigest()
        if digest in hashes or digest in known_hashes:
            raise ValueError("Ảnh trùng hoặc đã xuất hiện trong dữ liệu phát triển; không coi là ảnh mới.")
        from src.vision_features import open_image
        with open_image(data):
            pass
        hashes.add(digest)
        paths.add(source)
        checked.append((row, path, digest))
    return checked


def evaluate(rows, recognizer, raw_root, known_hashes):
    checked = validate_rows(rows, raw_root, known_hashes)
    # Validate the whole manifest before spending model time on any image.
    results = []
    for row, path, digest in checked:
        result, _, _ = recognizer.analyze(path, None, False)
        candidates = result.get("candidates", [])
        top1 = candidates[0]["class_id"] if candidates else None
        results.append({"source_path": row["source_path"], "sha256": digest,
                        "vehicle_id": row["vehicle_id"], "expected": row["class_id"],
                        "status": result["status"], "top1": top1,
                        "top3": [c["class_id"] for c in candidates],
                        "correct_top1": top1 == row["class_id"]})
    tesla = [r for r in results if r["expected"] in MODELS]
    others = [r for r in results if r["expected"] == "other_car"]
    return {"images": len(results), "vehicles_declared": len({r["vehicle_id"] for r in results}),
            "labels": dict(Counter(r["expected"] for r in results)),
            "statuses": dict(Counter(r["status"] for r in results)),
            "tesla_inputs": len(tesla), "tesla_top1_correct": sum(r["correct_top1"] for r in tesla),
            "tesla_top1_over_all_inputs": sum(r["correct_top1"] for r in tesla) / len(tesla) if tesla else None,
            "tesla_top3_correct": sum(r["expected"] in r["top3"] for r in tesla),
            "other_inputs": len(others), "other_returned_tesla": sum(r["top1"] is not None for r in others),
            "limitations": ["Exact-hash overlap checked; changed crops and physical-vehicle independence are not proven.",
                            "Labels and vehicle IDs are supplied by the reviewer; no training or tuning performed.",
                            "Single-input pipeline evaluation; not a benchmark of two-view comparison."],
            "results": results}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "data/processed/vision-recognition-v3-brand")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-reviewed", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Kiểm tra ảnh và manifest, không nạp mô hình hoặc tạo accuracy")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        model = args.model.resolve()
        if (output.exists() or not output.is_relative_to((ROOT / "data/processed").resolve())
                or not model.is_relative_to((ROOT / "data/processed").resolve())
                or args.manifest.stat().st_size > 1_000_000 or not args.labels_reviewed):
            raise ValueError("Chọn output mới trong data/processed, manifest tối đa 1 MB và xác nhận --labels-reviewed.")
        raw_manifest = args.manifest.read_bytes()
        rows = list(csv.DictReader(raw_manifest.decode("utf-8-sig").splitlines()))
        known = set()
        required = ("tesla-expanded-v2/splits.csv", "week4-newcheck-v1/review.csv",
                    "non-tesla-gate-review-v1/review.csv", "non-tesla-brand-training-v3/review.csv")
        if any(not (ROOT / "data/processed" / name).is_file() for name in required):
            raise ValueError("Thiếu manifest phát triển cũ; chưa kiểm tra được ảnh đã từng dùng.")
        historical = set((ROOT / "data/processed").glob("*/review.csv")) | set((ROOT / "data/processed").glob("*/splits.csv"))
        for manifest in historical:
            if manifest.resolve() == args.manifest.resolve():
                continue
            if manifest.stat().st_size > 10_000_000:
                raise ValueError("Manifest lịch sử quá lớn; chưa kiểm tra được trùng dữ liệu.")
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                known.update(r["sha256"] for r in csv.DictReader(handle) if r.get("sha256"))
        config = model / "model.json"
        if config.stat().st_size > 5_000_000:
            raise ValueError("Model metadata quá lớn.")
        metadata = json.loads(config.read_text(encoding="utf-8"))
        known.update(r["sha256"] for r in metadata["train_records"] if r.get("sha256"))
        if args.validate_only:
            checked = validate_rows(rows, ROOT / "data/raw", known)
            report = {"validation_only": True, "images": len(checked),
                      "labels": dict(Counter(r[0]["class_id"] for r in checked)),
                      "independence": "not_verified", "model_inference_performed": False}
        else:
            from src.recognize_vehicle import Recognizer
            recognizer = Recognizer(model)
            report = evaluate(rows, recognizer, ROOT / "data/raw", known)
            report["model_file_sha256"] = hashlib.sha256((model / "classifier.npz").read_bytes()).hexdigest()
        report["manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
        report["model_manifest_sha256"] = metadata["manifest_sha256"]
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".field-eval-") as temporary:
            staging = Path(temporary) / "report"
            staging.mkdir()
            (staging / "manifest.csv").write_bytes(raw_manifest)
            name = "validation.json" if args.validate_only else "metrics.json"
            (staging / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if output.exists():
                raise ValueError("Output vừa được tạo; giữ nguyên kết quả cũ.")
            staging.rename(output)
        action = "kiểm tra dữ liệu, chưa chạy AI" if args.validate_only else "đánh giá, không huấn luyện lại"
        print(f"Đã {action}: {report['images']} ảnh. Kết quả: {output}")
    except (ValueError, OSError, KeyError, UnicodeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
