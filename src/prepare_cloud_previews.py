"""Owner-only thumbnail publication; no raw-image writes, training or schema changes."""

import argparse
import io
import json
import os
from pathlib import Path
import sys

from src.cloud_data import CloudData
from src.recognition_workflow import reference_preview

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT, publish=False, client=None):
    root = Path(root).resolve()
    run = Path(os.environ.get("CARVISION_VISION_RUN", str(root / "data/processed/vision-recognition-v3-brand"))).resolve()
    config = run / "model.json"
    if not run.is_relative_to(root / "data/processed") or not config.is_file() or config.stat().st_size > 5_000_000:
        raise ValueError("Cần metadata model hợp lệ trong data/processed.")
    records = json.loads(config.read_text(encoding="utf-8"))["train_records"]
    if not isinstance(records, list) or not 0 < len(records) <= 1000:
        raise ValueError("Danh sách ảnh đối chiếu không hợp lệ.")
    cloud = CloudData() if client is None else client
    catalog = cloud.catalog()
    prepared = []
    # Validate every local source and matching catalog row before the first upload.
    for record in records:
        matches = [row for row in catalog if row.get("sha256") == record.get("sha256")
                   and row.get("class_id") == record.get("class_id") and row.get("decision") == "keep"]
        if len(matches) != 1:
            raise ValueError("Ảnh đối chiếu phải khớp đúng một bản ghi cloud đã giữ.")
        image = reference_preview(record, root / "data/raw")
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=82, method=4)
        data = output.getvalue()
        if len(data) > 250_000:
            raise ValueError("Ảnh thu nhỏ vượt giới hạn.")
        prepared.append((matches[0], data))
    added = 0
    if publish:
        for row, data in prepared:
            added += cloud.publish_thumbnail(row, data)
    return {"references": len(prepared), "original_bytes": sum(row["byte_size"] for row, _ in prepared),
            "thumbnail_bytes": sum(len(data) for _, data in prepared), "published": publish,
            "new_links": added, "existing_links": len(prepared) - added if publish else 0}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Upload bản nhỏ và liên kết provenance; giữ nguyên ảnh gốc")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(publish=args.publish), ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
