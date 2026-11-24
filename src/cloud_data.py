"""Server-side access to the private Supabase image catalog; no extra dependencies."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.data_audit import MODELS, audit_dataset

BUCKET = "carvision-datasets"
MAX_BYTES = 30_000_000


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Credentials must stay on the configured Supabase host.


class CloudData:
    def __init__(self, url=None, key=None):
        url = os.environ.get("CARVISION_SUPABASE_URL", "") if url is None else url
        self.key = os.environ.get("CARVISION_SUPABASE_KEY", "") if key is None else key
        if not isinstance(url, str) or not isinstance(self.key, str):
            raise ValueError("Cấu hình kết nối Supabase không hợp lệ.")
        self.url = url.rstrip("/")
        if not re.fullmatch(r"https://[a-z0-9]{20}\.supabase\.co", self.url) or not self.key:
            raise ValueError("Cần CARVISION_SUPABASE_URL và CARVISION_SUPABASE_KEY trong môi trường máy chủ. Xem docs/supabase_data.md.")

    def _get(self, path, maximum):
        headers = {"apikey": self.key}
        if self.key.startswith("eyJ"):
            headers["Authorization"] = "Bearer " + self.key
        try:
            with build_opener(NoRedirect).open(Request(self.url + path, headers=headers), timeout=60) as response:
                data = response.read(maximum + 1)
        except HTTPError as error:
            raise ValueError(f"Supabase HTTP {error.code}. Kiểm tra cấu hình và quyền đọc trên máy chủ.") from None
        except (URLError, TimeoutError, OSError):
            raise ValueError("Không kết nối được Supabase. Thử lại sau.") from None
        if len(data) > maximum:
            raise ValueError("Dữ liệu Supabase vượt giới hạn đọc.")
        return data

    def catalog(self, dataset=None):
        if dataset is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", dataset):
            raise ValueError("Tên dataset không hợp lệ.")
        rows = []
        while True:
            query = {"select": "*", "uploaded": "eq.true", "order": "dataset.asc,relative_path.asc", "limit": 500, "offset": len(rows)}
            if dataset:
                query["dataset"] = "eq." + dataset
            try:
                page = json.loads(self._get("/rest/v1/carvision_images?" + urlencode(query), 5_000_000))
                if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
                    raise ValueError("Danh mục Supabase không hợp lệ.")
                required = ("dataset", "relative_path", "class_id", "purpose", "decision", "split",
                            "review_status", "sha256", "object_path", "source_url", "license")
                for row in page:
                    if (any(not isinstance(row.get(field), str) or not row[field] for field in required)
                            or row["class_id"] not in MODELS or row["decision"] not in ("keep", "exclude")
                            or not isinstance(row.get("provenance"), dict)
                            or not isinstance(row.get("byte_size"), int)):
                        raise ValueError("Danh mục Supabase thiếu thông tin ảnh hợp lệ.")
            except (UnicodeError, json.JSONDecodeError):
                raise ValueError("Không đọc được danh mục Supabase.") from None
            rows.extend(page)
            if len(page) < 500:
                return rows
            if len(rows) >= 10_000:
                raise ValueError("Danh mục vượt giới hạn 10.000 ảnh; cần tải theo phiên bản nhỏ hơn.")

    def image(self, row):
        digest = row.get("sha256", "")
        object_path = row.get("object_path", "")
        size = row.get("byte_size", 0)
        if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or not isinstance(object_path, str) or not re.fullmatch(r"sha256/" + digest + r"\.(jpg|jpeg|png|webp|bmp|tif|tiff)", object_path)
                or not isinstance(size, int) or not 0 < size <= MAX_BYTES):
            raise ValueError("Thông tin ảnh trên cloud không hợp lệ.")
        data = self._get(f"/storage/v1/object/authenticated/{BUCKET}/{object_path}", size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Ảnh tải về không khớp checksum; không sử dụng ảnh này.")
        return data

    def reference_image(self, digest, class_id, thumbnail=False):
        """Fetch one training reference by its model-pinned checksum, not the dataset."""
        if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or not isinstance(class_id, str) or class_id not in MODELS):
            raise ValueError("Ảnh tham chiếu thiếu checksum hoặc nhãn hợp lệ.")
        query = {"select": "*", "sha256": "eq." + digest, "class_id": "eq." + class_id,
                 "decision": "eq.keep", "uploaded": "eq.true", "limit": 1}
        try:
            rows = json.loads(self._get("/rest/v1/carvision_images?" + urlencode(query), 20_000))
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("Không đọc được thông tin ảnh tham chiếu.") from None
        if not isinstance(rows, list) or len(rows) > 1:
            raise ValueError("Thông tin ảnh tham chiếu không hợp lệ.")
        if not rows:
            raise ValueError("Ảnh tham chiếu chưa có trong kho riêng.")
        row = rows[0]
        if (not isinstance(row, dict) or row.get("sha256") != digest
                or row.get("class_id") != class_id or row.get("decision") != "keep"
                or row.get("uploaded") is not True):
            raise ValueError("Ảnh tham chiếu không khớp mô hình.")
        preview = row.get("provenance", {}).get("preview_v1") if isinstance(row.get("provenance"), dict) else None
        if thumbnail and preview is not None:
            return self.thumbnail_image(row, preview)
        return self.image(row)

    def thumbnail_image(self, row, preview):
        digest = preview.get("sha256", "") if isinstance(preview, dict) else ""
        source = row.get("sha256", "")
        size = preview.get("byte_size", 0) if isinstance(preview, dict) else 0
        path = f"previews/v1/{source}/{digest}.webp"
        if (not re.fullmatch(r"[a-f0-9]{64}", source) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or preview.get("source_sha256") != source or preview.get("class_id") != row.get("class_id")
                or preview.get("object_path") != path or type(size) is not int or not 0 < size <= 250_000):
            raise ValueError("Thông tin ảnh thu nhỏ không khớp ảnh gốc.")
        data = self._get(f"/storage/v1/object/authenticated/{BUCKET}/{path}", size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Ảnh thu nhỏ không khớp checksum; không sử dụng ảnh này.")
        return data

    def publish_thumbnail(self, row, data):
        """Owner command only: immutable preview, then conditional provenance update."""
        source = row.get("sha256", "")
        if (not isinstance(source, str) or not re.fullmatch(r"[a-f0-9]{64}", source)
                or row.get("class_id") not in MODELS or row.get("decision") != "keep"
                or row.get("uploaded") is not True or not isinstance(row.get("provenance"), dict)
                or not isinstance(data, bytes) or not 0 < len(data) <= 250_000
                or not isinstance(row.get("dataset"), str) or not isinstance(row.get("relative_path"), str)):
            raise ValueError("Không thể công bố ảnh thu nhỏ cho bản ghi này.")
        digest = hashlib.sha256(data).hexdigest()
        preview = dict(source_sha256=source, class_id=row["class_id"], sha256=digest,
                       byte_size=len(data), object_path=f"previews/v1/{source}/{digest}.webp")
        old = row["provenance"].get("preview_v1")
        if old is not None:
            if old != preview or self.thumbnail_image(row, old) != data:
                raise ValueError("Ảnh thu nhỏ đã có khác phiên bản; giữ nguyên bản cũ.")
            return False
        headers = {"apikey": self.key, "Content-Type": "image/webp", "x-upsert": "false"}
        if self.key.startswith("eyJ"):
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.url + f"/storage/v1/object/{BUCKET}/{preview['object_path']}",
                          data=data, headers=headers, method="POST")
        try:
            with build_opener(NoRedirect).open(request, timeout=60) as response:
                response.read(20_001)
        except HTTPError as error:
            if error.code not in (400, 409):
                raise ValueError(f"Supabase HTTP {error.code} khi lưu ảnh thu nhỏ.") from None
            # A previous interrupted run may already have written this immutable object.
        except (URLError, TimeoutError, OSError):
            raise ValueError("Không lưu được ảnh thu nhỏ trên Supabase.") from None
        self.thumbnail_image(row, preview)
        query = {"dataset": "eq." + row["dataset"], "relative_path": "eq." + row["relative_path"],
                 "sha256": "eq." + source, "uploaded": "eq.true", "decision": "eq.keep",
                 "provenance": "eq." + json.dumps(row["provenance"], separators=(",", ":")), "select": "sha256"}
        headers.update({"Content-Type": "application/json", "Prefer": "return=representation"})
        payload = json.dumps({"provenance": {**row["provenance"], "preview_v1": preview}}).encode()
        request = Request(self.url + "/rest/v1/carvision_images?" + urlencode(query), data=payload,
                          headers=headers, method="PATCH")
        try:
            with build_opener(NoRedirect).open(request, timeout=60) as response:
                updated = json.loads(response.read(20_001))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            raise ValueError("Chưa cập nhật được liên kết ảnh thu nhỏ; ảnh gốc được giữ nguyên.") from None
        if updated != [{"sha256": source}]:
            raise ValueError("Bản ghi đã đổi trong lúc xử lý; tải lại danh mục trước khi thử lại.")
        return True

    def download(self, dataset, destination):
        destination = Path(destination).resolve()
        if destination.exists():
            raise ValueError("Thư mục đầu ra đã tồn tại; chọn thư mục mới để giữ dữ liệu cũ.")
        rows = [row for row in self.catalog(dataset) if row.get("decision") == "keep"]
        if not rows:
            raise ValueError("Chưa có ảnh đề xuất giữ đã upload cho dataset này.")
        paths = set()
        for row in rows:
            relative = row.get("relative_path", "")
            parts = PurePosixPath(relative).parts
            if (len(parts) != 2 or parts[0] not in MODELS or parts[0] != row.get("class_id")
                    or "\\" in relative or ":" in relative or ".." in parts
                    or relative in paths or row.get("dataset") != dataset):
                raise ValueError("Đường dẫn hoặc nhãn cloud không hợp lệ.")
            paths.add(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent, prefix=".cloud-download-") as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for row in rows:
                target = staging / row["relative_path"]
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(self.image(row))
            records = [{**row["provenance"], "relative_path": row["relative_path"],
                        "source_url": row["source_url"], "license": row["license"],
                        "vehicle_id": row["provenance"].get("vehicle_id", ""),
                        "cloud_purpose": row["purpose"], "cloud_split": row["split"],
                        "cloud_review_status": row["review_status"]} for row in rows]
            fields = sorted({key for row in records for key in row})
            with (staging / "provenance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            report = audit_dataset(staging)
            if report["summary"]["readable"] != len(rows):
                raise ValueError("Bộ ảnh tải về có ảnh không qua kiểm tra; chưa xuất dữ liệu.")
            (staging / "cloud_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            if destination.exists():
                raise ValueError("Thư mục đầu ra đã được tạo trong lúc tải; giữ nguyên dữ liệu đó.")
            staging.rename(destination)
        return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Tải một phiên bản ảnh Tesla riêng biệt từ Supabase.")
    parser.add_argument("dataset")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        print(f"Đã tải và kiểm tra {CloudData().download(args.dataset, args.destination)} ảnh.")
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
