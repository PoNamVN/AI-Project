"""Install a checksum-pinned runtime during hosting builds; never unpack arbitrary paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE = 2_500_000_000
MANIFEST = ".kickoff-build/runtime-bundle-manifest.json"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_path(name):
    return bool(re.fullmatch(
        r"data/processed/vision-recognition-v3-brand/(model\.json|classifier\.npz)"
        r"|data/processed/vision-onnx-v31/(runtime\.json|validation\.json|dino\.onnx|siglip\.onnx|detector\.onnx|anchors\.npz)"
        r"|\.kickoff-build/huggingface/models--(facebook--dinov2-small|google--siglip2-base-patch16-224)"
        r"/snapshots/[a-f0-9]{40}/(config\.json|preprocessor_config\.json|model\.safetensors|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json)"
        r"|\.kickoff-build/torch/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df\.pth"
        r"|\.kickoff-build/runtime-licenses/(runtime_notices\.md|licenses/(Apache-2\.0|Torchvision-BSD)\.txt)", name))


def install(archive_path, expected, root=ROOT):
    root, archive_path = Path(root).resolve(), Path(archive_path)
    if not re.fullmatch(r"[a-f0-9]{64}", expected or "") or archive_path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Cần SHA-256 đã chốt và gói runtime trong giới hạn 2,5 GB.")
    if file_hash(archive_path) != expected:
        raise ValueError("Runtime không khớp SHA-256; chưa cài đặt.")
    staging_parent = root / ".kickoff-build"
    if staging_parent.resolve() != staging_parent.absolute():
        raise ValueError("Thư mục cài runtime không được là liên kết.")
    staging_parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > 2000 or len(names) != len(set(names)) or sum(e.file_size for e in entries) > MAX_ARCHIVE:
            raise ValueError("Runtime có tệp trùng hoặc vượt giới hạn giải nén.")
        info = archive.getinfo(MANIFEST)
        if info.file_size > 1_000_000:
            raise ValueError("Manifest runtime quá lớn.")
        manifest = json.loads(archive.read(MANIFEST))
        records = manifest.get("files")
        if manifest.get("schema_version") != 1 or not isinstance(records, list) or len(records) != len(entries) - 1:
            raise ValueError("Manifest runtime không hợp lệ.")
        seen, chosen = set(), []
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Bản ghi runtime không hợp lệ.")
            name, size, digest = row.get("path", ""), row.get("bytes"), row.get("sha256", "")
            if (not isinstance(name, str) or name in seen or name not in names
                    or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                    or type(size) is not int or not 0 < size <= MAX_ARCHIVE):
                raise ValueError("Thông tin tệp runtime không hợp lệ.")
            seen.add(name)
            entry = archive.getinfo(name)
            if entry.file_size != size or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Runtime chứa liên kết hoặc kích thước không khớp.")
            if not selected_path(name):
                # A full team bundle may include old images/CSVs; cloud hosting never extracts them.
                if not (name.startswith("data/raw/") or (name.startswith("data/processed/") and name.endswith(".csv"))):
                    raise ValueError("Runtime chứa tệp ngoài danh sách cho phép.")
                continue
            target = root / name
            if target.resolve() != target.absolute() or not target.resolve().is_relative_to(root):
                raise ValueError("Đường dẫn runtime không an toàn.")
            if target.exists() and (not target.is_file() or file_hash(target) != digest):
                raise ValueError("Đã có tệp khác nội dung; giữ nguyên tệp hiện có.")
            chosen.append((entry, row, target))
        required = {"data/processed/vision-recognition-v3-brand/model.json", "data/processed/vision-recognition-v3-brand/classifier.npz"}
        if not required.issubset({r[1]["path"] for r in chosen}):
            raise ValueError("Runtime thiếu model CarVision.")
        with TemporaryDirectory(dir=staging_parent, prefix="install-") as temporary:
            staged = []
            for index, (entry, row, target) in enumerate(chosen):
                output = Path(temporary) / str(index)
                with archive.open(entry) as source, output.open("xb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                if file_hash(output) != row["sha256"]:
                    raise ValueError("Tệp runtime không khớp checksum; chưa cài đặt.")
                staged.append((output, target, row["sha256"]))
            for output, target, digest in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(output, target)  # Atomic no-overwrite on the same filesystem.
                except FileExistsError:
                    if target.is_symlink() or not target.is_file() or file_hash(target) != digest:
                        raise ValueError("Tệp đích vừa thay đổi; giữ nguyên dữ liệu.") from None
    return len(chosen)


class HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Chỉ tải runtime qua HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Kiểm tra/cài gói đã tải trên máy")
    parser.add_argument("--sha256", default=os.environ.get("CARVISION_RUNTIME_SHA256"))
    args = parser.parse_args()
    try:
        if args.archive:
            count = install(args.archive, args.sha256)
        else:
            url = os.environ.get("CARVISION_RUNTIME_URL", "")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Cần URL HTTPS riêng cho gói runtime.")
            if not re.fullmatch(r"[a-f0-9]{64}", args.sha256 or ""):
                raise ValueError("Cần CARVISION_RUNTIME_SHA256 đã chốt trước khi tải.")
            with TemporaryDirectory(prefix="carvision-download-") as temporary:
                archive = Path(temporary) / "runtime.zip"
                total = 0
                with build_opener(HTTPSRedirect).open(url, timeout=120) as source, archive.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > MAX_ARCHIVE:
                            raise ValueError("Runtime vượt giới hạn tải.")
                        output.write(chunk)
                count = install(archive, args.sha256)
    except Exception:
        # Signed artifact URLs and provider errors must never appear in public build logs.
        parser.exit(1, "Không cài được runtime. Kiểm tra URL, SHA-256, quyền tải và nội dung gói trên máy chủ; không gửi URL riêng hoặc khóa vào chat.\n")
    print(f"Runtime: {count} tệp đã kiểm checksum; không cài ảnh dataset hoặc khóa.")


if __name__ == "__main__":
    main()
