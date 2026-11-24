"""Read-only startup checks: no model loading, downloads, keys or image uploads."""

import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root).resolve()
    checks = []
    for module in ("streamlit", "pandas", "PIL", "numpy", "torch", "torchvision", "transformers", "safetensors"):
        checks.append((module, importlib.util.find_spec(module) is not None, True))
    model = Path(os.environ.get("CARVISION_VISION_RUN", str(root / "data/processed/vision-recognition-v3-brand"))).resolve()
    inside = model.is_relative_to((root / "data/processed").resolve())
    checks.append(("Model nằm trong data/processed", inside, True))
    for name, maximum in (("model.json", 5_000_000), ("classifier.npz", 20_000_000)):
        path = model / name
        checks.append((name, inside and path.is_file() and 0 < path.stat().st_size <= maximum, True))
    if inside and (model / "model.json").is_file() and (model / "model.json").stat().st_size <= 5_000_000:
        try:
            meta = json.loads((model / "model.json").read_text(encoding="utf-8"))
            for key in ("dino", "siglip"):
                name, revision = meta[key]
                if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[a-f0-9]{40}", revision):
                    raise ValueError("Invalid checkpoint")
                snapshot = root / ".kickoff-build/huggingface" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
                present = (snapshot / "config.json").is_file() and any(p.is_file() for p in snapshot.glob("*.safetensors"))
                checks.append((f"Cache {key} theo revision model", present, True))
        except (ValueError, KeyError, TypeError, OSError):
            checks.append(("Model metadata đọc được", False, True))
    detector = root / ".kickoff-build/torch/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
    checks.append(("Cache detector", detector.is_file() and detector.stat().st_size > 0, True))
    checks.append(("Ảnh Supabase: cấu hình máy chủ (tùy chọn)", bool(os.environ.get("CARVISION_SUPABASE_URL") and os.environ.get("CARVISION_SUPABASE_KEY")), False))
    return checks


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    checks = check()
    for name, ok, required in checks:
        print(f"[{'OK' if ok else 'THIEU' if required else 'CHUA CAU HINH'}] {name}")
    print("Chỉ kiểm tra thư viện và sự hiện diện tệp; cần chạy demo để xác minh load weights và inference.")
    print("Hướng dẫn: docs/vision_recognition.md; docs/week5_demo.md")
    raise SystemExit(0 if all(ok or not required for _, ok, required in checks) else 1)


if __name__ == "__main__":
    main()
