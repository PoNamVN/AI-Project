"""Compare frozen runtimes on the same old validation inputs; never fit or inspect new candidates."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quality_passed(baseline, candidate):
    a, b = baseline["results"], candidate["results"]
    if len(a) != len(b) or not a:
        return False
    for old, new in zip(a, b):
        if (old["source_path"] != new["source_path"] or old["expected"] != new["expected"]
                or (old["top1"] == old["expected"] and new["top1"] != new["expected"])):
            return False
    return (candidate["tesla_correct"] >= baseline["tesla_correct"]
            and candidate["tesla_macro_f1"] >= baseline["tesla_macro_f1"]
            and candidate["tesla_top3_correct"] >= baseline["tesla_top3_correct"]
            and candidate["other_returned_tesla"] <= baseline["other_returned_tesla"])


def measure(backend, runtime):
    import numpy as np
    from src.onnx_features import OnnxFeatures, file_hash
    from src.recognize_vehicle import Recognizer
    from src.vision_features import VisionFeatures
    model = ROOT / "data/processed/vision-recognition-v3-brand"
    with (ROOT / "data/processed/tesla-expanded-v2/splits.csv").open(encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "validation"]
    with (ROOT / "data/processed/non-tesla-gate-review-v1/review.csv").open(encoding="utf-8-sig") as handle:
        negatives = [{**r, "class_id": "other_car"} for r in csv.DictReader(handle)
                     if r["split"] == "validation" and r["decision"] == "keep"]
    if len(rows) != 29 or len(negatives) != 9:
        raise ValueError("Manifest validation đã thay đổi; cần thiết kế lại phép so sánh.")
    for row in rows + negatives:
        path = (ROOT / "data/raw" / row["source_path"]).resolve()
        if not path.is_relative_to(ROOT / "data/raw") or file_hash(path) != row["sha256"]:
            raise ValueError("Ảnh validation không khớp đường dẫn/checksum đã chốt.")
    start = time.perf_counter()
    engine = VisionFeatures(image_only=True) if backend == "torch" else OnnxFeatures(model, runtime, require_validation=False)
    recognizer = Recognizer(model, engine)
    loaded = time.perf_counter() - start
    results, times = [], []
    for row in rows + negatives:
        start = time.perf_counter()
        result, _, _ = recognizer.analyze(ROOT / "data/raw" / row["source_path"], None, False)
        times.append(time.perf_counter() - start)
        candidates = result.get("candidates", [])
        results.append(dict(source_path=row["source_path"], expected=row["class_id"], status=result["status"],
                            top1=candidates[0]["class_id"] if candidates else None,
                            top3=[r["class_id"] for r in candidates]))
    f1 = []
    for label in recognizer.labels:
        tp = sum(r["expected"] == label and r["top1"] == label for r in results[:29])
        fp = sum(r["expected"] != label and r["top1"] == label for r in results[:29])
        fn = sum(r["expected"] == label and r["top1"] != label for r in results[:29])
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return dict(backend=backend, tesla_correct=sum(r["expected"] == r["top1"] for r in results[:29]),
                tesla_macro_f1=sum(f1) / len(f1), tesla_top3_correct=sum(r["expected"] in r["top3"] for r in results[:29]),
                other_returned_tesla=sum(r["top1"] is not None for r in results[29:]),
                load_seconds=loaded, median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times, 95)),
                selection_split="existing_validation_not_independent_test", results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("torch", "onnx"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    runtime, output = args.runtime.resolve(), args.output.resolve()
    if (not runtime.is_relative_to(ROOT / "data/processed") or not output.is_relative_to(ROOT / "data/processed")
            or output.exists()):
        raise ValueError("Chọn runtime trong data/processed và output mới; không ghi đè.")
    if args.worker:
        report = measure(args.worker, runtime)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return
    proof = runtime / "validation.json"
    if proof.exists():
        raise ValueError("Runtime đã có bản validation; tạo runtime mới để giữ lịch sử.")
    output.mkdir(parents=True)
    for backend in ("torch", "onnx"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(runtime), "--output", str(output / (backend + ".json")),
                        "--worker", backend], check=True, cwd=ROOT)
    baseline = json.loads((output / "torch.json").read_text(encoding="utf-8"))
    candidate = json.loads((output / "onnx.json").read_text(encoding="utf-8"))
    from src.onnx_features import file_hash
    result = dict(passed=quality_passed(baseline, candidate), independent_evaluation=False,
                  comparison="no_correct_top1_regressions_on_same_38_validation_inputs",
                  runtime_manifest_sha256=file_hash(runtime / "runtime.json"),
                  baseline={k: v for k, v in baseline.items() if k != "results"},
                  candidate={k: v for k, v in candidate.items() if k != "results"})
    with proof.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
