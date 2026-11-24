"""Tesla candidate recognition with scoped rejection and inspectable visual evidence."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from src.data_audit import MODELS
from src.vehicle_reasoning import (
    PROBES,
    PROBE_VI,
    brand_margin,
    probabilities,
    ridge_scores,
)
from src.vision_features import DINO, SIGLIP, MODES, ROOT, VisionFeatures, open_image


def occlusion_tiles(image):
    tiles = []
    boxes = []
    fill = tuple(np.asarray(image.resize((1, 1)))[0, 0].tolist())
    for row in range(3):
        for col in range(3):
            box = (
                col * image.width // 3,
                row * image.height // 3,
                (col + 1) * image.width // 3,
                (row + 1) * image.height // 3,
            )
            masked = image.copy()
            ImageDraw.Draw(masked).rectangle(box, fill=fill)
            tiles.append(masked)
            boxes.append(box)
    return tiles, boxes


class Recognizer:
    def __init__(self, run, engine=None):
        run = Path(run).resolve()
        if not run.is_relative_to((ROOT / "data/processed").resolve()):
            raise ValueError("Model phải nằm trong data/processed.")
        config = run / "model.json"
        arrays = run / "classifier.npz"
        if config.stat().st_size > 5_000_000 or arrays.stat().st_size > 20_000_000:
            raise ValueError("Model metadata quá lớn.")
        self.metadata = json.loads(config.read_text(encoding="utf-8"))
        self.labels = self.metadata["class_ids"]
        self.mode = self.metadata["feature_mode"]
        if (
            self.metadata.get("schema_version") != 1
            or self.labels != sorted(MODELS)
            or self.mode not in MODES
            or self.metadata["dino"] != list(DINO)
            or self.metadata["siglip"] != list(SIGLIP)
            or self.metadata["classifier"] not in ("ridge", "zeroshot")
            or not np.isfinite(self.metadata["brand_threshold"])
        ):
            raise ValueError("Model metadata không hợp lệ.")
        with np.load(arrays, allow_pickle=False) as archive:
            self.weights = archive["weights"].copy()
            self.train_features = archive["train_features"].copy()
            self.brand_text = archive["brand_text"].copy()
            self.class_text = archive["class_text"].copy()
            self.probe_text = archive["probe_text"].copy()
            self.brand_weights = (
                archive["brand_weights"].copy()
                if self.metadata.get("brand_method") == "ridge"
                else None
            )
        arrays_loaded = [
            self.weights,
            self.train_features,
            self.brand_text,
            self.class_text,
            self.probe_text,
        ]
        if (
            any(
                a.ndim != 2
                or not np.issubdtype(a.dtype, np.number)
                or not np.isfinite(a).all()
                for a in arrays_loaded
            )
            or len(self.train_features) != len(self.metadata["train_records"])
            or self.train_features.shape[1] > 4096
            or len(self.class_text) != 5
            or len(self.brand_text) != len(self.metadata["brand_groups"])
            or set(self.metadata["brand_groups"]) != {"tesla", "other_car", "not_car"}
            or self.brand_text.shape[1] != self.class_text.shape[1]
            or self.probe_text.shape[1] != self.class_text.shape[1]
            or len(self.probe_text) != sum(len(items) for items in PROBES.values())
            or (
                self.metadata["classifier"] == "ridge"
                and self.weights.shape != (self.train_features.shape[1] + 1, 5)
            )
        ):
            raise ValueError("Classifier arrays không hợp lệ.")
        if self.brand_weights is not None and (
            self.brand_weights.shape != (self.train_features.shape[1] + 1, 2)
            or not np.isfinite(self.brand_weights).all()
        ):
            raise ValueError("Brand classifier không hợp lệ.")
        if engine is not None:
            self.engine = engine
        elif os.environ.get("CARVISION_INFERENCE_BACKEND", "torch") == "onnx":
            from src.onnx_features import OnnxFeatures
            self.engine = OnnxFeatures(run)
        elif os.environ.get("CARVISION_INFERENCE_BACKEND", "torch") == "torch":
            self.engine = VisionFeatures(image_only=True)
        else:
            raise ValueError("Backend nhận diện phải là onnx hoặc torch.")

    def scores(self, features):
        if self.metadata["classifier"] == "zeroshot":
            return np.asarray(features) @ self.class_text.T * 5
        return ridge_scores(features, self.weights)

    def analyze(self, source, vehicle_index=None, explain=True):
        image = open_image(source)
        detections = self.engine.detect(image)
        base = {
            "review_required": True,
            "year": None,
            "trim": None,
            "specifications": None,
            "detections": detections,
            "note": "Điểm xếp hạng chưa được hiệu chuẩn thành xác suất. Chưa xác định đời xe, phiên bản hoặc thông số.",
        }
        if not detections:
            return (
                {
                    **base,
                    "status": "no_vehicle",
                    "brand": None,
                    "candidates": [],
                    "message": "Chưa tìm được vùng ô tô rõ ràng. Hãy chụp trọn thân xe, đủ sáng và gần hơn.",
                },
                None,
                None,
            )
        if (
            vehicle_index is None
            and len(detections) > 1
            and detections[1]["area"] >= 0.5 * detections[0]["area"]
        ):
            return (
                {
                    **base,
                    "status": "multiple_vehicles",
                    "brand": None,
                    "candidates": [],
                    "message": "Có nhiều xe đáng chú ý. Chọn chiếc cần nhận diện trước.",
                },
                None,
                None,
            )
        index = 0 if vehicle_index is None else vehicle_index
        if not isinstance(index, int) or not 0 <= index < len(detections):
            raise ValueError("Chỉ số xe không hợp lệ.")
        box = detections[index]["box"]
        crop = image.crop(box)
        features = self.engine.modes(image, crop, self.mode)
        vector = features[self.mode]
        margin, brand_scores = brand_margin(
            features["siglip_crop"], self.brand_text, self.metadata["brand_groups"]
        )
        semantic_evidence = {"margin": margin, "similarities": brand_scores}
        if self.brand_weights is not None:
            scores = ridge_scores(vector[None], self.brand_weights)[0]
            margin = float(scores[1] - scores[0])
            brand_scores = {"other_car": float(scores[0]), "tesla": float(scores[1])}
        preview_scale = min(1.0, 1100 / image.width, 700 / image.height)
        preview_size = (max(1, round(image.width * preview_scale)), max(1, round(image.height * preview_scale)))
        annotated = image.resize(preview_size, Image.Resampling.LANCZOS)
        preview_box = [round(value * (annotated.width / image.width if i % 2 == 0 else annotated.height / image.height))
                       for i, value in enumerate(box)]
        ImageDraw.Draw(annotated).rectangle(
            preview_box, outline="#25b6c4", width=max(2, annotated.width // 250)
        )
        base.update(
            selected_box=box,
            brand_evidence={
                "method": "ridge"
                if self.brand_weights is not None
                else "text_similarity",
                "margin": margin,
                "threshold": self.metadata["brand_threshold"],
                "scores": brand_scores,
                "semantic_auxiliary": semantic_evidence,
            },
        )
        if margin < self.metadata["brand_threshold"]:
            return (
                {
                    **base,
                    "status": "outside_or_uncertain",
                    "brand": None,
                    "candidates": [],
                    "message": "Chưa đủ căn cứ xem đây là Tesla. Có thể là xe ngoài phạm vi hoặc ảnh khó; hệ thống không gán ép một dòng Tesla.",
                },
                annotated,
                None,
            )
        raw = self.scores(vector[None])[0]
        order = np.argsort(raw)[::-1]
        ranking = probabilities(raw[None])[0]
        winner = int(order[0])
        gap = float(raw[order[0]] - raw[order[1]])
        uncertain = gap < self.metadata.get("class_gap_threshold", 0.08)
        candidates = [
            {
                "class_id": self.labels[i],
                "model": MODELS[self.labels[i]],
                "ranking_score": float(ranking[i]),
            }
            for i in order[:3]
        ]
        reference_scores = self.train_features @ vector
        reference_order = np.argsort(reference_scores)[::-1]

        def reference_record(i):
            row = self.metadata["train_records"][int(i)]
            path = (ROOT / "data/raw" / row["source_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw").resolve()):
                raise ValueError("Đường dẫn ảnh đối chiếu không hợp lệ.")
            return {
                "source_path": row["source_path"],
                "sha256": row.get("sha256", ""),
                "class_id": row["class_id"],
                "model": MODELS[row["class_id"]],
                "cosine_similarity": float(reference_scores[i]),
                "source_url": row["source_url"],
                "license": row["license"],
                "license_url": row.get("license_url", ""),
                "artist_html": row.get("artist_html", ""),
            }

        references = [reference_record(i) for i in reference_order[:3]]
        candidate_references = []
        for candidate in candidates:
            match = next((i for i in reference_order
                          if self.metadata["train_records"][int(i)]["class_id"] == candidate["class_id"]), None)
            if match is not None:
                candidate_references.append(reference_record(match))
        probes = []
        cursor = 0
        similarities = self.probe_text @ features["siglip_crop"]
        for category, items in PROBES.items():
            values = similarities[cursor : cursor + len(items)]
            idx = int(np.argmax(values))
            cursor += len(items)
            key, text = items[idx]
            gap_probe = float(np.sort(values)[-1] - np.sort(values)[-2])
            unresolved = gap_probe < 0.015 or category == "Cửa xe"
            hypothesis = (
                "Chưa có bộ phát hiện kiểu cửa; cần đối chiếu thủ công."
                if category == "Cửa xe"
                else "Chưa đủ khác biệt giữa các mô tả để đưa ra gợi ý."
                if unresolved
                else PROBE_VI[key]
            )
            probes.append(
                {
                    "category": category,
                    "hypothesis": hypothesis,
                    "description_compared": text,
                    "cosine_similarity": float(values[idx]),
                    "margin": gap_probe,
                    "status": "unresolved_not_verified_part"
                    if unresolved
                    else "text_image_similarity_hypothesis_not_verified_part",
                }
            )
        weak_brand = margin < 0.08 if self.brand_weights is not None else margin < 0
        result = {
            **base,
            "status": "uncertain_brand"
            if weak_brand
            else "uncertain_model"
            if uncertain
            else "tesla_candidate",
            "brand": "Tesla (chưa xác nhận)" if weak_brand else "Tesla (gợi ý)",
            "message": f"Chưa chắc hãng Tesla; nếu là Tesla thì ứng viên đứng đầu là {candidates[0]['model']}."
            if weak_brand
            else f"{'Cần đối chiếu thêm' if uncertain else 'Ứng viên đứng đầu'}: Tesla {candidates[0]['model']}.",
            "candidates": candidates,
            "class_margin": gap,
            "references": references,
            "candidate_references": candidate_references,
            "visual_hypotheses": probes,
            "explanation": [
                "Detector khoanh vùng ô tô; bộ phân loại sử dụng "
                + (
                    "ảnh toàn khung"
                    if self.mode.endswith("_full")
                    else "vùng xe đã cắt có thêm viền"
                )
                + ".",
                "Đặc trưng ảnh được so sánh bằng bộ phân loại đã học từ ảnh train; ba ảnh tham chiếu chỉ giúp bạn đối chiếu, không chứng minh cùng một xe.",
                f"Trong {len(references)} ảnh train gần nhất, có {sum(r['class_id'] == self.labels[winner] for r in references)} ảnh mang nhãn {candidates[0]['model']}. Đây là đối chiếu tương đồng thị giác, không phải xác nhận nhãn.",
                f"Chênh lệch điểm thô với ứng viên thứ hai: {gap:.4f}. Đây không phải độ tin cậy đã hiệu chuẩn.",
                "Các mô tả thân xe/góc nhìn/cửa bên dưới là kiểm tra tương đồng ảnh–văn bản độc lập, không phải chuỗi suy luận của bộ phân loại. Chưa có bộ phát hiện logo, đèn hoặc cửa chuyên biệt.",
                "Cần đối chiếu tổng thể thân xe, mui, cửa và phần đuôi. Góc ảnh có thể không thể hiện dấu hiệu phân biệt; nên chụp thêm góc trước hoặc sau nếu các ứng viên gần nhau.",
            ],
        }
        heatmap = None
        if explain:
            analyzed = (image if self.mode.endswith("_full") else crop).copy()
            analyzed.thumbnail((672, 672))
            baseline = float(
                self.scores(self.engine.selected([analyzed], self.mode))[0, winner]
            )
            masked, boxes = occlusion_tiles(analyzed)
            altered = []
            for start in range(0, len(masked), 3):
                altered.append(
                    self.engine.selected(masked[start : start + 3], self.mode)
                )
            deltas = baseline - self.scores(np.concatenate(altered))[:, winner]
            result["occlusion"] = {
                "grid": 3,
                "class_id": self.labels[winner],
                "score_drops": deltas.tolist(),
                "working_size": list(analyzed.size),
                "baseline_score": baseline,
                "note": "Che lần lượt từng ô. Đỏ biểu thị che vùng đó làm giảm điểm ứng viên; không chứng minh vùng đó là logo, đèn hay nguyên nhân ngữ nghĩa.",
            }
            positions = [
                "trên trái",
                "trên giữa",
                "trên phải",
                "giữa trái",
                "chính giữa",
                "giữa phải",
                "dưới trái",
                "dưới giữa",
                "dưới phải",
            ]
            strongest = int(np.argmax(deltas))
            if float(deltas[strongest]) > 0:
                result["explanation"].append(
                    f"Thử che 9 vùng: vùng {positions[strongest]} làm điểm thô {candidates[0]['model']} giảm nhiều nhất ({float(deltas[strongest]):.4f}). Vùng đỏ cho thấy sự nhạy cảm của model, không xác nhận một bộ phận cụ thể."
                )
            heatmap = analyzed.copy().convert("RGBA")
            overlay = Image.new("RGBA", analyzed.size)
            draw = ImageDraw.Draw(overlay)
            maximum = max(float(np.maximum(deltas, 0).max()), 1e-8)
            for tile, drop in zip(boxes, deltas):
                draw.rectangle(
                    tile, fill=(255, 40, 20, int(150 * max(0, float(drop)) / maximum))
                )
            heatmap = Image.alpha_composite(heatmap, overlay).convert("RGB")
        return result, annotated, heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vehicle-index", type=int)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to(
            (ROOT / "data/processed").resolve()
        ):
            raise ValueError("Chọn thư mục báo cáo mới trong data/processed.")
        result, annotated, heatmap = Recognizer(args.run).analyze(
            args.image, args.vehicle_index
        )
        output.mkdir(parents=True)
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if annotated:
            annotated.save(output / "vehicle.jpg")
        if heatmap:
            heatmap.save(output / "evidence.jpg")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
