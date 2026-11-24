"""Compact CPU backend pinned to the same V3.1 classifier and encoder revisions."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.vision_features import DINO, SIGLIP, ROOT, VisionFeatures, normalized, padded_box


def preprocess(image, encoder):
    # Pillow approximates Torchvision's uint8 resize; validate the complete pipeline before use.
    if encoder == "dino":
        width, height = image.size
        size = (int(width * 256 / height), 256) if width >= height else (256, int(height * 256 / width))
        image = image.resize(size, Image.Resampling.BICUBIC)
        left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
        image = image.crop((left, top, left + 224, top + 224))
        mean = np.asarray([.485, .456, .406], np.float32) * 255
        std = np.asarray([.229, .224, .225], np.float32) * 255
    elif encoder == "siglip":
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        mean = np.asarray([.5, .5, .5], np.float32) * 255
        std = np.asarray([.5, .5, .5], np.float32) * 255
    else:
        raise ValueError("Unknown encoder")
    values = (np.asarray(image, dtype=np.float32) - mean) / std
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def detector_pixels(image):
    values = np.asarray(image)
    height, width = values.shape[:2]
    y = np.maximum((np.arange(320, dtype=np.float32) + .5) * (height / 320) - .5, 0)
    x = np.maximum((np.arange(320, dtype=np.float32) + .5) * (width / 320) - .5, 0)
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    dy = (y - y0.astype(np.float32))[:, None, None]
    dx = (x - x0.astype(np.float32))[None, :, None]
    top = values[y0[:, None], x0].astype(np.float32) * (1 - dx) + values[y0[:, None], x1].astype(np.float32) * dx
    bottom = values[y1[:, None], x0].astype(np.float32) * (1 - dx) + values[y1[:, None], x1].astype(np.float32) * dx
    result = (top * (1 - dy) + bottom * dy) / 255
    return np.ascontiguousarray(result.transpose(2, 0, 1)[None])


def nms(boxes, scores, threshold):
    order, keep = np.argsort(-scores, kind="stable"), []
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    while len(order):
        index, others = order[0], order[1:]
        keep.append(index)
        tl, br = np.maximum(boxes[index, :2], boxes[others, :2]), np.minimum(boxes[index, 2:], boxes[others, 2:])
        intersection = np.maximum(br - tl, 0).prod(axis=1)
        iou = intersection / np.maximum(area[index] + area[others] - intersection, 1e-8)
        order = others[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class OnnxFeatures(VisionFeatures):
    def __init__(self, model_run, runtime=None, *, require_validation=True):
        candidate = Path(runtime or ROOT / "data/processed/vision-onnx-v31").absolute()
        runtime = candidate.resolve()
        if candidate != runtime or not runtime.is_relative_to((ROOT / "data/processed").resolve()):
            raise ValueError("Runtime ONNX phải nằm trong data/processed.")
        metadata = runtime / "runtime.json"
        if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 64_000:
            raise ValueError("Thiếu metadata runtime ONNX hợp lệ.")
        self.config = json.loads(metadata.read_text(encoding="utf-8"))
        config = self.config
        if (not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("source_dino") != list(DINO)
                or config.get("source_siglip") != list(SIGLIP) or config.get("precision") != "weight_only_int8"
                or config.get("box_weights") != [10.0, 10.0, 5.0, 5.0]
                or config.get("nms_thresh") != .55 or config.get("topk_candidates") != 300
                or config.get("detections_per_img") != 300 or config.get("bbox_xform_clip") != 4.135166556742356):
            raise ValueError("Runtime ONNX không khớp cấu hình đã kiểm tra.")
        if require_validation:
            proof = runtime / "validation.json"
            if proof.is_symlink() or not proof.is_file() or proof.stat().st_size > 64_000:
                raise ValueError("Runtime ONNX chưa có kết quả validation phù hợp.")
            validation = json.loads(proof.read_text(encoding="utf-8"))
            if (not isinstance(validation, dict) or validation.get("passed") is not True
                    or validation.get("runtime_manifest_sha256") != file_hash(metadata)):
                raise ValueError("Kết quả validation không khớp runtime hiện tại.")
        files = config.get("files", {})
        if not isinstance(files, dict) or set(files) != {"dino.onnx", "siglip.onnx", "detector.onnx", "anchors.npz"}:
            raise ValueError("Danh sách tệp runtime không hợp lệ.")
        for name, record in files.items():
            path = runtime / name
            if (not isinstance(record, dict) or path.is_symlink() or path.resolve() != path.absolute()
                    or not path.is_file() or not 0 < path.stat().st_size <= 400_000_000
                    or path.stat().st_size != record.get("bytes") or file_hash(path) != record.get("sha256")):
                raise ValueError("Tệp ONNX không khớp checksum; không dùng mô hình này.")
        model_run = Path(model_run).resolve()
        if (not model_run.is_relative_to((ROOT / "data/processed").resolve())
                or file_hash(model_run / "model.json") != config.get("model_metadata_sha256")
                or file_hash(model_run / "classifier.npz") != config.get("classifier_sha256")):
            raise ValueError("Encoder ONNX và bộ phân loại không cùng phiên bản.")
        if (runtime / "anchors.npz").stat().st_size > 1_000_000:
            raise ValueError("Anchor metadata quá lớn.")
        with np.load(runtime / "anchors.npz", allow_pickle=False) as archive:
            self.anchors = archive["anchors"].copy()
        if (self.anchors.ndim != 2 or self.anchors.shape[1] != 4 or not 0 < len(self.anchors) <= 10_000
                or not np.isfinite(self.anchors).all() or np.any(self.anchors[:, 2:] <= self.anchors[:, :2])):
            raise ValueError("Anchor metadata không hợp lệ.")
        import onnxruntime as ort
        self.sessions = {}
        for name in ("dino", "siglip", "detector"):
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(2, os.cpu_count() or 1)
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            session = ort.InferenceSession(str(runtime / (name + ".onnx")), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            size = 320 if name == "detector" else 224
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].name != "pixels" or inputs[0].shape != [1, 3, size, size]:
                raise ValueError("Đầu vào model ONNX không đúng kích thước cố định.")
            self.sessions[name] = session

    def encode(self, images, encoder):
        if encoder not in ("dino", "siglip") or not 1 <= len(images) <= 16:
            raise ValueError("Invalid image batch")
        values = [self.sessions[encoder].run(None, {"pixels": preprocess(image, encoder)})[0][0] for image in images]
        return normalized(np.stack(values))

    def detect(self, image):
        offsets, logits = self.sessions["detector"].run(None, {"pixels": detector_pixels(image)})
        if (offsets.shape != (1, len(self.anchors), 4) or logits.shape != (1, len(self.anchors), 91)
                or not np.isfinite(offsets).all() or not np.isfinite(logits).all()):
            raise ValueError("Đầu ra detector ONNX không hợp lệ.")
        offsets, logits = offsets[0], logits[0]
        wh = self.anchors[:, 2:] - self.anchors[:, :2]
        centers = self.anchors[:, :2] + wh * .5
        delta = offsets / np.asarray(self.config["box_weights"], np.float32)
        pred_center = centers + delta[:, :2] * wh
        pred_wh = np.exp(np.minimum(delta[:, 2:], self.config["bbox_xform_clip"])) * wh
        boxes = np.clip(np.concatenate((pred_center - .5 * pred_wh, pred_center + .5 * pred_wh), axis=1), 0, 320)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = exp / exp.sum(axis=1, keepdims=True)
        predictions = []
        # Boxes below .45 cannot outrank returned vehicles; retain all classes for the global top-300 cut.
        for label in range(1, scores.shape[1]):
            indexes = np.flatnonzero(scores[:, label] >= .45)
            indexes = indexes[np.argsort(-scores[indexes, label], kind="stable")[:self.config["topk_candidates"]]]
            selected = indexes[nms(boxes[indexes], scores[indexes, label], self.config["nms_thresh"])]
            predictions.extend((float(scores[index, label]), label, int(index)) for index in selected)
        predictions.sort(reverse=True)
        result = []
        for score, label, index in predictions[:self.config["detections_per_img"]]:
            if label not in (3, 8):
                continue
            box = boxes[index] * np.asarray([image.width / 320, image.height / 320] * 2, np.float32)
            crop = padded_box(box.tolist(), image.size)
            result.append(dict(box=list(crop), detector_score=score, area=(crop[2] - crop[0]) * (crop[3] - crop[1])))
        return sorted(result, key=lambda row: row["area"], reverse=True)
