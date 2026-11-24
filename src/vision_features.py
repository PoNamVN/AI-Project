"""Frozen visual encoders and COCO vehicle localization; models stay on the server."""

import io
import os
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageOps
from src.data_audit import MAX_PIXELS

ROOT = Path(__file__).resolve().parents[1]
DINO = ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056")
SIGLIP = ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2")
MODES = ("dino_full", "dino_crop", "siglip_full", "siglip_crop", "combined_crop")


def normalized(features):
    array = np.asarray(features, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite image features")
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def open_image(source):
    if isinstance(source, bytes):
        if len(source) > 30_000_000:
            raise ValueError("Ảnh vượt giới hạn 30 MB.")
        source = io.BytesIO(source)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(source) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Ảnh vượt giới hạn 25 triệu pixel.")
                return ImageOps.exif_transpose(image).convert("RGB")
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise ValueError("Ảnh vượt giới hạn bộ nhớ.") from error


def padded_box(box, size):
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("Invalid vehicle box")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid vehicle box")
    dx, dy = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    result = (
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(size[0], int(x2 + dx + 1)),
        min(size[1], int(y2 + dy + 1)),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Vehicle box outside image")
    return result


class VisionFeatures:
    def __init__(self, download=False):
        import torch
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        cache = ROOT / ".kickoff-build/huggingface"
        torch.hub.set_dir(str(ROOT / ".kickoff-build/torch"))
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        )
        if not download and not checkpoint.is_file():
            raise ValueError(
                "Chưa có detector. Chạy bước chuẩn bị model theo docs/vision_recognition.md."
            )
        self.detector = ssdlite320_mobilenet_v3_large(weights=weights).eval()
        self.detect_transform = weights.transforms()
        options = dict(cache_dir=str(cache), local_files_only=not download)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            DINO[0], revision=DINO[1], **options
        )
        self.dino = AutoModel.from_pretrained(
            DINO[0], revision=DINO[1], use_safetensors=True, **options
        ).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(
            SIGLIP[0], revision=SIGLIP[1], **options
        )
        self.siglip = AutoModel.from_pretrained(
            SIGLIP[0], revision=SIGLIP[1], use_safetensors=True, **options
        ).eval()

    def detect(self, image):
        with self.torch.inference_mode():
            output = self.detector([self.detect_transform(image)])[0]
        result = []
        for box, label, score in zip(
            output["boxes"], output["labels"], output["scores"]
        ):
            if int(label) in (3, 8) and float(score) >= 0.45:
                crop = padded_box(box.tolist(), image.size)
                result.append(
                    {
                        "box": list(crop),
                        "detector_score": float(score),
                        "area": (crop[2] - crop[0]) * (crop[3] - crop[1]),
                    }
                )
        return sorted(result, key=lambda row: row["area"], reverse=True)

    def encode(self, images, encoder):
        with self.torch.inference_mode():
            if encoder == "dino":
                values = self.dino_processor(images=images, return_tensors="pt")
                tokens = self.dino(**values).last_hidden_state
                features = self.torch.cat(
                    (tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1
                )
            elif encoder == "siglip":
                values = self.siglip_processor(images=images, return_tensors="pt")
                features = self.siglip.vision_model(**values).pooler_output
            else:
                raise ValueError("Unknown encoder")
        return normalized(features.cpu().numpy())

    def texts(self, descriptions):
        tokens = self.siglip_processor.tokenizer(
            descriptions,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            features = self.siglip.text_model(**tokens).pooler_output
        return normalized(features.cpu().numpy())

    def modes(self, image, crop, mode=None):
        if mode is not None:
            siglip_crop = self.encode([crop], "siglip")[0]
            if mode == "siglip_crop":
                chosen = siglip_crop
            elif mode == "combined_crop":
                chosen = normalized(
                    np.concatenate((self.encode([crop], "dino")[0], siglip_crop))
                )
            elif mode in ("dino_full", "dino_crop", "siglip_full"):
                chosen = self.encode(
                    [image if mode.endswith("_full") else crop], mode.split("_")[0]
                )[0]
            else:
                raise ValueError("Unknown feature mode")
            return {mode: chosen, "siglip_crop": siglip_crop}
        dino = self.encode([image, crop], "dino")
        siglip = self.encode([image, crop], "siglip")
        return {
            "dino_full": dino[0],
            "dino_crop": dino[1],
            "siglip_full": siglip[0],
            "siglip_crop": siglip[1],
            "combined_crop": normalized(np.concatenate((dino[1], siglip[1]))),
        }

    def selected(self, images, mode):
        if mode == "combined_crop":
            return normalized(
                np.concatenate(
                    (self.encode(images, "dino"), self.encode(images, "siglip")), axis=1
                )
            )
        if mode not in MODES:
            raise ValueError("Unknown feature mode")
        return self.encode(images, mode.split("_")[0])
