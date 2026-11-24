"""Export the existing V3.1 image encoders; validation is a separate required step."""

import argparse
import json
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(output):
    from src.onnx_features import file_hash
    from src.vision_features import DINO, SIGLIP, VisionFeatures
    import numpy as np
    import onnx
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)

    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "data/processed"):
        raise ValueError("Chọn thư mục runtime mới trong data/processed; không ghi đè.")
    engine = VisionFeatures(image_only=True)

    class Encoder(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.model = engine.dino if name == "dino" else engine.siglip
            self.model.set_attn_implementation("eager")

        def forward(self, pixels):
            if self.name == "dino":
                tokens = self.model(pixel_values=pixels).last_hidden_state
                return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
            return self.model(pixel_values=pixels).pooler_output

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = engine.detector
            self.register_buffer("mean", torch.tensor(self.model.transform.image_mean)[None, :, None, None])
            self.register_buffer("std", torch.tensor(self.model.transform.image_std)[None, :, None, None])

        def forward(self, pixels):
            features = list(self.model.backbone((pixels - self.mean) / self.std).values())
            result = self.model.head(features)
            return result["bbox_regression"], result["cls_logits"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".onnx-export-") as temporary:
        staging = Path(temporary) / "runtime"
        staging.mkdir()
        for name in ("dino", "siglip", "detector"):
            graph = Path(temporary) / (name + "-float.onnx")
            wrapper = Detector() if name == "detector" else Encoder(name)
            size = 320 if name == "detector" else 224
            with torch.inference_mode():
                torch.onnx.export(wrapper.eval(), torch.zeros(1, 3, size, size), str(graph),
                                  input_names=["pixels"], output_names=["offsets", "logits"] if name == "detector" else ["features"],
                                  opset_version=17, dynamo=False, do_constant_folding=True)
            target = staging / (name + ".onnx")
            if name == "detector":
                graph.rename(target)
            else:
                quantizer = MatMulNBitsQuantizer(str(graph), bits=8, block_size=32, is_symmetric=True,
                                               accuracy_level=1, op_types_to_quantize=("MatMul",))
                quantizer.process()
                quantizer.model.save_model_to_file(str(target), False)
            onnx.checker.check_model(str(target))
        with torch.inference_mode():
            images, _ = engine.detector.transform([torch.zeros(3, 320, 320)], None)
            features = list(engine.detector.backbone(images.tensors).values())
            anchors = engine.detector.anchor_generator(images, features)[0].numpy()
        np.savez(staging / "anchors.npz", anchors=anchors)
        model = ROOT / "data/processed/vision-recognition-v3-brand"
        metadata = dict(schema_version=1, precision="weight_only_int8", source_dino=list(DINO), source_siglip=list(SIGLIP),
                        box_weights=list(engine.detector.box_coder.weights), bbox_xform_clip=engine.detector.box_coder.bbox_xform_clip,
                        nms_thresh=engine.detector.nms_thresh, topk_candidates=engine.detector.topk_candidates,
                        detections_per_img=engine.detector.detections_per_img,
                        model_metadata_sha256=file_hash(model / "model.json"), classifier_sha256=file_hash(model / "classifier.npz"),
                        files={p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in staging.iterdir()})
        (staging / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if output.exists():
            raise ValueError("Thư mục đích vừa được tạo; giữ nguyên dữ liệu.")
        staging.rename(output)
    return output


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(f"Exported: {export(args.output)}. Chưa được dùng mặc định trước khi chạy validate_compact_runtime.py.")
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, str(error) + "\n")
