"""Synthetic tiny checkpoint: the image-only path preserves normalized image features."""

import importlib.util
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from src.vision_features import VisionFeatures


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Install requirements-vision.txt")
class VisionRuntimeTests(unittest.TestCase):
    def test_incomplete_image_weights_are_rejected_without_loading_full_text_model(self):
        with patch("torchvision.models.detection.ssdlite320_mobilenet_v3_large"), \
                patch("pathlib.Path.is_file", return_value=True), \
                patch("torch.hub.set_dir"), patch("transformers.AutoImageProcessor.from_pretrained"), \
                patch("transformers.AutoModel.from_pretrained") as full_model, \
                patch("transformers.AutoProcessor.from_pretrained") as full_processor, \
                patch("transformers.SiglipVisionModel.from_pretrained", return_value=(Mock(), {"missing_keys": ["vision-fixture"]})):
            with self.assertRaisesRegex(ValueError, "thiếu hoặc lệch"):
                VisionFeatures(image_only=True)
            self.assertEqual(full_model.call_count, 1)  # DINO only; no full SigLIP text tower.
            full_processor.assert_not_called()

    def test_image_only_checkpoint_matches_full_encoder_without_text_weights(self):
        import torch
        from transformers import SiglipConfig, SiglipModel, SiglipVisionModel

        torch.manual_seed(42)
        config = SiglipConfig(
            text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=4, image_size=32, patch_size=16))
        full = SiglipModel(config).eval()
        pixels = torch.rand(2, 3, 32, 32)
        with TemporaryDirectory() as temporary:
            full.save_pretrained(temporary)
            image_only = SiglipVisionModel.from_pretrained(temporary, local_files_only=True).eval()
            self.assertFalse(hasattr(image_only, "text_model"))
            engine = VisionFeatures.__new__(VisionFeatures)
            engine.torch = torch
            engine.siglip_processor = lambda **kwargs: {"pixel_values": pixels}
            engine.siglip = full
            engine.image_only = False
            before = engine.encode([None, None], "siglip")
            engine.siglip = image_only
            engine.image_only = True
            after = engine.encode([None, None], "siglip")
            np.testing.assert_array_equal(before, after)
            engine.image_only = True
            with self.assertRaises(ValueError):
                engine.texts(["not loaded for photo recognition"])
