import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.check_environment import check


class EnvironmentTests(unittest.TestCase):
    def test_missing_model_and_cache_are_reported_and_cloud_is_optional(self):
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {"CARVISION_VISION_RUN": "", "CARVISION_SUPABASE_URL": "", "CARVISION_SUPABASE_KEY": ""}):
            checks = check(root)
            self.assertIn(("model.json", False, True), checks)
            self.assertIn(("Cache detector", False, True), checks)
            self.assertIn(("Ảnh Supabase: cấu hình máy chủ (tùy chọn)", False, False), checks)

    def test_malformed_metadata_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "data/processed/model"
            model.mkdir(parents=True)
            for content in ("not-json", json.dumps({"dino": ["../../escape", "x"]})):
                (model / "model.json").write_text(content)
                with patch.dict("os.environ", {"CARVISION_VISION_RUN": str(model)}):
                    self.assertIn(("Model metadata đọc được", False, True), check(root))
