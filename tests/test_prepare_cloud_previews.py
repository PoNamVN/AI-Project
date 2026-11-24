import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from src.prepare_cloud_previews import prepare


class PreviewPreparationTests(unittest.TestCase):
    def test_preflight_no_writes_until_every_source_passes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "data/raw/fixture/model_3/car.png"
            photo.parent.mkdir(parents=True)
            output = io.BytesIO()
            Image.new("RGB", (1000, 700), "blue").save(output, format="PNG")
            photo.write_bytes(output.getvalue())
            digest = hashlib.sha256(output.getvalue()).hexdigest()
            reference = dict(source_path="fixture/model_3/car.png", class_id="model_3", sha256=digest)
            model = root / "data/processed/vision-recognition-v3-brand/model.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({"train_records": [reference]}))
            cloud = Mock()
            cloud.catalog.return_value = [dict(sha256=digest, class_id="model_3", decision="keep", byte_size=len(output.getvalue()))]
            cloud.publish_thumbnail.return_value = True
            with patch.dict("os.environ", {"CARVISION_VISION_RUN": str(model.parent)}):
                report = prepare(root, client=cloud)
                self.assertEqual(report["references"], 1)
                cloud.publish_thumbnail.assert_not_called()
                self.assertEqual(prepare(root, publish=True, client=cloud)["new_links"], 1)
                data = cloud.publish_thumbnail.call_args.args[1]
                with Image.open(io.BytesIO(data)) as image:
                    self.assertEqual(image.format, "WEBP")
                    self.assertLessEqual(image.width, 500)
                    self.assertLessEqual(image.height, 320)
                cloud.publish_thumbnail.reset_mock()
                photo.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    prepare(root, publish=True, client=cloud)
                cloud.publish_thumbnail.assert_not_called()
