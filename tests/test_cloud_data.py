import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from src.cloud_data import CloudData, NoRedirect


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "CARVISION_SUPABASE_URL": "https://abcdefghijklmnopqrst.supabase.co",
            "CARVISION_SUPABASE_KEY": "fake-server-key-for-test-only",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cloud = CloudData()
        output = io.BytesIO()
        Image.new("RGB", (24, 16), "blue").save(output, format="PNG")
        self.data = output.getvalue()
        digest = hashlib.sha256(self.data).hexdigest()
        self.row = dict(dataset="test-v1", relative_path="model_3/example.png", class_id="model_3",
                        decision="keep", sha256=digest, byte_size=len(self.data),
                        object_path=f"sha256/{digest}.png", purpose="evaluation_only", split="unsplit",
                        review_status="candidate", source_url="https://example.org/car", license="CC0",
                        provenance={"artist_html": "Test artist", "license_url": "https://example.org/license"})

    def test_requires_valid_https_host_and_configuration(self):
        for url in ("", "http://abcdefghijklmnopqrst.supabase.co", "https://evil.example", "https://abcdefghijklmnopqrst.supabase.co/redirect"):
            with self.subTest(url=url), patch.dict(os.environ, {"CARVISION_SUPABASE_URL": url}):
                with self.assertRaises(ValueError): CloudData()

    def test_catalog_filter_and_pagination(self):
        with patch.object(self.cloud, "_get", side_effect=[json.dumps([self.row] * 500).encode(), b"[]"]) as get:
            self.assertEqual(len(self.cloud.catalog("test-v1")), 500)
            self.assertIn("offset=500", get.call_args_list[1].args[0])
            self.assertIn("dataset=eq.test-v1", get.call_args.args[0])
        with self.assertRaises(ValueError): self.cloud.catalog("../../other")

    def test_image_checksum_size_and_object_path(self):
        with patch.object(self.cloud, "_get", return_value=self.data):
            self.assertEqual(self.cloud.image(self.row), self.data)
            for updates in ({"object_path": "../secret"}, {"sha256": "0" * 64}, {"byte_size": 0}, {"byte_size": len(self.data) + 1}):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    self.cloud.image({**self.row, **updates})
        with patch.object(self.cloud, "_get", return_value=b"corrupted"):
            with self.assertRaises(ValueError): self.cloud.image(self.row)

    def test_reference_fetch_filters_by_checksum_and_never_downloads_catalog(self):
        row = {**self.row, "uploaded": True}
        with patch.object(self.cloud, "_get", side_effect=[json.dumps([row]).encode(), self.data]) as get:
            self.assertEqual(self.cloud.reference_image(row["sha256"], row["class_id"]), self.data)
            query = get.call_args_list[0].args[0]
            self.assertIn("limit=1", query)
            self.assertIn("sha256=eq.", query)
            self.assertIn("class_id=eq.model_3", query)
            self.assertEqual(get.call_count, 2)
        for response in (b"{}", b"[]", b"[null]", b"invalid", json.dumps([{**row, "sha256": "f" * 64}]).encode()):
            with self.subTest(response=response), patch.object(self.cloud, "_get", return_value=response) as get:
                with self.assertRaises(ValueError):
                    self.cloud.reference_image(row["sha256"], row["class_id"])
                self.assertEqual(get.call_count, 1)
        with patch.object(self.cloud, "_get") as get:
            with self.assertRaises(ValueError):
                self.cloud.reference_image("bad", "model_3")
            get.assert_not_called()

    def test_rejects_malformed_catalog(self):
        for result in (b"{}", b"[null]", b"[{}]", b"invalid JSON"):
            with self.subTest(result=result), patch.object(self.cloud, "_get", return_value=result):
                with self.assertRaises(ValueError): self.cloud.catalog()

    def test_thumbnail_is_pinned_to_original_and_verified(self):
        row = {**self.row, "uploaded": True}
        digest = hashlib.sha256(self.data).hexdigest()
        preview = dict(source_sha256=row["sha256"], class_id="model_3", sha256=digest,
                       byte_size=len(self.data), object_path=f"previews/v1/{row['sha256']}/{digest}.webp")
        row["provenance"] = {**row["provenance"], "preview_v1": preview}
        with patch.object(self.cloud, "_get", side_effect=[json.dumps([row]).encode(), self.data]) as get:
            self.assertEqual(self.cloud.reference_image(row["sha256"], "model_3", thumbnail=True), self.data)
            self.assertIn("/previews/v1/", get.call_args.args[0])
        for changes in ({"source_sha256": "f" * 64}, {"class_id": "model_y"}, {"object_path": "../escape"}, {"byte_size": 250001}):
            with self.subTest(changes=changes), patch.object(self.cloud, "_get") as get:
                with self.assertRaises(ValueError): self.cloud.thumbnail_image(row, {**preview, **changes})
                get.assert_not_called()
        with patch.object(self.cloud, "_get", return_value=b"wrong"):
            with self.assertRaises(ValueError): self.cloud.thumbnail_image(row, preview)

    def test_publication_never_overwrites_or_drops_existing_provenance(self):
        row = {**self.row, "uploaded": True}
        with patch("src.cloud_data.build_opener") as opener, patch.object(self.cloud, "thumbnail_image", return_value=self.data):
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps([{"sha256": row["sha256"]}]).encode()
            self.assertTrue(self.cloud.publish_thumbnail(row, self.data))
            requests = opener.return_value.open.call_args_list
            self.assertEqual(requests[0].args[0].get_header("X-upsert"), "false")
            request = requests[1].args[0]
            self.assertEqual(request.method, "PATCH")
            self.assertIn("provenance=eq.", request.full_url)
            self.assertEqual(json.loads(request.data)["provenance"]["artist_html"], "Test artist")
            self.assertNotIn("preview_v1", row["provenance"])
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = b"[]"
            with self.assertRaisesRegex(ValueError, "Bản ghi đã đổi"):
                self.cloud.publish_thumbnail(row, self.data)

    def test_download_preserves_provenance_and_excludes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with patch.object(self.cloud, "catalog", return_value=[self.row, {**self.row, "decision": "exclude"}]), patch.object(self.cloud, "_get", return_value=self.data):
                self.assertEqual(self.cloud.download("test-v1", destination), 1)
            self.assertEqual((destination / self.row["relative_path"]).read_bytes(), self.data)
            self.assertIn("Test artist", (destination / "provenance.csv").read_text(encoding="utf-8-sig"))
            self.assertEqual(json.loads((destination / "cloud_manifest.json").read_text())[0]["purpose"], "evaluation_only")
            with self.assertRaises(ValueError): self.cloud.download("test-v1", destination)

    def test_download_rejects_paths_before_network_and_cleans_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            for relative in ("../escape.png", "/model_3/test.png", "model_3/../../escape.png", "model_3/C:other.png", "model_3/test\\x.png"):
                with patch.object(self.cloud, "catalog", return_value=[{**self.row, "relative_path": relative}]), patch.object(self.cloud, "image") as image:
                    with self.subTest(relative=relative), self.assertRaises(ValueError):
                        self.cloud.download("test-v1", destination)
                    image.assert_not_called()
            with patch.object(self.cloud, "catalog", return_value=[self.row]), patch.object(self.cloud, "image", side_effect=ValueError("network")):
                with self.assertRaises(ValueError): self.cloud.download("test-v1", destination)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_http_error_redacts_credentials_and_refuses_redirects(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example"))
        with patch("src.cloud_data.build_opener") as opener:
            opener.return_value.open.side_effect = HTTPError("https://test", 403, self.cloud.key, {}, None)
            with self.assertRaises(ValueError) as caught: self.cloud._get("/path", 1)
            self.assertNotIn(self.cloud.key, str(caught.exception))

    def test_http_read_is_bounded(self):
        with patch("src.cloud_data.build_opener") as opener:
            response = opener.return_value.open.return_value.__enter__.return_value
            response.read.return_value = b"abc"
            with self.assertRaises(ValueError): self.cloud._get("/path", 2)
            response.read.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
