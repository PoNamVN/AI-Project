import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from src.install_runtime import install, file_hash, selected_path
from src.serve_public import command

spec = importlib.util.spec_from_file_location("bundle_fixture", Path(__file__).resolve().parents[1] / "tests/test_runtime_bundle.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class PublicDeploymentTests(unittest.TestCase):
    def test_public_command_requires_cloud_runtime_and_forces_internal_tools_off(self):
        with patch.dict(os.environ, {"PORT": "10000", "CARVISION_ENABLE_DEV_TOOLS": "1"}), \
                patch("src.serve_public.CloudData") as cloud, patch("src.serve_public.check", return_value=[("fixture", True, True)]):
            args = command()
            self.assertEqual(os.environ["CARVISION_ENABLE_DEV_TOOLS"], "0")
            self.assertIn("--server.address=0.0.0.0", args)
            self.assertIn("--server.maxUploadSize=30", args)
            self.assertNotIn("--server.enableXsrfProtection=false", args)
            os.environ["PORT"] = "10000;unsafe"
            with self.assertRaises(ValueError): command()
            os.environ["PORT"] = "10000"
            cloud.side_effect = ValueError("fixture missing config")
            with self.assertRaises(ValueError): command()
        with patch("src.serve_public.CloudData"), patch("src.serve_public.check", return_value=[("fixture", False, True)]):
            with self.assertRaises(ValueError): command()

    def test_installer_verifies_hashes_ignores_photos_and_preserves_existing_files(self):
        with TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            source.mkdir()
            target.mkdir()
            fixture.RuntimeBundleTests().fixture(source)
            archive = source / ".kickoff-build/full.zip"
            fixture.bundle.build(source, archive)
            digest = file_hash(archive)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                install(archive, "0" * 64, target)
            self.assertFalse((target / "data").exists())
            self.assertGreater(install(archive, digest, target), 2)
            self.assertFalse((target / "data/raw").exists())
            self.assertFalse((target / ".env").exists())
            self.assertGreater(install(archive, digest, target), 2)
            existing = target / fixture.bundle.MODEL / "model.json"
            existing.write_bytes(b"keep-this-file")
            with self.assertRaisesRegex(ValueError, "giữ nguyên"):
                install(archive, digest, target)
            self.assertEqual(existing.read_bytes(), b"keep-this-file")

    def test_installer_rejects_unlisted_files_and_corrupt_entries_before_writing(self):
        for name, body, recorded in (("../escape.py", b"bad", None),
                                      ("data/processed/vision-recognition-v3-brand/model.json", b"changed", b"original")):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "bad.zip"
                records = []
                with zipfile.ZipFile(archive, "w") as output:
                    paths = {"data/processed/vision-recognition-v3-brand/model.json": b"{}",
                             "data/processed/vision-recognition-v3-brand/classifier.npz": b"fixture", name: body}
                    for relative, data in paths.items():
                        output.writestr(relative, data)
                        hashed = recorded if relative == name and recorded else data
                        records.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(hashed).hexdigest()))
                    output.writestr(".kickoff-build/runtime-bundle-manifest.json", json.dumps(dict(schema_version=1, files=records)))
                with self.assertRaises(ValueError): install(archive, file_hash(archive), root)
                self.assertFalse((root / "data").exists())
                self.assertFalse(selected_path("../escape.py"))
