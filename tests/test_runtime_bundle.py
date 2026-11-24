import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('build_runtime_bundle', Path(__file__).resolve().parents[1] / 'scripts/build_runtime_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class RuntimeBundleTests(unittest.TestCase):
    def fixture(self, root):
        reference = b'reference-placeholder-not-a-real-dataset'
        model = {'dino': ['facebook/dinov2-small', 'a' * 40],
                 'siglip': ['google/siglip2-base-patch16-224', 'b' * 40],
                 'train_records': [{'source_path': 'fixture/model_s/car.jpg', 'class_id': 'model_s',
                                    'sha256': hashlib.sha256(reference).hexdigest(), 'source_url': 'https://example.org/car',
                                    'license': 'test-only', 'artist_html': 'test', 'license_url': 'https://example.org/license'}]}
        paths = {bundle.MODEL + '/model.json': json.dumps(model).encode(), bundle.MODEL + '/classifier.npz': b'fixture',
                 bundle.DETECTOR: b'fixture', 'data/raw/fixture/model_s/car.jpg': reference,
                 'docs/runtime_notices.md': b'fixture', 'docs/licenses/Apache-2.0.txt': b'fixture',
                 'docs/licenses/Torchvision-BSD.txt': b'fixture', '.env': b'never-export-this-fixture'}
        paths.update({'data/processed/' + name: b'sha256\n' for name in bundle.HISTORY})
        for key in ('dino', 'siglip'):
            name, rev = model[key]
            base = f'.kickoff-build/huggingface/models--{name.replace("/", "--")}/snapshots/{rev}'
            names = ['config.json', 'preprocessor_config.json', 'model.safetensors']
            if key == 'siglip':
                names += ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
            paths.update({base + '/' + n: b'fixture' for n in names})
        for relative, data in paths.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return model

    def test_allowlist_hashes_and_no_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            target = root / '.kickoff-build/runtime.zip'
            bundle.build(root, target)
            with zipfile.ZipFile(target) as archive:
                self.assertNotIn('.env', archive.namelist())
                manifest = json.loads(archive.read('.kickoff-build/runtime-bundle-manifest.json'))
                for row in manifest['files']:
                    data = archive.read(row['path'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])
                    self.assertEqual(len(data), row['bytes'])
            self.assertEqual(bundle.sha256(target), target.with_suffix('.zip.sha256').read_text().split()[0])
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                bundle.build(root, target)
            with self.assertRaisesRegex(ValueError, 'under .kickoff-build'):
                bundle.build(root, root / 'public.zip')

    def test_bad_reference_and_failed_build_publish_nothing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.fixture(root)
            required = root / 'data/processed' / bundle.HISTORY[0]
            required.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or invalid'):
                bundle.runtime_files(root)
            self.fixture(root)
            model['train_records'][0]['source_path'] = '../model_s/car.jpg'
            (root / bundle.MODEL / 'model.json').write_text(json.dumps(model))
            with self.assertRaises(ValueError):
                bundle.runtime_files(root)
            self.fixture(root)
            with patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaisesRegex(ValueError, 'regular files'):
                    bundle.runtime_files(root)
            (root / 'data/raw/fixture/model_s/car.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                bundle.runtime_files(root)
            self.fixture(root)
            target = root / '.kickoff-build/failed.zip'
            with patch.object(bundle.zipfile.ZipFile, 'writestr', side_effect=OSError('simulated disk error')):
                with self.assertRaises(OSError):
                    bundle.build(root, target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.zip.sha256').exists())

    def test_cloud_bundle_excludes_dataset_and_preserves_licenses(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            target = root / '.kickoff-build/cloud.zip'
            bundle.build(root, target, cloud_only=True)
            with zipfile.ZipFile(target) as archive:
                self.assertFalse(any(name.startswith('data/raw/') or name.endswith('.csv') for name in archive.namelist()))
                self.assertIn('.kickoff-build/runtime-licenses/runtime_notices.md', archive.namelist())
                self.assertTrue(json.loads(archive.read('.kickoff-build/runtime-bundle-manifest.json'))['cloud_references_required'])

    def test_compact_bundle_needs_matching_validation_and_does_not_include_torch_cache(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            runtime = root / 'data/processed/vision-onnx-v31'
            runtime.mkdir()
            config = dict(model_metadata_sha256=bundle.sha256(root / bundle.MODEL / 'model.json'),
                          classifier_sha256=bundle.sha256(root / bundle.MODEL / 'classifier.npz'), files={})
            for name in ('dino.onnx', 'siglip.onnx', 'detector.onnx', 'anchors.npz'):
                (runtime / name).write_bytes(b'fixture')
                config['files'][name] = dict(bytes=7, sha256=bundle.sha256(runtime / name))
            (runtime / 'runtime.json').write_text(json.dumps(config))
            (runtime / 'validation.json').write_text(json.dumps(dict(passed=True, runtime_manifest_sha256=bundle.sha256(runtime / 'runtime.json'))))
            target = root / '.kickoff-build/compact.zip'
            bundle.build(root, target, compact=True)
            with zipfile.ZipFile(target) as archive:
                self.assertIn('data/processed/vision-onnx-v31/dino.onnx', archive.namelist())
                self.assertFalse(any('huggingface/' in name or name.startswith('data/raw/') for name in archive.namelist()))
            (runtime / 'dino.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                bundle.runtime_files(root, compact=True)
