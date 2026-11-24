import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.setup_cloud import PROJECT_URL, save_config, setup, verify_cloud


class CloudSetupTests(unittest.TestCase):
    def fixture(self, root):
        image = io.BytesIO()
        Image.new('RGB', (24, 16), 'blue').save(image, format='PNG')
        data = image.getvalue()
        model = root / 'data/processed/vision-recognition-v3-brand/model.json'
        model.parent.mkdir(parents=True)
        model.write_text(json.dumps({'train_records': [{'class_id': 'model_3', 'sha256': hashlib.sha256(data).hexdigest()}]}))
        return data

    def test_probe_reads_one_reference_without_loading_models_or_catalog(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root = Path(temporary)
            data = self.fixture(root)
            client = Mock()
            client.reference_image.return_value = data
            report = verify_cloud(root, client)
            self.assertEqual(report, {'class_id': 'model_3', 'bytes': len(data), 'width': 24, 'height': 16})
            client.reference_image.assert_called_once_with(hashlib.sha256(data).hexdigest(), 'model_3')
            client.catalog.assert_not_called()
            self.assertFalse((root / '.env').exists())

    def test_setup_saves_only_after_success_and_never_overwrites(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root = Path(temporary)
            data = self.fixture(root)
            client = Mock()
            client.reference_image.return_value = data
            with patch('src.setup_cloud.sys.stdin.isatty', return_value=True), \
                    patch('src.setup_cloud.getpass.getpass', return_value='fixture-server-secret-key'), \
                    patch('src.setup_cloud.CloudData', return_value=client) as constructor:
                result = setup(root)
            constructor.assert_called_once_with(PROJECT_URL, 'fixture-server-secret-key')
            self.assertEqual(result['width'], 24)
            saved = (root / '.env').read_bytes()
            with patch('src.setup_cloud.getpass.getpass') as prompt:
                with self.assertRaisesRegex(ValueError, 'Đã có'):
                    setup(root)
                prompt.assert_not_called()
            self.assertEqual(saved, (root / '.env').read_bytes())

    def test_invalid_key_network_failure_and_noninteractive_input_write_nothing(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root = Path(temporary)
            self.fixture(root)
            with patch('src.setup_cloud.sys.stdin.isatty', return_value=False):
                with self.assertRaisesRegex(ValueError, 'terminal'):
                    setup(root)
            with patch('src.setup_cloud.sys.stdin.isatty', return_value=True), \
                    patch('src.setup_cloud.getpass.getpass', return_value='fixture-key\nCARVISION_X=injected'):
                with self.assertRaises(ValueError):
                    setup(root)
            client = Mock()
            client.reference_image.side_effect = ValueError('Supabase HTTP 401')
            with patch('src.setup_cloud.sys.stdin.isatty', return_value=True), \
                    patch('src.setup_cloud.getpass.getpass', return_value='fixture-server-secret-key'), \
                    patch('src.setup_cloud.CloudData', return_value=client):
                with self.assertRaisesRegex(ValueError, '401'):
                    setup(root)
            self.assertFalse((root / '.env').exists())
            with self.assertRaises(ValueError):
                save_config(root, 'a\nb')
