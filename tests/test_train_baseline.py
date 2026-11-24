import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'Install requirements-baseline.txt for CNN tests')
class BaselineTests(unittest.TestCase):
    def test_original_loader_checks_hash_and_uses_full_resolution(self):
        import hashlib
        from unittest.mock import patch
        from PIL import Image
        from src.train_transfer import load_originals

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB', (500, 300), 'red').save(root / 'car.png')
            digest = hashlib.sha256((root / 'car.png').read_bytes()).hexdigest()
            entries = [('train/model_3/1.png', root / 'unused.png', 'model_3')]
            records = {entries[0][0]: {'source_path': 'car.png', 'sha256': digest}}
            images, targets = load_originals(entries, ['model_3'], records, root)
            self.assertEqual(images.shape, (1, 224, 224, 3))
            self.assertEqual(targets.tolist(), [0])
            with patch('src.train_transfer.MAX_PIXELS', 10):
                with self.assertRaisesRegex(ValueError, 'memory limit'):
                    load_originals(entries, ['model_3'], records, root)
            records[entries[0][0]]['sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_originals(entries, ['model_3'], records, root)
            records[entries[0][0]]['source_path'] = '../outside.png'
            with self.assertRaisesRegex(ValueError, 'Invalid original path'):
                load_originals(entries, ['model_3'], records, root)

    def test_mobilenet_preprocessing_freeze_and_inference_augmentation(self):
        from unittest.mock import patch
        import numpy as np
        import tensorflow as tf
        from src.train_baseline import build_model

        constructor = tf.keras.applications.MobileNetV2
        with patch('src.train_baseline.tf.keras.applications.MobileNetV2',
                   side_effect=lambda **kw: constructor(**{**kw, 'weights': None})):
            model = build_model(5, 'mobilenetv2', augment=True)
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
        self.assertFalse(backbone.trainable)
        self.assertEqual(sum(int(np.prod(v.shape)) for v in model.trainable_weights), 6405)
        scale = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling))
        np.testing.assert_allclose(scale(tf.constant([0., 127.5, 255.])).numpy(), [-1, 0, 1], atol=1e-6)
        image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        np.testing.assert_allclose(model(image, training=False), model(image, training=False))
        self.assertEqual(model.output_shape, (None, 5))
        from src.train_transfer import enable_fine_tuning
        enable_fine_tuning(model)
        self.assertTrue(backbone.trainable)
        self.assertFalse(backbone.layers[1].trainable)
        self.assertTrue(any(layer.trainable for layer in backbone.layers[-30:]))
        self.assertTrue(all(not layer.trainable for layer in backbone.layers
                            if isinstance(layer, tf.keras.layers.BatchNormalization)))

    def test_class_metrics_include_missed_classes(self):
        import numpy as np
        from src.train_baseline import evaluate

        metrics = evaluate(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 1]), 3)
        self.assertEqual(metrics['correct'], 2)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['macro_f1'], 0.5)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0, 0], [0, 1, 0], [0, 2, 0]])

    def test_reject_vehicle_leakage(self):
        from src.train_baseline import load_manifest

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fields = ('relative_path', 'class_id', 'vehicle_id', 'sha256', 'split')
            with (root / 'splits.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                for split, digest in [('train', 'a'), ('test', 'b')]:
                    relative = f'{split}/model_3/1.png'
                    target = root / relative
                    target.parent.mkdir(parents=True)
                    target.touch()
                    writer.writerow((relative, 'model_3', 'same-car', digest, split))
            with self.assertRaisesRegex(ValueError, 'Vehicle appears'):
                load_manifest(root)
