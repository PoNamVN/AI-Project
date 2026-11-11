import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'Install requirements-baseline.txt for CNN tests')
class BaselineTests(unittest.TestCase):
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
