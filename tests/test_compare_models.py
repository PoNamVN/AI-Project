import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.compare_models import compare


class ComparisonTests(unittest.TestCase):
    def test_validation_selection_and_different_manifest_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for name, val_loss in [('a', .4), ('b', .2)]:
                run = root / name
                run.mkdir()
                metrics = dict(class_ids=['model_3', 'model_y'], manifest_sha256='same', epochs_run=1,
                               best_epoch=1, history={'val_loss': [val_loss]},
                               test=dict(images=2, correct=1, accuracy=.5, macro_f1=1/3,
                                         confusion_matrix=[[1, 0], [1, 0]]))
                (run / 'metrics.json').write_text(json.dumps(metrics))
                with (run / 'test_predictions.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(('relative_path', 'actual', 'predicted'))
                    writer.writerows([('test/model_3/1.png', 'model_3', 'model_3'),
                                      ('test/model_y/2.png', 'model_y', 'model_3')])
                runs.append(run)
            compare(runs, root / 'report')
            self.assertEqual(json.loads((root / 'report/selection.json').read_text())['selected_run'], 'b')
            metrics['manifest_sha256'] = 'different'
            (runs[1] / 'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'same manifest'):
                compare(runs, root / 'invalid')
