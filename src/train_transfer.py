"""Adapt MobileNetV2 using original images, class balancing and cautious fine-tuning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from src.data_audit import MAX_PIXELS
from src.train_baseline import build_model, evaluate, load_manifest


def prepare_image(source):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Image exceeds memory limit')
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                return np.asarray(ImageOps.pad(rgb, (224, 224), method=Image.Resampling.LANCZOS, color='black'))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Image exceeds memory limit') from error


def load_originals(entries, labels, records, originals):
    images, targets = [], []
    originals = originals.resolve()
    for relative, _, label in entries:
        record = records[relative]
        source = (originals / record['source_path']).resolve()
        if not source.is_relative_to(originals) or not source.is_file():
            raise ValueError(f'Invalid original path: {relative}')
        with source.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Original image changed: {relative}')
        images.append(prepare_image(source))
        targets.append(labels.index(label))
    return np.stack(images), np.asarray(targets, dtype=np.int32)


def enable_fine_tuning(model):
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - 30 and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return backbone


def train(dataset, originals, output):
    rows, labels = load_manifest(dataset)
    with (dataset / 'splits.csv').open(encoding='utf-8-sig', newline='') as handle:
        records = {row['relative_path']: row for row in csv.DictReader(handle)}
    x, y = load_originals(rows['train'], labels, records, originals)
    x_val, y_val = load_originals(rows['validation'], labels, records, originals)
    counts = np.bincount(y, minlength=len(labels))
    weights = {i: float(len(y) / (len(labels) * count)) for i, count in enumerate(counts)}
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    model = build_model(len(labels), 'mobilenetv2', image_size=(224, 224))
    output.mkdir(parents=True)
    histories = {}
    for phase, rate, epochs in [('head', 0.001, 25), ('fine_tune', 0.00001, 15)]:
        if phase == 'fine_tune':
            enable_fine_tuning(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f'Phase: {phase}', flush=True)
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        histories[phase] = model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs,
                                     batch_size=8, class_weight=weights, callbacks=[stop], verbose=2).history
        model.save(output / f'{phase}.keras')
    selected = min(histories, key=lambda phase: min(histories[phase]['val_loss']))
    model = tf.keras.models.load_model(output / f'{selected}.keras')
    val_probs = model.predict(x_val, verbose=0)
    # Hold back test image decoding until both training stages and selection finish.
    x_test, actual = load_originals(rows['test'], labels, records, originals)
    probabilities = model.predict(x_test, verbose=0)
    predicted = probabilities.argmax(axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -min(3, len(labels)):]
    metrics = {
        'architecture': 'mobilenetv2_224_balanced_finetune', 'tensorflow': tf.__version__,
        'seed': 42, 'input_size': [224, 224], 'batch_size': 8,
        'manifest_sha256': hashlib.sha256((dataset / 'splits.csv').read_bytes()).hexdigest(),
        'class_ids': labels, 'class_weights': {labels[i]: w for i, w in weights.items()},
        'selection': 'lowest validation loss across the two stages', 'selected_stage': selected,
        'selected_stage_epoch': int(np.argmin(histories[selected]['val_loss'])) + 1,
        'best_epoch': int(np.argmin(histories[selected]['val_loss'])) + 1 + (len(histories['head']['loss']) if selected == 'fine_tune' else 0),
        'learning_rates': {'head': 0.001, 'fine_tune': 0.00001},
        'epochs_run': sum(len(h['loss']) for h in histories.values()),
        'history': {key: histories['head'][key] + histories['fine_tune'][key] for key in histories['head']},
        'histories': histories, 'split_sizes': {key: len(value) for key, value in rows.items()},
        'validation': evaluate(y_val, val_probs.argmax(axis=1), len(labels)),
        'test': evaluate(actual, predicted, len(labels)),
        'top3_test_accuracy': float(np.mean(np.any(top3 == actual[:, None], axis=1))),
        'limitations': ['Known 14-image development test already inspected in earlier experiments.',
                        'Single seed, estimated labels/groups, no independent street-photo accuracy estimate.'],
    }
    model.save(output / 'model.keras')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (output / 'test_predictions.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('relative_path', 'actual', 'predicted'))
        writer.writerows((relative, label, labels[p]) for (relative, _, label), p in zip(rows['test'], predicted))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset, originals, output = args.dataset.resolve(), args.originals.resolve(), args.output.resolve()
        processed = (Path(__file__).resolve().parents[1] / 'data/processed').resolve()
        if (output == processed or not output.is_relative_to(processed) or output.exists()
                or output.is_relative_to(dataset) or output.is_relative_to(originals)):
            raise ValueError('Choose a new output folder under data/processed, outside image inputs')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.transfer-', dir=output.parent) as temporary:
            staging = Path(temporary) / 'run'
            metrics = train(dataset, originals, staging)
            staging.rename(output)
        print(json.dumps({key: metrics[key] for key in ['selected_stage', 'test', 'top3_test_accuracy']}, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
