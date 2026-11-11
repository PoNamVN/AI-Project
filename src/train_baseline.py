"""Train and evaluate the Week 3 Tesla CNN on a frozen split."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import tensorflow as tf

from src.data_audit import MODELS
from src.prepare_dataset import IMAGE_SIZE, SPLITS


def load_manifest(root):
    rows = defaultdict(list)
    vehicles, hashes, paths = {}, set(), set()
    with (root / "splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "class_id", "vehicle_id", "sha256", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("splits.csv is missing required columns")
        for row in reader:
            split, label = row["split"], row["class_id"]
            relative = row["relative_path"] or ""
            path = (root / relative).resolve()
            if (split not in SPLITS or label not in MODELS or not path.is_relative_to(root)
                    or path.parent != root / split / label or not path.is_file() or path in paths):
                raise ValueError(f"Invalid or repeated split entry: {relative}")
            vehicle, digest = row["vehicle_id"], row["sha256"]
            if not vehicle or not digest or vehicles.setdefault(vehicle, (split, label)) != (split, label):
                raise ValueError(f"Vehicle appears in multiple splits or classes: {relative}")
            if digest in hashes:
                raise ValueError(f"Repeated original image: {relative}")
            hashes.add(digest)
            paths.add(path)
            rows[split].append((relative, path, label))
    labels = sorted({label for _, _, label in rows["train"]})
    if len(labels) < 2 or any({label for _, _, label in rows[split]} != set(labels) for split in SPLITS):
        raise ValueError("Each split must contain the same two or more classes")
    return rows, labels


def load_images(entries, labels):
    images, targets = [], []
    for _, path, label in entries:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != IMAGE_SIZE:
                raise ValueError(f"Expected a prepared RGB image of size {IMAGE_SIZE}: {path}")
            images.append(np.asarray(image))
        targets.append(labels.index(label))
    return np.stack(images), np.asarray(targets, dtype=np.int32)


def evaluate(actual, predicted, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    true_positive = matrix.diagonal()
    total = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * true_positive, total, out=np.zeros(classes), where=total != 0)
    return {
        "correct": int(true_positive.sum()), "images": len(actual),
        "accuracy": float(true_positive.sum() / len(actual)),
        "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
    }


def train(root, output, seed, epochs):
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    rows, labels = load_manifest(root)
    x_train, y_train = load_images(rows["train"], labels)
    x_val, y_val = load_images(rows["validation"], labels)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input((*IMAGE_SIZE, 3)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=8, callbacks=[stop], verbose=2).history

    x_test, y_test = load_images(rows["test"], labels)
    predicted = model.predict(x_test, batch_size=8, verbose=0).argmax(axis=1)
    majority = int(np.bincount(y_train, minlength=len(labels)).argmax())
    metrics = {
        "tensorflow": tf.__version__, "seed": seed, "epochs_limit": epochs,
        "epochs_run": len(history["loss"]), "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "selection": "lowest validation loss", "batch_size": 8, "parameters": model.count_params(),
        "manifest_sha256": hashlib.sha256((root / "splits.csv").read_bytes()).hexdigest(),
        "class_ids": labels, "split_sizes": {split: len(rows[split]) for split in SPLITS},
        "test": evaluate(y_test, predicted, len(labels)),
        "majority_class": labels[majority],
        "majority_test": evaluate(y_test, np.full_like(y_test, majority), len(labels)),
        "history": history,
    }
    output.mkdir(parents=True)
    model.save(output / "baseline.keras")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "test_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "actual", "predicted"))
        writer.writerows((relative, label, labels[index])
                         for (relative, _, label), index in zip(rows["test"], predicted))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    try:
        root, output = args.dataset.resolve(), args.output.resolve()
        processed = (Path(__file__).resolve().parents[1] / "data" / "processed").resolve()
        if (output == processed or not output.is_relative_to(processed)
                or output.is_relative_to(root) or output.exists()):
            raise ValueError("Choose a new output folder under data/processed, outside the input dataset")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temporary:
            staging = Path(temporary) / "run"
            metrics = train(root, staging, args.seed, args.epochs)
            staging.rename(output)
        print(json.dumps(metrics["test"], indent=2))
        print(f"Results: {output}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
