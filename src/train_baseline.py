"""Train a Tesla CNN or MobileNetV2 on a frozen split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import tensorflow as tf

from src.classification_metrics import evaluate
from src.dataset_manifest import load_manifest
from src.prepare_dataset import IMAGE_SIZE, SPLITS


def load_images(entries, labels):
    images, targets = [], []
    for _, path, label in entries:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != IMAGE_SIZE:
                raise ValueError(f"Expected a prepared RGB image of size {IMAGE_SIZE}: {path}")
            images.append(np.asarray(image))
        targets.append(labels.index(label))
    return np.stack(images), np.asarray(targets, dtype=np.int32)


def build_model(classes, architecture="cnn", augment=False, seed=42, image_size=IMAGE_SIZE):
    layers = tf.keras.layers
    if architecture == "mobilenetv2":
        inputs = layers.Input((*image_size, 3))
        x = inputs
        if augment:
            x = layers.RandomFlip("horizontal", seed=seed)(x)
            x = layers.RandomRotation(0.03, fill_mode="constant", seed=seed + 1)(x)
            x = layers.RandomZoom(0.1, fill_mode="constant", seed=seed + 2)(x)
        # MobileNetV2 expects pixels scaled from [0, 255] to [-1, 1].
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*image_size, 3), include_top=False, weights="imagenet",
        )
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs)
    if architecture != "cnn" or augment:
        raise ValueError("Use cnn without augmentation, or mobilenetv2 with optional augmentation")
    return tf.keras.Sequential([
        layers.Input((*IMAGE_SIZE, 3)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(classes, activation="softmax"),
    ])


def train(root, output, seed, epochs, architecture="cnn", augment=False):
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    rows, labels = load_manifest(root)
    x_train, y_train = load_images(rows["train"], labels)
    x_val, y_val = load_images(rows["validation"], labels)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = build_model(len(labels), architecture, augment, seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=8, callbacks=[stop], verbose=2).history

    val_predicted = model.predict(x_val, batch_size=8, verbose=0).argmax(axis=1)
    x_test, y_test = load_images(rows["test"], labels)
    predicted = model.predict(x_test, batch_size=8, verbose=0).argmax(axis=1)
    majority = int(np.bincount(y_train, minlength=len(labels)).argmax())
    metrics = {
        "tensorflow": tf.__version__, "seed": seed, "epochs_limit": epochs,
        "epochs_run": len(history["loss"]), "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "selection": "lowest validation loss", "batch_size": 8, "parameters": model.count_params(),
        "architecture": architecture, "augmentation": augment,
        "learning_rate": 0.001, "loss": "sparse_categorical_crossentropy",
        "pretrained_weights": "imagenet" if architecture == "mobilenetv2" else None,
        "backbone_frozen": architecture == "mobilenetv2",
        "augmentation_config": {"horizontal_flip": True, "rotation": 0.03, "zoom": 0.1} if augment else {},
        "validation": evaluate(y_val, val_predicted, len(labels)),
        "trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_weights),
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
    parser.add_argument("--architecture", choices=("cnn", "mobilenetv2"), default="cnn")
    parser.add_argument("--augment", action="store_true")
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
            metrics = train(root, staging, args.seed, args.epochs, args.architecture, args.augment)
            staging.rename(output)
        print(json.dumps(metrics["test"], indent=2))
        print(f"Results: {output}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
