"""Fit a supervised Tesla/other head without changing the Tesla model classifier."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from src.train_visual import digest, feature_record, read_csv
from src.dataset_manifest import load_manifest
from src.vehicle_reasoning import fit_ridge, ridge_scores, summary
from src.vision_features import ROOT, VisionFeatures


def train(source, output):
    if (
        not source.is_relative_to((ROOT / "data/processed").resolve())
        or output.exists()
        or output.is_relative_to(source)
        or not output.is_relative_to((ROOT / "data/processed").resolve())
    ):
        raise ValueError("Use a source and new output under data/processed")
    metadata = json.loads((source / "model.json").read_text(encoding="utf-8"))
    dataset = ROOT / "data/processed/tesla-expanded-v2"
    if digest(dataset / "splits.csv") != metadata["manifest_sha256"]:
        raise ValueError(
            "Positive dataset changed since the parent classifier was trained"
        )
    load_manifest(dataset.resolve())
    with np.load(source / "classifier.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    mode = metadata["feature_mode"]
    engine = VisionFeatures()
    cache = ROOT / "data/processed/vision-v3-feature-cache"
    positives = read_csv(dataset / "splits.csv")
    negatives = [
        r
        for r in read_csv(
            ROOT / "data/processed/non-tesla-brand-training-v3/review.csv"
        )
        if r["decision"] == "keep"
    ]
    negative_val = [
        r
        for r in read_csv(ROOT / "data/processed/non-tesla-gate-review-v1/review.csv")
        if r["decision"] == "keep" and r["split"] == "validation"
    ]
    negative_checks = [
        r
        for r in read_csv(ROOT / "data/processed/non-tesla-gate-review-v1/review.csv")
        if r["decision"] == "keep" and r["split"] == "test"
    ]
    external = read_csv(ROOT / "data/processed/week4-newcheck-v1/review.csv")
    if (
        len(negatives) < 20
        or len({r["sha256"] for r in negatives}) != len(negatives)
        or any(
            r["split"] != "train" or r["review_status"] != "exterior_candidate_checked"
            for r in negatives
        )
        or {r["sha256"] for r in negatives}
        & {r["sha256"] for r in negative_val + negative_checks + positives + external}
    ):
        raise ValueError(
            "Need at least 20 separate negative training images without gate-set duplicates"
        )
    train_rows = [r for r in positives if r["split"] == "train"] + negatives
    val_rows = [r for r in positives if r["split"] == "validation"] + negative_val

    def extract(rows):
        images = []
        for index, row in enumerate(rows, 1):
            features, _ = feature_record(engine, row, cache)
            images.append(features[mode])
            if index % 20 == 0 or index == len(rows):
                print("Brand features", index, "/", len(rows), flush=True)
        return np.stack(images)

    x = extract(train_rows)
    v = extract(val_rows)
    y = np.asarray([1] * (len(train_rows) - len(negatives)) + [0] * len(negatives))
    truth = np.asarray(
        [1] * (len(val_rows) - len(negative_val)) + [0] * len(negative_val)
    )
    candidates = []
    weights = {}
    for regularization in (0.01, 0.1, 1.0):
        w = fit_ridge(x, y, 2, regularization)
        scores = ridge_scores(v, w)
        metrics = summary(truth, scores)
        metrics["tesla_retained"] = int(np.sum((scores.argmax(1) == 1) & (truth == 1)))
        candidates.append({"regularization": regularization, "validation": metrics})
        weights[regularization] = w
    eligible = [
        c
        for c in candidates
        if c["validation"]["tesla_retained"] >= 0.9 * int((truth == 1).sum())
    ]
    if not eligible:
        raise ValueError(
            "No brand head retains 90% Tesla validation; preserving prior model"
        )
    selected = max(
        eligible,
        key=lambda c: (
            c["validation"]["macro_f1"],
            c["validation"]["accuracy"],
            -c["validation"]["loss"],
        ),
    )
    arrays["brand_weights"] = weights[selected["regularization"]]
    metadata.update(
        brand_method="ridge",
        brand_threshold=0.0,
        brand_training_images={"tesla": int((y == 1).sum()), "other": len(negatives)},
        brand_training_manifest_sha256=digest(
            ROOT / "data/processed/non-tesla-brand-training-v3/review.csv"
        ),
        brand_head_selection={
            "rule": "macro-F1 on 29 Tesla + 9 other validation, fixed zero margin, retain >=90% Tesla; no check-set tuning",
            "candidates": candidates,
            "selected": selected,
        },
    )
    metadata["legacy_text_gate_validation"] = metadata.pop("brand_validation", None)
    metadata["brand_validation"] = {
        "tesla_images": int((truth == 1).sum()),
        "tesla_retained": selected["validation"]["tesla_retained"],
        "other_images": len(negative_val),
        "other_rejected": selected["validation"]["confusion_matrix"][0][0],
    }
    metadata["limitations"] = [
        s for s in metadata["limitations"] if not s.startswith("Brand gate calibrated")
    ]
    metadata["limitations"].append(
        "Binary brand head has few other-car examples; check sets already inspected during development, not a fresh benchmark."
    )
    metadata["parent_model_manifest_sha256"] = digest(source / "model.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".brand-") as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        np.savez_compressed(staging / "classifier.npz", **arrays)
        (staging / "model.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if output.exists():
            raise ValueError("Output already exists")
        staging.rename(output)
    print("BRAND_HEAD_SELECTED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train(args.source.resolve(), args.output.resolve())
