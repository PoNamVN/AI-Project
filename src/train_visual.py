"""Frozen DINOv2/SigLIP2 comparison, selected only on validation; private local artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from src.data_audit import MODELS
from src.dataset_manifest import load_manifest
from src.vehicle_reasoning import (
    BRAND_PROMPTS,
    PROBES,
    brand_margin,
    fit_ridge,
    ridge_scores,
    summary,
)
from src.vision_features import (
    DINO,
    SIGLIP,
    MODES,
    ROOT,
    VisionFeatures,
    normalized,
    open_image,
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def feature_record(engine, row, cache):
    source = (ROOT / "data/raw" / row["source_path"]).resolve()
    if (
        not source.is_relative_to((ROOT / "data/raw").resolve())
        or digest(source) != row["sha256"]
    ):
        raise ValueError("Source path or checksum mismatch")
    saved = cache / (row["sha256"] + ".npz")
    if saved.is_file():
        with np.load(saved, allow_pickle=False) as archive:
            features = {key: archive[key].copy() for key in MODES}
            detections = json.loads(str(archive["detections"]))
        return features, detections
    image = open_image(source)
    detections = engine.detect(image)
    crop = image.crop(detections[0]["box"]) if detections else image
    features = engine.modes(image, crop)
    np.savez_compressed(saved, **features, detections=json.dumps(detections))
    return features, detections


def train(output, download=False):
    if output.exists() or not output.resolve().is_relative_to(
        (ROOT / "data/processed").resolve()
    ):
        raise ValueError("Choose a new output directory under data/processed")
    output.mkdir(parents=True)
    cache = ROOT / "data/processed/vision-v3-feature-cache"
    cache.mkdir(exist_ok=True)
    engine = VisionFeatures(download=download)
    dataset = ROOT / "data/processed/tesla-expanded-v2"
    _, validated_labels = load_manifest(dataset.resolve())
    manifest = read_csv(dataset / "splits.csv")
    labels = sorted(MODELS)
    if validated_labels != labels:
        raise ValueError("Frozen manifest must cover exactly the five Tesla classes")
    train_rows = [r for r in manifest if r["split"] == "train"]
    val_rows = [r for r in manifest if r["split"] == "validation"]
    assert len(train_rows) == 105 and len(val_rows) == 29
    features = {}
    detections = {}

    def extract(rows):
        for index, row in enumerate(rows, 1):
            key = row["sha256"]
            if key not in features:
                features[key], detections[key] = feature_record(engine, row, cache)
            if index % 10 == 0 or index == len(rows):
                print("Features", index, "/", len(rows), flush=True)

    extract(train_rows + val_rows)

    def matrix(rows, mode):
        return np.stack([features[r["sha256"]][mode] for r in rows])

    y = np.asarray([labels.index(r["class_id"]) for r in train_rows])
    yv = np.asarray([labels.index(r["class_id"]) for r in val_rows])
    brand_texts = [text for prompts in BRAND_PROMPTS.values() for text in prompts]
    brand_groups = [group for group, prompts in BRAND_PROMPTS.items() for _ in prompts]
    brand_embeddings = engine.texts(brand_texts)
    class_prompts = [
        f"a photo of a Tesla {MODELS[label]} vehicle"
        if label == "cybertruck"
        else f"a photo of a Tesla {MODELS[label]} car"
        for label in labels
    ]
    class_embeddings = engine.texts(class_prompts)
    probe_texts = [text for items in PROBES.values() for _, text in items]
    probe_embeddings = engine.texts(probe_texts)
    candidates = []
    weights = {}
    for mode in MODES:
        for regularization in (0.01, 0.1, 1.0):
            name = f"{mode}-ridge-{regularization}"
            w = fit_ridge(matrix(train_rows, mode), y, len(labels), regularization)
            metrics = summary(yv, ridge_scores(matrix(val_rows, mode), w))
            weights[name] = w
            candidates.append(
                {
                    "name": name,
                    "mode": mode,
                    "regularization": regularization,
                    "validation": metrics,
                }
            )
    zero = summary(yv, (matrix(val_rows, "siglip_crop") @ class_embeddings.T) * 5)
    candidates.append(
        {
            "name": "siglip_zeroshot",
            "mode": "siglip_crop",
            "regularization": None,
            "validation": zero,
        }
    )
    selected = max(
        candidates,
        key=lambda item: (
            item["validation"]["macro_f1"],
            item["validation"]["accuracy"],
            -item["validation"]["loss"],
        ),
    )
    negatives = [
        r
        for r in read_csv(ROOT / "data/processed/non-tesla-gate-review-v1/review.csv")
        if r["decision"] == "keep"
    ]
    assert len(negatives) >= 10 and all(
        r["review_status"] == "exterior_candidate_checked" for r in negatives
    )
    neg_val = [r for r in negatives if r["split"] == "validation"]
    assert len(neg_val) >= 5
    extract(neg_val)
    margins = np.asarray(
        [
            brand_margin(f, brand_embeddings, brand_groups)[0]
            for f in matrix(val_rows, "siglip_crop")
        ]
    )
    neg_margins = np.asarray(
        [
            brand_margin(f, brand_embeddings, brand_groups)[0]
            for f in matrix(neg_val, "siglip_crop")
        ]
    )
    # Preserve >=90% of known Tesla validation images before maximizing rejection of the small negative batch.
    thresholds = sorted(set([0.0, *margins.tolist(), *neg_margins.tolist()]))
    eligible = [t for t in thresholds if np.mean(margins >= t) >= 0.9]
    threshold = max(
        eligible,
        key=lambda t: (np.mean(neg_margins < t), np.mean(margins >= t), -abs(t)),
    )
    metadata = {
        "schema_version": 1,
        "class_ids": labels,
        "feature_mode": selected["mode"],
        "classifier": "zeroshot" if selected["name"] == "siglip_zeroshot" else "ridge",
        "selected_candidate": selected["name"],
        "selection_rule": "highest macro-F1, then accuracy, then lowest loss on the same 29 validation images",
        "validation_candidates": candidates,
        "brand_threshold": float(threshold),
        "brand_validation": {
            "tesla_retained": int((margins >= threshold).sum()),
            "tesla_images": 29,
            "other_rejected": int((neg_margins < threshold).sum()),
            "other_images": len(neg_val),
        },
        "class_gap_threshold": 0.08,
        "dino": DINO,
        "siglip": SIGLIP,
        "manifest_sha256": digest(dataset / "splits.csv"),
        "train_records": train_rows,
        "brand_groups": brand_groups,
        "brand_prompts": brand_texts,
        "limitations": [
            "Small, previously inspected Tesla development checks; no fresh independent Tesla test.",
            "Brand gate calibrated on 10 provisional non-Tesla images; not open-world brand recognition.",
            "Text probes are similarity hypotheses, not verified part detectors or causal explanations.",
            "Model class scores are not calibrated probabilities.",
        ],
    }
    w = weights.get(selected["name"], np.zeros((1, 5), dtype=np.float32))
    provenance = {}
    for row in metadata["train_records"]:
        dataset_name = Path(row["source_path"]).parts[0]
        if dataset_name not in provenance:
            provenance[dataset_name] = {
                r["relative_path"]: r
                for r in read_csv(ROOT / "data/raw" / dataset_name / "provenance.csv")
            }
        record = provenance[dataset_name][
            Path(*Path(row["source_path"]).parts[1:]).as_posix()
        ]
        row.update(
            artist_html=record.get("artist_html", ""),
            license_url=record.get("license_url", ""),
        )
    np.savez_compressed(
        output / "classifier.npz",
        weights=w,
        train_features=matrix(train_rows, selected["mode"]),
        brand_text=brand_embeddings,
        class_text=class_embeddings,
        probe_text=probe_embeddings,
    )
    # Freeze every model/threshold choice before extracting check-set features.
    (output / "model.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SELECTED", selected["name"], json.dumps(selected["validation"]), flush=True)
    tests = [r for r in manifest if r["split"] == "test"]
    external = [
        {
            **r,
            "source_path": "tesla-week4-new-review/" + r["relative_path"],
            "class_id": r["relative_path"].split("/")[0],
        }
        for r in read_csv(ROOT / "data/processed/week4-newcheck-v1/review.csv")
        if r["decision"] == "keep"
    ]
    neg_test = [r for r in negatives if r["split"] == "test"]
    assert len(tests) == 14 and len(external) == 16 and len(neg_test) >= 5
    training_hashes = {r["sha256"] for r in train_rows + val_rows}
    assert not training_hashes & {r["sha256"] for r in tests + external}
    if {r["sha256"] for r in neg_val} & {r["sha256"] for r in neg_test}:
        raise ValueError("Non-Tesla images appear in both validation and check sets")
    evaluation = {}
    for name, rows in [("known_check_14", tests), ("separate_check_16", external)]:
        extract(rows)
        actual = np.asarray([labels.index(r["class_id"]) for r in rows])
        x = matrix(rows, selected["mode"])
        scores = (
            x @ class_embeddings.T * 5
            if metadata["classifier"] == "zeroshot"
            else ridge_scores(x, w)
        )
        result = summary(actual, scores)
        accepted = np.asarray(
            [
                brand_margin(f, brand_embeddings, brand_groups)[0] >= threshold
                and bool(detections[r["sha256"]])
                and not (
                    len(detections[r["sha256"]]) > 1
                    and detections[r["sha256"]][1]["area"]
                    >= 0.5 * detections[r["sha256"]][0]["area"]
                )
                for f, r in zip(matrix(rows, "siglip_crop"), rows)
            ]
        )
        result["accepted"] = int(accepted.sum())
        result["end_to_end_top1_correct"] = int(
            ((scores.argmax(1) == actual) & accepted).sum()
        )
        result["end_to_end_accuracy"] = result["end_to_end_top1_correct"] / len(rows)
        result["end_to_end_top3_correct"] = int(
            (
                np.any(np.argsort(scores, axis=1)[:, -3:] == actual[:, None], axis=1)
                & accepted
            ).sum()
        )
        sorted_scores = np.sort(scores, axis=1)
        confident = accepted & (
            (sorted_scores[:, -1] - sorted_scores[:, -2])
            >= metadata["class_gap_threshold"]
        )
        result["class_confident_images"] = int(confident.sum())
        result["class_confident_correct"] = int(
            ((scores.argmax(1) == actual) & confident).sum()
        )
        with (output / (name + "_predictions.csv")).open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_path", "actual", "top1", "accepted", "detections"])
            writer.writerows(
                (
                    r["source_path"],
                    r["class_id"],
                    labels[p],
                    bool(ok),
                    len(detections[r["sha256"]]),
                )
                for r, p, ok in zip(rows, scores.argmax(1), accepted)
            )
        evaluation[name] = result
        print("CHECK", name, json.dumps(result), flush=True)
    extract(neg_test)
    accepted = [
        brand_margin(f, brand_embeddings, brand_groups)[0] >= threshold
        and bool(detections[r["sha256"]])
        for f, r in zip(matrix(neg_test, "siglip_crop"), neg_test)
    ]
    evaluation["non_tesla_check"] = {
        "images": len(neg_test),
        "false_tesla_accepts": int(sum(accepted)),
        "rejected": len(neg_test) - int(sum(accepted)),
    }
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    print("NEGATIVE_CHECK", json.dumps(evaluation["non_tesla_check"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or not destination.is_relative_to(
        (ROOT / "data/processed").resolve()
    ):
        parser.error("Choose a new output folder under data/processed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".visual-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "run"
        train(staging, args.download)
        if destination.exists():
            raise ValueError("Output appeared during training; preserving it")
        staging.rename(destination)
