"""Compare saved runs that share the same Tesla split manifest."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def compare(runs, output):
    if len(runs) < 2:
        raise ValueError("Provide at least two runs")
    snapshots = []
    for run in runs:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        with (run / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        truth = {row["relative_path"]: row["actual"] for row in predictions}
        if (len(truth) != len(predictions) or not truth
                or len(predictions) != metrics["test"]["images"]):
            raise ValueError(f"Invalid test predictions: {run}")
        labels = metrics["class_ids"]
        matrix = [[0] * len(labels) for _ in labels]
        for row in predictions:
            matrix[labels.index(row["actual"])][labels.index(row["predicted"])] += 1
        if matrix != metrics["test"]["confusion_matrix"]:
            raise ValueError(f"Metrics do not match predictions: {run}")
        if snapshots:
            previous = snapshots[0][1]
            if (metrics["manifest_sha256"] != previous["manifest_sha256"]
                    or metrics["class_ids"] != previous["class_ids"] or truth != snapshots[0][3]):
                raise ValueError("Runs must share the same manifest, classes and test images")
        snapshots.append((run.name, metrics, predictions, truth))
    if len({name for name, *_ in snapshots}) != len(snapshots):
        raise ValueError("Run directory names must be unique")
    output.mkdir(parents=True)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "epochs", "selected_epoch", "validation_loss", "test_correct", "test_images", "accuracy", "macro_f1"))
        for name, m, *_ in snapshots:
            writer.writerow((name, m["epochs_run"], m["best_epoch"], min(m["history"]["val_loss"]),
                             m["test"]["correct"], m["test"]["images"], m["test"]["accuracy"], m["test"]["macro_f1"]))
    with (output / "errors.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "relative_path", "actual", "predicted"))
        for name, _, predictions, _ in snapshots:
            writer.writerows((name, row["relative_path"], row["actual"], row["predicted"])
                             for row in predictions if row["actual"] != row["predicted"])
    with (output / "per_class.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "class_id", "correct", "support", "precision", "recall", "f1"))
        for name, m, *_ in snapshots:
            matrix = m["test"]["confusion_matrix"]
            for i, label in enumerate(m["class_ids"]):
                support = sum(matrix[i])
                predicted = sum(row[i] for row in matrix)
                tp = matrix[i][i]
                writer.writerow((name, label, tp, support, tp / predicted if predicted else 0,
                                 tp / support if support else 0, 2 * tp / (support + predicted) if support + predicted else 0))
    selected = min(snapshots, key=lambda item: min(item[1]["history"]["val_loss"]))[0]
    (output / "selection.json").write_text(json.dumps({
        "selected_run": selected, "criterion": "lowest validation loss, not test accuracy",
        "manifest_sha256": snapshots[0][1]["manifest_sha256"],
        "limitations": ["14-image pilot test previously inspected in Week 3; independent external evaluation remains necessary.",
                        "Vehicle grouping and labels are estimated from images and source clues.",
                        "Single seed per experiment; differences do not establish statistical significance."],
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        processed = (Path(__file__).resolve().parents[1] / "data" / "processed").resolve()
        if output == processed or not output.is_relative_to(processed) or output.exists():
            raise ValueError("Choose a new report directory inside data/processed")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".compare-", dir=output.parent) as temporary:
            staging = Path(temporary) / "report"
            compare([path.resolve() for path in args.runs], staging)
            staging.rename(output)
        print(f"Comparison saved: {output}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
