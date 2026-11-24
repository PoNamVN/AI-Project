"""Classification metrics shared by TensorFlow and frozen-feature experiments."""

import numpy as np


def evaluate(actual, predicted, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    true_positive = matrix.diagonal()
    total = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * true_positive, total, out=np.zeros(classes), where=total != 0)
    return {
        "correct": int(true_positive.sum()),
        "images": len(actual),
        "accuracy": float(true_positive.sum() / len(actual)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }
