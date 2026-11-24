"""Small numeric helpers; evidence is measured similarity, never invented vehicle parts."""

import numpy as np
from src.classification_metrics import evaluate

BRAND_PROMPTS = {
    "tesla": [
        "a photo of a Tesla electric car",
        "a photo of a Tesla Cybertruck pickup truck",
    ],
    "other_car": [
        f"a photo of a {brand} car"
        for brand in [
            "Toyota",
            "Honda",
            "BMW",
            "Mercedes-Benz",
            "Audi",
            "Ford",
            "Volkswagen",
            "Hyundai",
            "Kia",
            "BYD",
            "Volvo",
            "Rivian",
            "Lucid",
            "Porsche",
            "Nissan",
        ]
    ],
    "not_car": [
        "a photo of a motorcycle",
        "a photo of a bicycle",
        "a photo of an empty road",
        "a photo of a toy car",
        "a photo of a room",
        "a photo of an animal",
    ],
}
PROBES = {
    "Kiểu thân xe": [
        ("sedan", "a low sedan car with a separate rear trunk"),
        ("suv", "a tall crossover SUV with a rear hatch"),
        ("pickup", "an angular stainless steel pickup truck"),
    ],
    "Góc nhìn": [
        ("front", "the front of a car"),
        ("rear", "the rear of a car"),
        ("side", "the side profile of a car"),
    ],
    "Cửa xe": [
        ("falcon", "a Tesla Model X car with rear falcon wing doors open"),
        ("normal", "a car with conventional doors closed"),
    ],
}
PROBE_VI = {
    "sedan": "Dáng xe thấp kiểu sedan",
    "suv": "Dáng crossover/SUV và phần đuôi hatchback",
    "pickup": "Dáng bán tải với đường nét góc cạnh",
    "front": "Góc nhìn phía trước",
    "rear": "Góc nhìn phía sau",
    "side": "Góc nhìn bên hông",
    "falcon": "Cửa sau mở lên kiểu cánh chim",
    "normal": "Cửa thông thường hoặc đang đóng",
}


def fit_ridge(features, targets, classes, regularization):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=int)
    if (
        x.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or regularization <= 0
        or set(y) != set(range(classes))
    ):
        raise ValueError("Invalid training features or labels")
    x = np.column_stack((x, np.ones(len(x))))
    weights = np.sqrt(len(y) / (classes * np.bincount(y, minlength=classes)[y]))
    xw = x * weights[:, None]
    yw = np.eye(classes)[y] * weights[:, None]
    return (
        xw.T @ np.linalg.solve(xw @ xw.T + regularization * np.eye(len(y)), yw)
    ).astype(np.float32)


def ridge_scores(features, weights):
    features, weights = np.asarray(features), np.asarray(weights)
    if (
        features.ndim != 2
        or weights.ndim != 2
        or features.shape[1] + 1 != weights.shape[0]
        or not np.isfinite(features).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Classifier and feature dimensions do not match")
    return np.column_stack((features, np.ones(len(features)))) @ weights


def probabilities(scores):
    values = np.asarray(scores, dtype=np.float64) * 10
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid classification scores")
    values -= values.max(axis=1, keepdims=True)
    exponents = np.exp(values)
    return exponents / exponents.sum(axis=1, keepdims=True)


def summary(actual, scores):
    p = probabilities(scores)
    actual = np.asarray(actual, dtype=int)
    result = evaluate(actual, p.argmax(1), p.shape[1])
    result["top3_correct"] = int(
        np.any(np.argsort(p, axis=1)[:, -3:] == actual[:, None], axis=1).sum()
    )
    result["top3_accuracy"] = result["top3_correct"] / len(actual)
    result["loss"] = float(
        -np.log(np.maximum(p[np.arange(len(actual)), actual], 1e-9)).mean()
    )
    return result


def brand_margin(image_feature, text_features, groups):
    scores = np.asarray(text_features) @ np.asarray(image_feature)
    maxima = {
        group: float(np.max(scores[np.asarray(groups) == group]))
        for group in BRAND_PROMPTS
    }
    return maxima["tesla"] - max(maxima["other_car"], maxima["not_car"]), maxima
