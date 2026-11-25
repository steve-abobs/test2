"""Метрики качества ML, включая базовые проверки калибровки."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..schemas import Category, ModelMetrics

LABEL_ORDER: list[Category] = [
    Category.PRIORITY_1,
    Category.REQUIRES_INTERVENTION,
    Category.NO_INTERVENTION,
]


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray | None,
    *,
    labels: Sequence[str] | None = None,
    model_name: str,
    split: str,
) -> ModelMetrics:
    """Считает macro-F1, per-class метрики, confusion matrix и калибровку."""
    labels = list(labels or [category.value for category in LABEL_ORDER])
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))

    per_class_raw = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    precision, recall, f1, support = per_class_raw
    per_class: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }

    calibration: dict[str, float] = {}
    if y_proba is not None and len(y_proba) == len(y_true_arr):
        y_proba = np.asarray(y_proba, dtype=float)
        label_to_index = {label: index for index, label in enumerate(labels)}
        y_true_indices = np.array([label_to_index[value] for value in y_true_arr])
        one_hot = np.zeros_like(y_proba)
        one_hot[np.arange(len(y_true_indices)), y_true_indices] = 1.0
        brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))
        probabilities = np.clip(y_proba, 0.0, 1.0)
        predicted_indices = np.argmax(probabilities, axis=1)
        confidences = probabilities[np.arange(len(probabilities)), predicted_indices]
        correct = (predicted_indices == y_true_indices).astype(float)
        ece = 0.0
        n_bins = 10
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for lower, upper in zip(bins[:-1], bins[1:]):
            if upper == 1.0:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)
            if not np.any(mask):
                continue
            ece += float(np.mean(mask)) * abs(
                float(np.mean(confidences[mask])) - float(np.mean(correct[mask]))
            )
        calibration = {
            "brier_score_multiclass": brier,
            "ece_top_label_10_bins": float(ece),
            "mean_confidence": float(np.mean(confidences)),
            "accuracy_at_top_label": float(np.mean(correct)),
            "confidence_minus_accuracy": float(np.mean(confidences) - np.mean(correct)),
        }

    return ModelMetrics(
        model_name=model_name,
        split=split,  # type: ignore[arg-type]
        sample_size=int(len(y_true_arr)),
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        macro_f1=float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true_arr, y_pred_arr, labels=labels).astype(int).tolist(),
        labels=labels,
        calibration=calibration,
    )
