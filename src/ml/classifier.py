"""LogisticRegression поверх E5 embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..schemas import Category, MLPrediction

CLASSIFIER_NAME = "logreg_e5"


def build_classifier(*, C: float = 1.0, class_weight: str = "balanced") -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )


@dataclass(frozen=True)
class Prediction:
    category: Category
    probabilities: dict[Category, float]
    margin: float
    low_margin: bool

    def to_schema(self, *, model_name: str, embedding_model: str) -> MLPrediction:
        return MLPrediction(
            category=self.category,
            probabilities=self.probabilities,
            margin=self.margin,
            low_margin=self.low_margin,
            model_name=model_name,
            embedding_model=embedding_model,
        )


def align_probabilities(model: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    """Возвращает матрицу вероятностей в фиксированном порядке Category."""
    labels = [Category(category) for category in model.classes_]
    if len(labels) != len(Category):
        raise ValueError(
            "Классификатор обучен не на всех трёх категориях: "
            f"{[label.value for label in labels]}"
        )
    order = [
        Category.PRIORITY_1,
        Category.REQUIRES_INTERVENTION,
        Category.NO_INTERVENTION,
    ]
    index_by_label = {label: index for index, label in enumerate(labels)}
    return np.stack([proba[:, index_by_label[label]] for label in order], axis=1)


def predict_one(
    model: LogisticRegression,
    vector: np.ndarray,
    *,
    low_margin_threshold: float,
) -> Prediction:
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    labels = [Category(category) for category in model.classes_]
    prob_by_label = {label: float(prob) for label, prob in zip(labels, proba)}
    for category in Category:
        prob_by_label.setdefault(category, 0.0)
    ranked = sorted(prob_by_label.items(), key=lambda item: item[1], reverse=True)
    margin = float(ranked[0][1] - ranked[1][1])
    total = sum(prob_by_label.values())
    if abs(total - 1.0) > 1e-6:
        prob_by_label = {label: value / total for label, value in prob_by_label.items()}
    return Prediction(
        category=ranked[0][0],
        probabilities=prob_by_label,
        margin=margin,
        low_margin=margin < low_margin_threshold,
    )


def evaluate_predictions(model: LogisticRegression, X: np.ndarray) -> tuple[list[str], np.ndarray]:
    predictions = model.predict(X)
    probabilities = align_probabilities(model, model.predict_proba(X))
    return [str(value) for value in predictions], probabilities
