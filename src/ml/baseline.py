"""Baseline TF-IDF + LogisticRegression для сравнения с E5."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .classifier import build_classifier


@dataclass
class BaselineModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    C: float

    def as_pipeline(self) -> Pipeline:
        return Pipeline([("tfidf", self.vectorizer), ("clf", self.classifier)])

    def predict(self, texts: list[str]) -> list[str]:
        return [str(value) for value in self.classifier.predict(self.vectorizer.transform(texts))]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.vectorizer.transform(texts))


def build_baseline(*, C: float = 1.0) -> BaselineModel:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )
    classifier = build_classifier(C=C, class_weight="balanced")
    return BaselineModel(vectorizer=vectorizer, classifier=classifier, C=C)


def fit_baseline(texts: list[str], labels: list[str], *, C: float) -> BaselineModel:
    model = build_baseline(C=C)
    matrix = model.vectorizer.fit_transform(texts)
    model.classifier.fit(matrix, labels)
    return model


def align_baseline_probabilities(model: BaselineModel, texts: list[str]) -> np.ndarray:
    from ..schemas import Category
    from .classifier import align_probabilities

    probabilities = model.classifier.predict_proba(model.vectorizer.transform(texts))
    return align_probabilities(model.classifier, probabilities)
