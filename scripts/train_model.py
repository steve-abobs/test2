#!/usr/bin/env python
"""Обучение E5 + LogisticRegression и TF-IDF baseline на готовом split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import load_settings
from src.ml import artifacts
from src.ml.baseline import fit_baseline
from src.ml.classifier import CLASSIFIER_NAME, align_probabilities, build_classifier
from src.ml.embeddings import E5Embedder, EmbeddingUnavailableError
from src.ml.metrics import LABEL_ORDER, compute_metrics
from src.ml.text import DEFAULT_E5_PREFIX, build_ml_text
from src.schemas import DatasetMetadata, ModelBundleMetadata, ModelMetrics, SplitBundle, SyntheticCase
from src.utils import read_json, read_jsonl


def load_dataset(generated_dir: Path) -> tuple[list[SyntheticCase], SplitBundle, DatasetMetadata]:
    cases = [SyntheticCase.model_validate(row) for row in read_jsonl(generated_dir / "dataset.jsonl")]
    split = SplitBundle.model_validate(read_json(generated_dir / "split.json"))
    metadata = DatasetMetadata.model_validate(read_json(generated_dir / "dataset_metadata.json"))
    return cases, split, metadata


def select_by_index(cases: list[SyntheticCase], ids: list[str]) -> list[SyntheticCase]:
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"В split.json есть ID, отсутствующие в датасете: {missing[:5]}")
    return [by_id[case_id] for case_id in ids]


def ml_texts(cases: list[SyntheticCase], *, prefix: str) -> list[str]:
    return [
        build_ml_text(
            process_type=case.process_type,
            crm_text=case.crm_text,
            chat_messages=case.chat_messages,
            e5_prefix=prefix,
        )
        for case in cases
    ]


def metrics_for(
    *,
    y_true: list[str],
    y_pred: list[str],
    y_proba: np.ndarray | None,
    model_name: str,
    split: str,
) -> ModelMetrics:
    return compute_metrics(
        y_true,
        y_pred,
        y_proba,
        labels=[category.value for category in LABEL_ORDER],
        model_name=model_name,
        split=split,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-grid", default="0.1,0.5,1.0,2.0,5.0")
    parser.add_argument("--baseline-c-grid", default="0.5,1.0,2.0")
    args = parser.parse_args()

    settings = load_settings()
    cases, split, dataset_meta = load_dataset(settings.generated_dir)
    train = select_by_index(cases, split.train)
    validation = select_by_index(cases, split.validation)
    test = select_by_index(cases, split.test)

    train_texts = ml_texts(train, prefix=settings.e5_prefix)
    validation_texts = ml_texts(validation, prefix=settings.e5_prefix)
    test_texts = ml_texts(test, prefix=settings.e5_prefix)
    train_labels = [case.label.value for case in train]
    validation_labels = [case.label.value for case in validation]
    test_labels = [case.label.value for case in test]

    embedder = E5Embedder(
        settings.embedding_model,
        cache_path=artifacts.embedding_cache_path(settings.artifacts_dir),
        e5_prefix=settings.e5_prefix,
    )
    try:
        train_vectors = embedder.encode(train_texts)
        validation_vectors = embedder.encode(validation_texts)
        test_vectors = embedder.encode(test_texts)
    except EmbeddingUnavailableError as exc:
        print(f"ОШИБКА embeddings: {exc}", file=sys.stderr)
        return 2

    from sklearn.metrics import f1_score

    grid = [float(value) for value in args.c_grid.split(",") if value.strip()]
    selection: dict[float, float] = {}
    best_c = grid[0]
    best_score = -1.0
    for c_value in grid:
        candidate = build_classifier(C=c_value, class_weight="balanced")
        candidate.fit(train_vectors, train_labels)
        score = float(
            f1_score(
                validation_labels,
                candidate.predict(validation_vectors),
                average="macro",
                zero_division=0,
            )
        )
        selection[c_value] = score
        if score > best_score + 1e-9:
            best_score, best_c = score, c_value

    classifier = build_classifier(C=best_c, class_weight="balanced")
    classifier.fit(train_vectors, train_labels)

    validation_predictions = [str(value) for value in classifier.predict(validation_vectors)]
    validation_proba = align_probabilities(classifier, classifier.predict_proba(validation_vectors))
    test_predictions = [str(value) for value in classifier.predict(test_vectors)]
    test_proba = align_probabilities(classifier, classifier.predict_proba(test_vectors))
    validation_metrics = metrics_for(
        y_true=validation_labels,
        y_pred=validation_predictions,
        y_proba=validation_proba,
        model_name=f"{CLASSIFIER_NAME}_c{best_c:g}",
        split="validation",
    )
    test_metrics = metrics_for(
        y_true=test_labels,
        y_pred=test_predictions,
        y_proba=test_proba,
        model_name=f"{CLASSIFIER_NAME}_c{best_c:g}",
        split="test",
    )

    baseline_texts = {
        "train": ml_texts(train, prefix=""),
        "validation": ml_texts(validation, prefix=""),
        "test": ml_texts(test, prefix=""),
    }
    baseline_grid = [float(value) for value in args.baseline_c_grid.split(",") if value.strip()]
    baseline_selection: dict[float, float] = {}
    baseline_best_c = baseline_grid[0]
    baseline_best_score = -1.0
    for c_value in baseline_grid:
        candidate = fit_baseline(baseline_texts["train"], train_labels, C=c_value)
        score = float(
            f1_score(
                validation_labels,
                candidate.predict(baseline_texts["validation"]),
                average="macro",
                zero_division=0,
            )
        )
        baseline_selection[c_value] = score
        if score > baseline_best_score + 1e-9:
            baseline_best_score, baseline_best_c = score, c_value

    baseline = fit_baseline(baseline_texts["train"], train_labels, C=baseline_best_c)
    baseline_validation_metrics = metrics_for(
        y_true=validation_labels,
        y_pred=baseline.predict(baseline_texts["validation"]),
        y_proba=baseline.predict_proba(baseline_texts["validation"]),
        model_name=f"tfidf_logreg_c{baseline_best_c:g}",
        split="validation",
    )
    baseline_test_metrics = metrics_for(
        y_true=test_labels,
        y_pred=baseline.predict(baseline_texts["test"]),
        y_proba=baseline.predict_proba(baseline_texts["test"]),
        model_name=f"tfidf_logreg_c{baseline_best_c:g}",
        split="test",
    )

    warnings: list[str] = []
    if dataset_meta.engine != "deepseek":
        warnings.append(
            "Датасет собран offline-шаблонами (не DeepSeek). Метрики проверяют только "
            "корректность пайплайна и не свидетельствуют о качестве на реальных данных."
        )
    warnings.append(
        "Синтетические метрики не доказывают качество на реальных кейсах; "
        "контрольные кейсы C-17/C-24/P-08/P-09 не входили в обучение."
    )

    bundle_metadata = ModelBundleMetadata(
        embedding_model=settings.embedding_model,
        e5_prefix=settings.e5_prefix,
        classifier=CLASSIFIER_NAME,
        class_weight="balanced",
        selected_c=best_c,
        classes=[category.value for category in LABEL_ORDER],
        low_margin_threshold=settings.ml_low_margin_threshold,
        dataset_engine=dataset_meta.engine,
        dataset_size=len(cases),
        split_sizes={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        baseline_validation_metrics=baseline_validation_metrics,
        baseline_test_metrics=baseline_test_metrics,
        warnings=warnings,
    )
    metrics_payload = {
        "validation": validation_metrics.model_dump(mode="json"),
        "test": test_metrics.model_dump(mode="json"),
        "baseline_validation": baseline_validation_metrics.model_dump(mode="json"),
        "baseline_test": baseline_test_metrics.model_dump(mode="json"),
        "selection": {"c_grid": selection, "selected_c": best_c},
        "baseline_selection": {
            "c_grid": baseline_selection,
            "selected_c": baseline_best_c,
        },
        "dataset": dataset_meta.model_dump(mode="json"),
        "warnings": warnings,
    }
    artifacts.save_model_bundle(
        settings.artifacts_dir,
        classifier=classifier,
        baseline=baseline,
        metadata=bundle_metadata,
        metrics_payload=metrics_payload,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_c": best_c,
                "validation_macro_f1": validation_metrics.macro_f1,
                "test_macro_f1": test_metrics.macro_f1,
                "baseline_test_macro_f1": baseline_test_metrics.macro_f1,
                "dataset_engine": dataset_meta.engine,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


