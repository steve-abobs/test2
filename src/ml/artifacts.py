"""Пути и чтение/запись ML-артефактов."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ..schemas import ModelBundleMetadata
from ..utils import ensure_dir, read_json, write_json

MODEL_DIRNAME = "ml"
CLASSIFIER_FILENAME = "classifier.joblib"
BASELINE_FILENAME = "baseline.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"
EMBEDDING_CACHE_FILENAME = "embeddings_cache.npz"


def model_dir(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / MODEL_DIRNAME


def classifier_path(artifacts_dir: Path) -> Path:
    return model_dir(artifacts_dir) / CLASSIFIER_FILENAME


def baseline_path(artifacts_dir: Path) -> Path:
    return model_dir(artifacts_dir) / BASELINE_FILENAME


def metadata_path(artifacts_dir: Path) -> Path:
    return model_dir(artifacts_dir) / METADATA_FILENAME


def metrics_path(artifacts_dir: Path) -> Path:
    return model_dir(artifacts_dir) / METRICS_FILENAME


def embedding_cache_path(artifacts_dir: Path) -> Path:
    return model_dir(artifacts_dir) / EMBEDDING_CACHE_FILENAME


def save_model_bundle(
    artifacts_dir: Path,
    *,
    classifier: Any,
    baseline: Any,
    metadata: ModelBundleMetadata,
    metrics_payload: dict[str, Any],
) -> None:
    directory = ensure_dir(model_dir(artifacts_dir))
    joblib.dump(classifier, directory / CLASSIFIER_FILENAME)
    joblib.dump(baseline, directory / BASELINE_FILENAME)
    write_json(directory / METADATA_FILENAME, metadata.model_dump(mode="json"))
    write_json(directory / METRICS_FILENAME, metrics_payload)


def load_classifier(artifacts_dir: Path) -> Any:
    return joblib.load(classifier_path(artifacts_dir))


def load_metadata(artifacts_dir: Path) -> ModelBundleMetadata:
    return ModelBundleMetadata.model_validate(read_json(metadata_path(artifacts_dir)))


def load_metrics(artifacts_dir: Path) -> dict[str, Any]:
    return read_json(metrics_path(artifacts_dir))
