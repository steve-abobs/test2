"""ML-сервис inference: загрузка артефактов и предсказание с margin."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..schemas import ComponentStatus, MLAnalysisResult, NormalizedCase
from . import artifacts
from .classifier import CLASSIFIER_NAME, align_probabilities, predict_one
from .embeddings import E5Embedder, EmbeddingUnavailableError
from .text import build_ml_text_for_case


class MLModelUnavailableError(RuntimeError):
    pass


class MLService:
    """Единая точка ML inference. Ничего не обучает на лету."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._classifier = None
        self._metadata = None
        self._load_error: str | None = None
        self._embedder: E5Embedder | None = None

    @property
    def ready(self) -> bool:
        return self._classifier is not None and self._metadata is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self.ready:
            return
        try:
            metadata = artifacts.load_metadata(self.settings.artifacts_dir)
            classifier = artifacts.load_classifier(self.settings.artifacts_dir)
        except FileNotFoundError:
            self._load_error = (
                "ML-артефакты не найдены. Сначала сгенерируйте датасет и обучите модель: "
                "см. README («Обучение модели»)."
            )
            raise MLModelUnavailableError(self._load_error) from None
        except Exception as exc:
            self._load_error = f"Не удалось загрузить ML-артефакты: {exc}"
            raise MLModelUnavailableError(self._load_error) from exc
        if metadata.embedding_model != self.settings.embedding_model:
            self._load_error = (
                "Модель в артефактах обучена на другом embedding-модели "
                f"({metadata.embedding_model}), чем указано в конфигурации "
                f"({self.settings.embedding_model}). Переобучите модель."
            )
            raise MLModelUnavailableError(self._load_error)
        self._classifier = classifier
        self._metadata = metadata

    @property
    def metadata(self):
        self._ensure_loaded()
        return self._metadata

    def _get_embedder(self) -> E5Embedder:
        if self._embedder is None:
            self._embedder = E5Embedder(
                self.settings.embedding_model,
                cache_path=artifacts.embedding_cache_path(self.settings.artifacts_dir),
                e5_prefix=self.settings.e5_prefix,
            )
        return self._embedder

    def analyze(self, case: NormalizedCase) -> MLAnalysisResult:
        """Возвращает ML-результат; ошибки не маскируются под успех."""
        try:
            self._ensure_loaded()
            text = build_ml_text_for_case(case, e5_prefix=self.settings.e5_prefix)
            vector = self._get_embedder().encode_one(text)
            prediction = predict_one(
                self._classifier,
                vector,
                low_margin_threshold=self.settings.ml_low_margin_threshold,
            )
            schema = prediction.to_schema(
                model_name=CLASSIFIER_NAME,
                embedding_model=self.settings.embedding_model,
            )
            return MLAnalysisResult(
                status=ComponentStatus.OK,
                prediction=schema,
                details={
                    "threshold": self.settings.ml_low_margin_threshold,
                    "e5_prefix": self.settings.e5_prefix,
                    "model_name": CLASSIFIER_NAME,
                    "embedding_model": self.settings.embedding_model,
                },
            )
        except (MLModelUnavailableError, EmbeddingUnavailableError) as exc:
            return MLAnalysisResult(
                status=ComponentStatus.UNAVAILABLE,
                error=str(exc),
                details={"threshold": self.settings.ml_low_margin_threshold},
            )
        except Exception as exc:
            return MLAnalysisResult(
                status=ComponentStatus.ERROR,
                error=f"Ошибка ML inference: {exc}",
                details={"threshold": self.settings.ml_low_margin_threshold},
            )

    def raw_probabilities(self, vectors) -> "object":
        """Только для отладки/тестов: вероятности в фиксированном порядке."""
        self._ensure_loaded()
        return align_probabilities(self._classifier, self._classifier.predict_proba(vectors))
