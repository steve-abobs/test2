"""Sentence Transformer embeddings для E5."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Модель embeddings недоступна (нет пакета, модели или сети)."""


class E5Embedder:
    """Обёртка intfloat/multilingual-e5-small с ленивой загрузкой и кэшем."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_path: Path,
        e5_prefix: str = "query: ",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.e5_prefix = e5_prefix
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_path, model_name=model_name)
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise EmbeddingUnavailableError(
                "sentence-transformers не установлен. Установите зависимости из requirements.txt."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # модель может отсутствовать локально / нет сети
            raise EmbeddingUnavailableError(
                f"Не удалось загрузить embedding-модель {self.model_name}: {exc}"
            ) from exc
        return self._model

    def encode(self, texts: Sequence[str], *, use_cache: bool = True) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        result: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        for index, text in enumerate(texts):
            cached = self.cache.get(text) if use_cache else None
            if cached is not None:
                result[index] = cached
            else:
                missing_texts.append(text)
                missing_positions.append(index)

        if missing_texts:
            model = self._load_model()
            try:
                encoded = model.encode(
                    missing_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as exc:  # pragma: no cover - зависит от окружения
                raise EmbeddingUnavailableError(f"Ошибка кодирования embeddings: {exc}") from exc
            encoded = np.asarray(encoded, dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(missing_texts):
                raise EmbeddingUnavailableError("Sentence Transformer вернул неожиданную форму массива.")
            for position, text, vector in zip(missing_positions, missing_texts, encoded):
                result[position] = vector
                if use_cache:
                    self.cache.put(text, vector)
            if use_cache:
                self.cache.save()

        if any(vector is None for vector in result):
            raise EmbeddingUnavailableError("Не удалось получить все embeddings.")
        return np.stack([vector for vector in result if vector is not None]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
