"""Кэш embeddings Sentence Transformer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import ensure_dir, stable_hash


class EmbeddingCache:
    """Простой дисковый кэш: hash(текст) -> вектор.

    Формат npz: keys (str), vectors (float32). Ключ включает имя модели и
    E5-префикс, поэтому смена модели не переиспользует несовместимые векторы.
    """

    def __init__(self, path: Path, *, model_name: str):
        self.path = Path(path)
        self.model_name = model_name
        self._vectors: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = data["keys"].tolist()
                vectors = data["vectors"]
                if len(keys) != len(vectors):
                    return
                self._vectors = {
                    str(key): np.asarray(vector, dtype=np.float32)
                    for key, vector in zip(keys, vectors)
                }
        except (KeyError, ValueError, OSError):
            # Повреждённый кэш не должен ломать inference: просто пересоберём.
            self._vectors = {}

    def key_for(self, text: str) -> str:
        return stable_hash(text, prefix=f"embedding|{self.model_name}")

    def get(self, text: str) -> np.ndarray | None:
        return self._vectors.get(self.key_for(text))

    def put(self, text: str, vector: np.ndarray) -> None:
        self._vectors[self.key_for(text)] = np.asarray(vector, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        ensure_dir(self.path.parent)
        keys = list(self._vectors.keys())
        if not keys:
            return
        vectors = np.stack([self._vectors[key] for key in keys]).astype(np.float32)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp.npz")
        np.savez_compressed(tmp, keys=np.array(keys), vectors=vectors)
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._vectors)
