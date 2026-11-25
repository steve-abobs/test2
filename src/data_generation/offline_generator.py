"""Явный offline fallback: шаблонные кейсы без DeepSeek.

Используется только когда ключа нет и пользователь сознательно выбрал
``--engine offline`` или ``auto``. Метаданные фиксируют это, чтобы метрики
нельзя было выдать за результат генеративной модели.
"""
from __future__ import annotations

from ..schemas import SyntheticCase
from .generator import GenerationStats
from .offline_families import FAMILIES, build_offline_cases


def generate_offline_dataset(
    *,
    max_per_family: int = 10,
) -> tuple[list[SyntheticCase], GenerationStats]:
    cases = build_offline_cases(max_per_family=max_per_family)
    stats = GenerationStats(
        requested=len(cases),
        received=len(cases),
        accepted=len(cases),
        errors=0,
        batches=len(FAMILIES),
    )
    return cases, stats
