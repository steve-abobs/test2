"""Общие фикстуры тестов (без сети и без скачивания моделей)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_settings
from src.schemas import ChatMessage, NormalizedCase, ProcessType


@pytest.fixture()
def settings(tmp_path: Path):
    base = load_settings()
    return replace(
        base,
        deepseek_api_key=None,
        data_dir=tmp_path / "data",
        generated_dir=tmp_path / "data" / "generated",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        reference_cases_path=PROJECT_ROOT / "data" / "reference_cases.json",
    )


@pytest.fixture()
def normalized_case() -> NormalizedCase:
    return NormalizedCase(
        case_id="TEST-1",
        process_type=ProcessType.DISMISSAL,
        crm_text="Последний рабочий день сегодня. Подтверждение руководителя не зафиксировано.",
        chat_messages=[
            ChatMessage(author="Иван", role="Сотрудник", text="Руководитель в курсе, мы обсудили."),
            ChatMessage(author="Анна", role="Руководитель", text="Подтверждение пока не оформляла."),
        ],
    )
