"""Промпт LLM-аудитора не должен содержать ML-информацию."""
from __future__ import annotations

import pytest

from src.llm.audit import build_audit_prompts, parse_audit
from src.llm.client import LLMResponseError
from src.schemas import NormalizedCase


def test_audit_prompt_contains_only_case_sources(normalized_case: NormalizedCase) -> None:
    system_prompt, user_prompt = build_audit_prompts(normalized_case)

    assert "независимый аудитор" in system_prompt
    assert normalized_case.crm_text in user_prompt
    assert "1. Иван (Сотрудник):" in user_prompt
    assert "2. Анна (Руководитель):" in user_prompt
    for forbidden in ("вероятн", "probability", "margin", "logreg", "ML-", "label_rationale"):
        assert forbidden not in user_prompt


def test_parse_audit_accepts_string_contradictions() -> None:
    audit = parse_audit(
        {
            "category": "requires_intervention",
            "rationale": "CRM и чат расходятся по подтверждению.",
            "known_facts": [],
            "contradictions": ["Руководитель не подтвердил в CRM"],
            "missing_information": "подтверждение",
            "checks": "Проверить карточку",
            "recommended_actions": "Запросить подтверждение",
            "start_with": "Открыть CRM",
            "needs_human_review": True,
        }
    )

    assert audit.contradictions[0].description == "Руководитель не подтвердил в CRM"
    assert audit.missing_information == ["подтверждение"]
    assert audit.checks == ["Проверить карточку"]


def test_parse_audit_rejects_incomplete_payload() -> None:
    with pytest.raises(LLMResponseError):
        parse_audit({"category": "priority_1", "rationale": "нет обязательных полей"})
