"""Интеграция пайплайна без сети: деградация компонентов не маскируется."""
from __future__ import annotations

from dataclasses import replace

from src.llm.audit import audit_case
from src.pipeline.normalize import normalize_case
from src.pipeline.orchestrator import AnalysisPipeline
from src.schemas import CaseInput, Category, ComponentStatus, ProcessType


def make_input() -> CaseInput:
    return CaseInput(
        case_id="INT-1",
        process_type=ProcessType.DISMISSAL,
        crm_text="Последний рабочий день сегодня; подтверждение руководителя отсутствует.",
        chat_messages=[
            {"author": "Иван", "role": "Сотрудник", "text": "Руководитель в курсе."},
        ],
    )


def test_pipeline_reports_degraded_components_and_requires_review(settings) -> None:
    result = AnalysisPipeline(settings).analyze(make_input())

    assert result.ml.status == ComponentStatus.UNAVAILABLE
    assert result.llm.status == ComponentStatus.NOT_CONFIGURED
    assert result.merge.final_category == Category.REQUIRES_INTERVENTION
    assert result.merge.human_review_required is True
    assert result.summary.known_facts == []
    assert result.summary.ml_category is None
    assert result.summary.llm_category is None
    assert result.summary.component_warnings
    assert result.llm.details == {}


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_audit_case_returns_ok_for_valid_fake_response_and_hides_key(settings) -> None:
    settings = replace(settings, deepseek_api_key="sk-test-secret", deepseek_model="deepseek-chat")
    client = FakeClient(
        {
            "category": "requires_intervention",
            "rationale": "CRM не содержит подтверждения руководителя.",
            "known_facts": [
                {
                    "fact": "Подтверждение руководителя отсутствует",
                    "source_type": "crm",
                    "source_reference": "CRM",
                    "source_quote": "подтверждение руководителя отсутствует",
                }
            ],
            "contradictions": [],
            "missing_information": ["подтверждение руководителя"],
            "checks": ["Проверить карточку увольнения"],
            "recommended_actions": ["Запросить подтверждение"],
            "start_with": "Открыть карточку увольнения",
            "needs_human_review": True,
        }
    )
    case = normalize_case(make_input())

    result = audit_case(case, settings, client=client)

    assert result.status == ComponentStatus.OK
    assert result.audit is not None
    assert result.audit.category == Category.REQUIRES_INTERVENTION
    assert "sk-test-secret" not in str(result.details)
    assert "sk-test-secret" not in client.calls[0][1]
    assert "Последний рабочий день сегодня" in client.calls[0][1]
    assert "вероятн" not in client.calls[0][1]
