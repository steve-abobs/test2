"""Инварианты детерминированной merge policy."""
from __future__ import annotations

from src.pipeline.merge import merge_results
from src.pipeline.normalize import normalize_case
from src.pipeline.summary import build_summary
from src.schemas import (
    CaseInput,
    Category,
    ComponentStatus,
    Fact,
    LLMAudit,
    LLMAnalysisResult,
    MLPrediction,
    MLAnalysisResult,
    ProcessType,
    SourceType,
)


def make_ml(
    category: Category | None,
    *,
    status: ComponentStatus = ComponentStatus.OK,
    low_margin: bool = False,
) -> MLAnalysisResult:
    if status != ComponentStatus.OK or category is None:
        return MLAnalysisResult(status=status, error="компонент недоступен")
    probabilities = {item: 0.1 for item in Category}
    probabilities[category] = 0.8
    prediction = MLPrediction(
        category=category,
        probabilities=probabilities,
        margin=0.7,
        low_margin=low_margin,
        model_name="logreg",
        embedding_model="intfloat/multilingual-e5-small",
    )
    return MLAnalysisResult(status=ComponentStatus.OK, prediction=prediction)


def make_llm(
    category: Category | None,
    *,
    status: ComponentStatus = ComponentStatus.OK,
    needs_review: bool = False,
    facts: bool = True,
    contradictions: bool = False,
    missing_information: bool = False,
) -> LLMAnalysisResult:
    if status != ComponentStatus.OK or category is None:
        return LLMAnalysisResult(status=status, error="LLM недоступен")
    audit = LLMAudit(
        category=category,
        rationale="Обоснование на основе источников.",
        known_facts=(
            [
                Fact(
                    fact="CRM содержит запись о процессе.",
                    source_type=SourceType.CRM,
                    source_reference="CRM",
                )
            ]
            if facts
            else []
        ),
        contradictions=(
            [{"description": "CRM и чат расходятся", "sources": ["CRM"]}]
            if contradictions
            else []
        ),
        missing_information=["подтверждение руководителя"] if missing_information else [],
        checks=["Проверить CRM."],
        recommended_actions=["Запросить подтверждение."],
        start_with="Открыть карточку в CRM.",
        needs_human_review=needs_review,
    )
    return LLMAnalysisResult(status=ComponentStatus.OK, audit=audit)


def test_no_intervention_requires_agreement_and_evidence() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(Category.NO_INTERVENTION),
    )

    assert result.final_category == Category.NO_INTERVENTION
    assert result.human_review_required is False
    assert "NO_INTERVENTION_CONFIRMED" in result.rule_codes


def test_no_intervention_blocked_when_llm_not_configured() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(None, status=ComponentStatus.NOT_CONFIGURED),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert result.human_review_required is True
    assert "LLM_NOT_CONFIGURED" in result.rule_codes


def test_no_intervention_blocked_when_ml_unavailable() -> None:
    result = merge_results(
        make_ml(None, status=ComponentStatus.UNAVAILABLE),
        make_llm(Category.NO_INTERVENTION),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert result.human_review_required is True
    assert "ML_UNAVAILABLE" in result.rule_codes


def test_ml_no_intervention_cannot_downgrade_llm_requires_intervention() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(Category.REQUIRES_INTERVENTION),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert result.human_review_required is True
    assert "ML_LLM_DISAGREEMENT" in result.rule_codes


def test_priority_1_from_ml_is_sticky() -> None:
    result = merge_results(
        make_ml(Category.PRIORITY_1),
        make_llm(Category.NO_INTERVENTION),
    )

    assert result.final_category == Category.PRIORITY_1
    assert result.human_review_required is True
    assert "PRIORITY1_FROM_ML" in result.rule_codes


def test_priority_1_from_llm_is_sticky_even_if_ml_fails() -> None:
    result = merge_results(
        make_ml(None, status=ComponentStatus.ERROR),
        make_llm(Category.PRIORITY_1),
    )

    assert result.final_category == Category.PRIORITY_1
    assert result.human_review_required is True
    assert "PRIORITY1_FROM_LLM" in result.rule_codes


def test_low_margin_forces_human_review_and_blocks_no_intervention() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION, low_margin=True),
        make_llm(Category.NO_INTERVENTION),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert result.human_review_required is True
    assert "ML_LOW_MARGIN" in result.rule_codes


def test_llm_review_flag_forces_requires_intervention() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(Category.NO_INTERVENTION, needs_review=True),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert "LLM_NEEDS_REVIEW" in result.rule_codes


def test_llm_contradictions_raise_floor() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(Category.NO_INTERVENTION, contradictions=True),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert "LLM_CONTRADICTIONS" in result.rule_codes


def test_llm_no_intervention_without_facts_is_not_enough() -> None:
    result = merge_results(
        make_ml(Category.NO_INTERVENTION),
        make_llm(Category.NO_INTERVENTION, facts=False),
    )

    assert result.final_category == Category.REQUIRES_INTERVENTION
    assert result.human_review_required is True
    assert "INSUFFICIENT_EVIDENCE" in result.rule_codes


def test_summary_attributes_manual_review_to_missing_llm_not_missing_facts() -> None:
    case = normalize_case(
        CaseInput(
            case_id="S-1",
            process_type=ProcessType.CARD_DELIVERY_RECEIPT,
            crm_text="Получатель подтвердил приёмку карты в CRM.",
            chat_messages=[
                {"author": "Получатель", "role": "Клиент", "text": "Карту получил, всё в порядке."}
            ],
        )
    )
    ml = make_ml(Category.NO_INTERVENTION)
    llm = make_llm(None, status=ComponentStatus.NOT_CONFIGURED)
    merge = merge_results(ml, llm)

    summary = build_summary(case, ml, llm, merge)

    assert summary.final_category == Category.REQUIRES_INTERVENTION
    assert summary.human_review_required is True
    assert "llm-аудит недоступен" in summary.start_with.lower()
    assert any("LLM-аудит недоступен" in check for check in summary.checks)
    assert all("Запросить подтверждение" not in action for action in summary.recommended_actions)
