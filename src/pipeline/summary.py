"""Сводка результата: только подтверждённые факты и явные ограничения."""
from __future__ import annotations

from ..schemas import (
    AnalysisSummary,
    Category,
    ComponentStatus,
    LLMAnalysisResult,
    MergeResult,
    MLAnalysisResult,
    NormalizedCase,
    PROCESS_TYPE_TITLES_RU,
)

FALLBACK_CHECKS: dict[Category, list[str]] = {
    Category.PRIORITY_1: [
        "Немедленно проверить в CRM статус обязательного действия и срок.",
        "Связаться с ответственным и зафиксировать результат в официальном процессе.",
    ],
    Category.REQUIRES_INTERVENTION: [
        "Сверить CRM и чат: какие обязательные подтверждения отсутствуют.",
        "Запросить подтверждение у ответственного и зафиксировать его в CRM.",
    ],
    Category.NO_INTERVENTION: [
        "Убедиться, что в CRM остаются подтверждения приёмки/согласования.",
    ],
}

FALLBACK_ACTIONS: dict[Category, str] = {
    Category.PRIORITY_1: "Поднять кейс ответственному руководителю и проверить критический шаг.",
    Category.REQUIRES_INTERVENTION: "Проверить недостающие подтверждения и назначить ответственного.",
    Category.NO_INTERVENTION: "Закрыть кейс, сохранив ссылки на подтверждения в CRM.",
}

# Если ML не видит проблем, но LLM недоступен, нельзя формулировать проверки так,
# будто подтверждения уже признаны отсутствующими. Причина проверки — отсутствие
# независимого аудита, а не автоматически найденный дефект.
MANUAL_REVIEW_CHECKS_WHEN_LLM_UNAVAILABLE: list[str] = [
    "Проверить CRM и чат вручную: автоматический LLM-аудит недоступен.",
    "Убедиться, что обязательные подтверждения есть в CRM, а не только в переписке.",
]
MANUAL_REVIEW_ACTION_WHEN_LLM_UNAVAILABLE = (
    "Провести ручную проверку кейса и подтвердить итоговую категорию."
)
MANUAL_REVIEW_START_WHEN_LLM_UNAVAILABLE = (
    "Проверить кейс вручную: LLM-аудит недоступен, автоматическое подтверждение невозможно."
)


def _fallback(values: dict[Category, list[str]], category: Category) -> list[str]:
    return list(values.get(category, values[Category.REQUIRES_INTERVENTION]))


def build_summary(
    case: NormalizedCase,
    ml: MLAnalysisResult,
    llm: LLMAnalysisResult,
    merge: MergeResult,
) -> AnalysisSummary:
    audit = llm.audit if llm.status == ComponentStatus.OK else None
    ml_says_no_intervention = (
        ml.status == ComponentStatus.OK
        and ml.prediction is not None
        and ml.prediction.category == Category.NO_INTERVENTION
    )
    manual_review_due_to_missing_llm = ml_says_no_intervention and llm.status != ComponentStatus.OK

    if manual_review_due_to_missing_llm:
        checks = list(MANUAL_REVIEW_CHECKS_WHEN_LLM_UNAVAILABLE)
        actions = [MANUAL_REVIEW_ACTION_WHEN_LLM_UNAVAILABLE]
        start_with = MANUAL_REVIEW_START_WHEN_LLM_UNAVAILABLE
    else:
        checks = (
            list(audit.checks)
            if audit and audit.checks
            else _fallback(FALLBACK_CHECKS, merge.final_category)
        )
        actions = (
            list(audit.recommended_actions)
            if audit and audit.recommended_actions
            else [FALLBACK_ACTIONS[merge.final_category]]
        )
        if audit and audit.start_with:
            start_with = audit.start_with
        else:
            start_with = FALLBACK_ACTIONS[merge.final_category]

    warnings: list[str] = [f"Данные: {message}" for message in case.warnings]
    if ml.status != ComponentStatus.OK:
        warnings.append(f"ML: {ml.error or ml.status.value}")
    if llm.status != ComponentStatus.OK:
        warnings.append(f"LLM: {llm.error or llm.status.value}")
    warnings.extend(f"LLM: {message}" for message in llm.warnings)
    if audit is not None and audit.uncertainty_reasons:
        warnings.extend(f"LLM: {message}" for message in audit.uncertainty_reasons)

    prediction = ml.prediction
    return AnalysisSummary(
        case_id=case.case_id,
        process_type=case.process_type,
        process_type_title=PROCESS_TYPE_TITLES_RU[case.process_type],
        final_category=merge.final_category,
        title=merge.title,
        known_facts=list(audit.known_facts) if audit else [],
        contradictions=list(audit.contradictions) if audit else [],
        missing_information=list(audit.missing_information) if audit else [],
        checks=checks,
        recommended_actions=actions,
        start_with=start_with,
        priority_rationale=list(merge.reasons),
        human_review_required=merge.human_review_required,
        ml_category=prediction.category if prediction else None,
        ml_probabilities=dict(prediction.probabilities) if prediction else {},
        ml_margin=prediction.margin if prediction else None,
        llm_category=audit.category if audit else None,
        llm_rationale=audit.rationale if audit else llm.error,
        component_warnings=warnings,
    )
