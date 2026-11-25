"""Детерминированная merge policy ML и LLM.

Главные инварианты:
- итоговая категория никогда не ниже любой доступной категории компонентов;
- ``priority_1`` от любого компонента сохраняется;
- ``no_intervention`` возможен только при согласии обоих компонентов,
  достаточных подтверждениях и отсутствии причин для ручной проверки;
- сбой/недоступность компонента исключает ``no_intervention``.
"""
from __future__ import annotations

from ..schemas import (
    CATEGORY_ORDER,
    CATEGORY_TITLES_RU,
    Category,
    ComponentStatus,
    LLMAnalysisResult,
    MergeResult,
    MLAnalysisResult,
)

RULE_TITLES: dict[str, str] = {
    "ML_LLM_AGREEMENT": "ML и LLM дали одинаковую категорию.",
    "ML_LLM_DISAGREEMENT": "ML и LLM расходятся: выбрана более осторожная категория.",
    "ML_LOW_MARGIN": "ML-предсказание с низким margin: нужна ручная проверка.",
    "ML_UNAVAILABLE": "ML-компонент недоступен: автоматический вывод ограничен.",
    "ML_ERROR": "ML-компонент завершился ошибкой: нужна ручная проверка.",
    "LLM_NOT_CONFIGURED": "LLM-аудит не настроен: нужна ручная проверка.",
    "LLM_ERROR": "LLM-аудит завершился ошибкой: нужна ручная проверка.",
    "LLM_NEEDS_REVIEW": "LLM отметил необходимость ручной проверки.",
    "LLM_CONTRADICTIONS": "LLM нашёл противоречия между источниками.",
    "LLM_MISSING_INFORMATION": "LLM указал недостающие обязательные сведения.",
    "INSUFFICIENT_EVIDENCE": "Подтверждений недостаточно для no_intervention.",
    "PRIORITY1_FROM_ML": "ML предложил priority_1; понижение запрещено.",
    "PRIORITY1_FROM_LLM": "LLM предложил priority_1; понижение запрещено.",
    "NO_INTERVENTION_CONFIRMED": "Оба компонента согласны, подтверждения достаточны.",
    "DEFAULT_REQUIRES_INTERVENTION": "Нет надёжного основания для иной категории.",
}

MIN_INTERVENTION = Category.REQUIRES_INTERVENTION


def _audit_floor(audit) -> Category | None:
    """Минимально допустимая категория из ограничений LLM-аудита."""
    if audit.category == Category.PRIORITY_1:
        return Category.PRIORITY_1
    if audit.contradictions or audit.missing_information or audit.needs_human_review:
        return MIN_INTERVENTION
    return None


def _evidence_sufficient(llm: LLMAnalysisResult) -> bool:
    if llm.status != ComponentStatus.OK or llm.audit is None:
        return False
    audit = llm.audit
    if not audit.known_facts:
        return False
    if audit.contradictions or audit.missing_information or audit.needs_human_review:
        return False
    return all(fact.source_reference.strip() for fact in audit.known_facts)


def merge_results(ml: MLAnalysisResult, llm: LLMAnalysisResult) -> MergeResult:
    reasons: list[str] = []
    codes: list[str] = []

    def note(code: str) -> None:
        if code not in codes:
            codes.append(code)
            reasons.append(RULE_TITLES[code])

    def at_least(category: Category) -> Category:
        return max((final, category), key=lambda item: CATEGORY_ORDER[item])

    ml_category = ml.prediction.category if ml.prediction is not None else None
    llm_category = llm.audit.category if llm.audit is not None else None
    evidence_sufficient = _evidence_sufficient(llm)
    human_review = False

    if ml.status == ComponentStatus.OK:
        if ml.prediction is not None and ml.prediction.low_margin:
            note("ML_LOW_MARGIN")
            human_review = True
    elif ml.status == ComponentStatus.UNAVAILABLE:
        note("ML_UNAVAILABLE")
        human_review = True
    else:
        note("ML_ERROR")
        human_review = True

    if llm.status == ComponentStatus.NOT_CONFIGURED:
        note("LLM_NOT_CONFIGURED")
        human_review = True
    elif llm.status != ComponentStatus.OK or llm.audit is None:
        note("LLM_ERROR")
        human_review = True

    available = [category for category in (ml_category, llm_category) if category is not None]
    if ml_category is not None and llm_category is not None:
        if ml_category == llm_category:
            note("ML_LLM_AGREEMENT")
        else:
            note("ML_LLM_DISAGREEMENT")
            human_review = True
    if available:
        final = max(available, key=lambda item: CATEGORY_ORDER[item])
    else:
        final = MIN_INTERVENTION
        note("DEFAULT_REQUIRES_INTERVENTION")

    if ml_category == Category.PRIORITY_1:
        note("PRIORITY1_FROM_ML")
    if llm_category == Category.PRIORITY_1:
        note("PRIORITY1_FROM_LLM")

    if llm.status == ComponentStatus.OK and llm.audit is not None:
        audit = llm.audit
        if audit.needs_human_review:
            note("LLM_NEEDS_REVIEW")
            human_review = True
        if audit.contradictions:
            note("LLM_CONTRADICTIONS")
            human_review = True
        if audit.missing_information:
            note("LLM_MISSING_INFORMATION")
            human_review = True
        if not evidence_sufficient and llm_category != Category.PRIORITY_1:
            note("INSUFFICIENT_EVIDENCE")
            human_review = True
        floor = _audit_floor(audit)
        if floor is not None:
            final = at_least(floor)

    if ml.status != ComponentStatus.OK or llm.status != ComponentStatus.OK:
        final = at_least(MIN_INTERVENTION)

    if final == Category.NO_INTERVENTION:
        if (
            ml_category == Category.NO_INTERVENTION
            and llm_category == Category.NO_INTERVENTION
            and evidence_sufficient
            and not human_review
        ):
            note("NO_INTERVENTION_CONFIRMED")
        else:
            note("INSUFFICIENT_EVIDENCE")
            human_review = True
            final = MIN_INTERVENTION

    if final == Category.PRIORITY_1:
        human_review = True
    if not reasons:
        note("DEFAULT_REQUIRES_INTERVENTION")

    return MergeResult(
        final_category=final,
        title=CATEGORY_TITLES_RU[final],
        human_review_required=human_review,
        reasons=reasons,
        rule_codes=codes,
        evidence_sufficient=evidence_sufficient,
    )

