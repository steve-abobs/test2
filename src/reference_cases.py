"""Загрузка и запуск контрольных кейсов (не входят в обучающий датасет)."""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .schemas import AnalysisResult, CaseInput, ReferenceCase
from .utils import read_json


class ReferenceCaseError(RuntimeError):
    pass


def load_reference_cases(path: Path) -> list[ReferenceCase]:
    if not path.exists():
        raise ReferenceCaseError(f"Файл контрольных кейсов не найден: {path}")
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ReferenceCaseError("reference_cases.json должен содержать список")
    return [ReferenceCase.model_validate(item) for item in raw]


def reference_to_input(case: ReferenceCase) -> CaseInput:
    return CaseInput(
        case_id=case.case_id,
        process_type=case.process_type,
        crm_text=case.crm_text,
        chat_messages=case.chat_messages,
    )


def reference_result_row(reference: ReferenceCase, result: AnalysisResult) -> dict:
    """Плоская запись для отчёта; ожидания не смешиваются с выводом модели."""
    return {
        "case_id": reference.case_id,
        "process_type": reference.process_type.value,
        "crm_text": reference.crm_text,
        "chat_messages": [message.model_dump(mode="json") for message in reference.chat_messages],
        "expected_review_focus": list(reference.expected_review_focus),
        "final_category": result.summary.final_category.value,
        "human_review_required": result.summary.human_review_required,
        "ml_category": result.summary.ml_category.value if result.summary.ml_category else None,
        "ml_margin": result.summary.ml_margin,
        "llm_category": result.summary.llm_category.value if result.summary.llm_category else None,
        "llm_status": result.llm.status.value,
        "rule_codes": list(result.merge.rule_codes),
        "start_with": result.summary.start_with,
        "component_warnings": list(result.summary.component_warnings),
    }


def run_reference_cases(
    settings: Settings,
    *,
    pipeline=None,
) -> list[dict]:
    from .pipeline.orchestrator import AnalysisPipeline

    pipeline = pipeline or AnalysisPipeline(settings)
    rows: list[dict] = []
    for reference in load_reference_cases(settings.reference_cases_path):
        result = pipeline.analyze(reference_to_input(reference))
        rows.append(reference_result_row(reference, result))
    return rows
