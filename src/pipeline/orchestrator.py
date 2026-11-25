"""Оркестрация: независимый ML и LLM, затем детерминированный merge."""
from __future__ import annotations

from ..config import Settings
from ..llm.audit import audit_case
from ..llm.client import DeepSeekClient
from ..ml.service import MLService
from ..schemas import AnalysisResult, CaseInput, NormalizedCase
from .merge import merge_results
from .normalize import normalize_case
from .summary import build_summary


class AnalysisPipeline:
    """Один экземпляр на UI-сессию: ML-модель и embedding-кэш переиспользуются."""

    def __init__(self, settings: Settings, *, llm_client: DeepSeekClient | None = None):
        self.settings = settings
        self.ml = MLService(settings)
        self.llm_client = llm_client

    def analyze(self, case_input: CaseInput | NormalizedCase) -> AnalysisResult:
        if isinstance(case_input, CaseInput):
            case = normalize_case(case_input, max_input_chars=self.settings.max_input_chars)
        else:
            case = case_input

        # ML получает только нормализованный кейс; LLM — тот же кейс без ML-выхода.
        ml_result = self.ml.analyze(case)
        llm_result = audit_case(case, self.settings, client=self.llm_client)
        merge = merge_results(ml_result, llm_result)
        summary = build_summary(case, ml_result, llm_result, merge)
        return AnalysisResult(case=case, ml=ml_result, llm=llm_result, merge=merge, summary=summary)


def analyze_case(
    case_input: CaseInput | NormalizedCase,
    settings: Settings,
    *,
    pipeline: AnalysisPipeline | None = None,
    llm_client: DeepSeekClient | None = None,
) -> AnalysisResult:
    pipeline = pipeline or AnalysisPipeline(settings, llm_client=llm_client)
    return pipeline.analyze(case_input)
