"""Pydantic-схемы проекта.

Схемы намеренно разделяют:
- исходные данные кейса (CRM и чат) — их получает LLM-аудитор;
- ML-предсказание — его LLM-аудитор не получает никогда;
- LLM-аудит — независимый результат;
- результат merge — детерминированное объединение.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    """Три допустимые категории кейса."""

    PRIORITY_1 = "priority_1"
    REQUIRES_INTERVENTION = "requires_intervention"
    NO_INTERVENTION = "no_intervention"


CATEGORY_ORDER: dict[Category, int] = {
    Category.NO_INTERVENTION: 0,
    Category.REQUIRES_INTERVENTION: 1,
    Category.PRIORITY_1: 2,
}

CATEGORY_TITLES_RU: dict[Category, str] = {
    Category.PRIORITY_1: "Приоритет 1 — срочное вмешательство",
    Category.REQUIRES_INTERVENTION: "Требуется проверка и вмешательство",
    Category.NO_INTERVENTION: "Дополнительное вмешательство не требуется",
}


class ProcessType(str, Enum):
    ONBOARDING = "onboarding"
    DISMISSAL = "dismissal"
    CARD_DELIVERY = "card_delivery"
    CARD_DELIVERY_RECEIPT = "card_delivery_receipt"
    OTHER = "other"


PROCESS_TYPE_TITLES_RU: dict[ProcessType, str] = {
    ProcessType.ONBOARDING: "Оформление сотрудника",
    ProcessType.DISMISSAL: "Увольнение сотрудника",
    ProcessType.CARD_DELIVERY: "Доставка карты",
    ProcessType.CARD_DELIVERY_RECEIPT: "Приёмка карты",
    ProcessType.OTHER: "Другое",
}


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceType(str, Enum):
    CRM = "crm"
    CHAT = "chat"


class ComponentStatus(str, Enum):
    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ChatMessage(BaseModel):
    """Одно сообщение рабочего чата.

    ``role`` опционален: во входном тексте роль может быть указана как
    «Автор (Роль): текст» либо отсутствовать.
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    author: str = Field(min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)
    timestamp: str | None = Field(default=None, max_length=100)

    @field_validator("author", "text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле не может быть пустым")
        return value.strip()

    @field_validator("role", "timestamp")
    @classmethod
    def _optional_clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


# ---------------------------------------------------------------------------
# Входные и нормализованные данные
# ---------------------------------------------------------------------------


class CaseInput(BaseModel):
    """Ручной ввод или контрольный кейс до нормализации."""

    case_id: str | None = Field(default=None, max_length=120)
    process_type: ProcessType
    crm_text: str = Field(default="", max_length=50_000)
    chat_messages: list[ChatMessage] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def _clean_case_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("crm_text")
    @classmethod
    def _strip_crm(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _require_some_source(self) -> "CaseInput":
        if not self.crm_text and not self.chat_messages:
            raise ValueError("нужен хотя бы один источник: CRM-текст или сообщения чата")
        return self


class NormalizedCase(BaseModel):
    """Нормализованный кейс. Индексы сообщений — с 1, как в UI и промпте."""

    case_id: str
    process_type: ProcessType
    crm_text: str
    chat_messages: list[ChatMessage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def message_by_index(self, index: int) -> ChatMessage | None:
        if 1 <= index <= len(self.chat_messages):
            return self.chat_messages[index - 1]
        return None

    def chat_reference_label(self, index: int) -> str:
        message = self.message_by_index(index)
        if message is None:
            return f"Чат, сообщение {index} (источник не найден)"
        role = f" ({message.role})" if message.role else ""
        return f"Чат, сообщение {index}: {message.author}{role}"


# ---------------------------------------------------------------------------
# LLM-аудит
# ---------------------------------------------------------------------------


class Fact(BaseModel):
    """Факт с обязательной привязкой к источнику."""

    fact: str = Field(min_length=1, max_length=2_000)
    source_type: SourceType
    source_reference: str = Field(min_length=1, max_length=200)
    source_quote: str | None = Field(default=None, max_length=2_000)

    @field_validator("fact", "source_reference")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле не может быть пустым")
        return value.strip()


class Contradiction(BaseModel):
    description: str = Field(min_length=1, max_length=2_000)
    sources: list[str] = Field(default_factory=list)
    resolution_check: str | None = Field(default=None, max_length=1_000)

    @field_validator("description")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("описание противоречия не может быть пустым")
        return value.strip()


class LLMAudit(BaseModel):
    """Структурированный ответ независимого DeepSeek-аудитора."""

    category: Category
    rationale: str = Field(min_length=1, max_length=5_000)
    known_facts: list[Fact] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    start_with: str = Field(min_length=1, max_length=1_000)
    needs_human_review: bool
    uncertainty_reasons: list[str] = Field(default_factory=list)

    @field_validator("rationale", "start_with")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле не может быть пустым")
        return value.strip()

    @field_validator("contradictions", mode="before")
    @classmethod
    def _coerce_contradictions(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            coerced: list[Any] = []
            for item in value:
                if isinstance(item, str):
                    coerced.append({"description": item, "sources": []})
                else:
                    coerced.append(item)
            return coerced
        if isinstance(value, str):
            return [{"description": value, "sources": []}]
        return value

    @field_validator("missing_information", "checks", "recommended_actions", "uncertainty_reasons", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------


class MLPrediction(BaseModel):
    category: Category
    probabilities: dict[Category, float]
    margin: float = Field(ge=0.0, le=1.0)
    low_margin: bool
    model_name: str
    embedding_model: str

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[Category, float]) -> dict[Category, float]:
        missing = set(Category) - set(value)
        if missing:
            raise ValueError(f"нет вероятностей для классов: {sorted(c.value for c in missing)}")
        cleaned: dict[Category, float] = {}
        for key, prob in value.items():
            prob = float(prob)
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"вероятность вне [0, 1]: {key}={prob}")
            cleaned[Category(key)] = prob
        total = sum(cleaned.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"сумма вероятностей должна быть 1.0, получено {total:.6f}")
        return cleaned

    def probability(self, category: Category) -> float:
        return float(self.probabilities.get(category, 0.0))


class MLAnalysisResult(BaseModel):
    status: ComponentStatus
    prediction: MLPrediction | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "MLAnalysisResult":
        if self.status == ComponentStatus.OK and self.prediction is None:
            raise ValueError("при статусе ok требуется prediction")
        return self


class LLMAnalysisResult(BaseModel):
    status: ComponentStatus
    audit: LLMAudit | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "LLMAnalysisResult":
        if self.status == ComponentStatus.OK and self.audit is None:
            raise ValueError("при статусе ok требуется audit")
        return self


# ---------------------------------------------------------------------------
# Merge / summary / result
# ---------------------------------------------------------------------------


class MergeResult(BaseModel):
    final_category: Category
    title: str
    human_review_required: bool
    reasons: list[str]
    rule_codes: list[str]
    evidence_sufficient: bool


class AnalysisSummary(BaseModel):
    case_id: str
    process_type: ProcessType
    process_type_title: str
    final_category: Category
    title: str
    known_facts: list[Fact]
    contradictions: list[Contradiction]
    missing_information: list[str]
    checks: list[str]
    recommended_actions: list[str]
    start_with: str
    priority_rationale: list[str]
    human_review_required: bool
    ml_category: Category | None
    ml_probabilities: dict[Category, float]
    ml_margin: float | None
    llm_category: Category | None
    llm_rationale: str | None
    component_warnings: list[str]


class AnalysisResult(BaseModel):
    case: NormalizedCase
    ml: MLAnalysisResult
    llm: LLMAnalysisResult
    merge: MergeResult
    summary: AnalysisSummary
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def technical_json(self) -> dict[str, Any]:
        """JSON для раскрываемого блока UI. Секретов здесь нет."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Синтетический датасет и контрольные кейсы
# ---------------------------------------------------------------------------


class SyntheticCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=120)
    scenario_family: str = Field(min_length=1, max_length=200)
    process_type: ProcessType
    crm_text: str = Field(min_length=1, max_length=20_000)
    chat_messages: list[ChatMessage] = Field(min_length=1)
    label: Category
    label_rationale: str = Field(min_length=1, max_length=5_000)
    required_evidence: list[str] = Field(min_length=1)
    risk_level: RiskLevel

    @field_validator("case_id", "scenario_family")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле не может быть пустым")
        return value.strip()

    @field_validator("required_evidence")
    @classmethod
    def _evidence_non_blank(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("required_evidence не может быть пустым")
        return cleaned

    @model_validator(mode="after")
    def _risk_consistency(self) -> "SyntheticCase":
        if self.label == Category.PRIORITY_1 and self.risk_level != RiskLevel.HIGH:
            raise ValueError("priority_1 должен иметь risk_level=high")
        if self.label == Category.NO_INTERVENTION and self.risk_level == RiskLevel.HIGH:
            raise ValueError("no_intervention не должен иметь risk_level=high")
        return self


class ReferenceCase(BaseModel):
    case_id: str
    process_type: ProcessType
    crm_text: str
    chat_messages: list[ChatMessage]
    expected_review_focus: list[str]


class DatasetMetadata(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine: str
    engine_note: str
    model: str | None = None
    target_size: int
    actual_size: int
    label_counts: dict[str, int]
    process_type_counts: dict[str, int]
    family_count: int
    split_sizes: dict[str, int]
    split_family_counts: dict[str, int]
    excluded_reference_cases: list[str]
    dedup_removed: int
    generation_errors: int
    source_files: dict[str, str]


class SplitBundle(BaseModel):
    train: list[str]
    validation: list[str]
    test: list[str]


class ModelMetrics(BaseModel):
    model_name: str
    split: Literal["validation", "test", "train"]
    sample_size: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str]
    calibration: dict[str, float]


class ModelBundleMetadata(BaseModel):
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_model: str
    e5_prefix: str
    classifier: str
    class_weight: str
    selected_c: float
    classes: list[str]
    low_margin_threshold: float
    dataset_engine: str
    dataset_size: int
    split_sizes: dict[str, int]
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics
    baseline_validation_metrics: ModelMetrics
    baseline_test_metrics: ModelMetrics
    warnings: list[str] = Field(default_factory=list)
