"""Пакетная генерация синтетических кейсов через DeepSeek.

Генератор не используется для inference. Все ответы проходят Pydantic-валидацию;
ошибки сохраняются, а не превращаются в «успешные» кейсы.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Settings
from ..schemas import Category, ChatMessage, ProcessType, RiskLevel, SyntheticCase
from ..utils import stable_hash
from ..llm.client import DeepSeekClient, LLMResponseError

GENERATION_SYSTEM_PROMPT = """Ты создаёшь учебный синтетический датасет для службы поддержки.
Верни только JSON-объект вида {"cases": [...]}, без markdown и пояснений.
Каждый кейс строго соответствует схеме:
{
  "case_id": "уникальная строка",
  "scenario_family": "латиницей, устойчивое имя сценария",
  "process_type": "onboarding | dismissal | card_delivery | card_delivery_receipt | other",
  "crm_text": "текст CRM-карточки",
  "chat_messages": [{"author": "...", "role": "...", "text": "..."}],
  "label": "priority_1 | requires_intervention | no_intervention",
  "label_rationale": "почему именно эта категория",
  "required_evidence": ["минимум 2 обязательных подтверждения"],
  "risk_level": "low | medium | high"
}

Смысловые требования:
- priority_1: критический срок сегодня/истёк или уже возникший существенный риск,
  при этом обязательное действие не выполнено/не подтверждено.
- requires_intervention: противоречие, нестыковка статусов, неясный ответственный
  или недостающие данные без установленной критической срочности.
- no_intervention: подтверждения достаточны, CRM и чат согласуются, открытых проблем нет.
- «Отправил документы» не равно «получены и проверены»; «руководитель в курсе» не равно
  формальному подтверждению; обещание курьера не равно доставке. CRM и чат — разные источники.
- Не делай класс зависимым от одного слова: меняй формулировки, процессы, города, сроки.
- Внутри одного scenario_family сохраняй один и тот же паттерн риска, меняя детали.
- Не воспроизводи следующие контрольные кейсы и их формулировки: C-17, C-24, P-08, P-09.
- Пиши по-русски, естественно и разнообразно; без персональных данных реальных людей.
"""


@dataclass
class GenerationStats:
    requested: int = 0
    received: int = 0
    accepted: int = 0
    errors: int = 0
    batches: int = 0
    error_messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchRequest:
    label: Category
    process_type: ProcessType
    count: int


def plan_batches(target_size: int, batch_size: int) -> list[BatchRequest]:
    """Равномерно распределяет классы и процессы по пакетам."""
    if target_size < 1:
        raise ValueError("target_size должен быть >= 1")
    if batch_size < 1:
        raise ValueError("batch_size должен быть >= 1")
    requests: list[BatchRequest] = []
    produced = 0
    labels = list(Category)
    processes = list(ProcessType)
    index = 0
    while produced < target_size:
        count = min(batch_size, target_size - produced)
        requests.append(
            BatchRequest(
                label=labels[index % len(labels)],
                process_type=processes[index % len(processes)],
                count=count,
            )
        )
        produced += count
        index += 1
    return requests


def build_batch_prompt(request: BatchRequest, batch_number: int) -> str:
    return (
        f"Пакет {batch_number}. Сгенерируй {request.count} кейсов.\n"
        f"Целевая категория всех кейсов: {request.label.value}.\n"
        f"Целевой process_type всех кейсов: {request.process_type.value}.\n"
        "Используй 2–4 разных scenario_family внутри пакета и 2–5 сообщений чата "
        "в каждом кейсе. Не смешивай категории. Верни только JSON."
    )


class DeepSeekDatasetGenerator:
    def __init__(self, settings: Settings, *, client: DeepSeekClient | None = None):
        self.settings = settings
        self.client = client or DeepSeekClient(settings)
        if not settings.has_deepseek_key:
            raise LLMResponseError(
                "Генерация через DeepSeek недоступна: DEEPSEEK_API_KEY не задан. "
                "Используйте --engine offline или задайте ключ."
            )

    def generate(
        self,
        *,
        target_size: int = 1200,
        batch_size: int | None = None,
    ) -> tuple[list[SyntheticCase], GenerationStats]:
        batch_size = batch_size or self.settings.generation_batch_size
        requests = plan_batches(target_size, batch_size)
        stats = GenerationStats(requested=target_size, batches=len(requests))
        cases: list[SyntheticCase] = []
        used_ids: set[str] = set()

        for batch_number, request in enumerate(requests, start=1):
            try:
                payload = self.client.complete_json(
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    user_prompt=build_batch_prompt(request, batch_number),
                )
            except Exception as exc:  # noqa: BLE001 - ошибки сохраняются в статистике
                stats.errors += 1
                stats.error_messages.append(f"Пакет {batch_number}: {exc}")
                continue

            raw_cases = payload.get("cases")
            if not isinstance(raw_cases, list):
                stats.errors += 1
                stats.error_messages.append(f"Пакет {batch_number}: нет списка cases")
                continue

            accepted_in_batch = 0
            for index, raw in enumerate(raw_cases):
                stats.received += 1
                if accepted_in_batch >= request.count:
                    break
                if not isinstance(raw, dict):
                    stats.errors += 1
                    stats.error_messages.append(
                        f"Пакет {batch_number}, кейс {index}: ожидался объект"
                    )
                    continue
                candidate = dict(raw)
                candidate["case_id"] = f"DS-{batch_number:04d}-{index:02d}"
                try:
                    case = SyntheticCase.model_validate(candidate)
                except Exception as exc:  # noqa: BLE001 - фиксируем причину отказа
                    stats.errors += 1
                    stats.error_messages.append(
                        f"Пакет {batch_number}, кейс {index}: {str(exc).splitlines()[0]}"
                    )
                    continue
                if case.label != request.label or case.process_type != request.process_type:
                    stats.errors += 1
                    stats.error_messages.append(
                        f"Пакет {batch_number}, кейс {index}: не совпали label/process "
                        f"({case.label.value}, {case.process_type.value})"
                    )
                    continue
                if case.case_id in used_ids:
                    stats.errors += 1
                    stats.error_messages.append(
                        f"Пакет {batch_number}, кейс {index}: дубликат case_id {case.case_id}"
                    )
                    continue
                used_ids.add(case.case_id)
                cases.append(case)
                accepted_in_batch += 1

            if accepted_in_batch < request.count:
                stats.errors += 1
                stats.error_messages.append(
                    f"Пакет {batch_number}: принято {accepted_in_batch} из {request.count}"
                )

        stats.accepted = len(cases)
        return cases, stats


def normalize_generated_ids(cases: list[SyntheticCase], *, prefix: str = "DS") -> list[SyntheticCase]:
    """Страховка от повторяющихся ID у сторонних моделей."""
    seen: set[str] = set()
    normalized: list[SyntheticCase] = []
    for case in cases:
        case_id = case.case_id
        if case_id in seen:
            case_id = f"{prefix}-{stable_hash(case.model_dump_json(), prefix='case-id')[:12]}"
        seen.add(case_id)
        normalized.append(case.model_copy(update={"case_id": case_id}))
    return normalized


def make_offline_case(
    *,
    case_id: str,
    scenario_family: str,
    process_type: ProcessType,
    crm_text: str,
    chat: list[tuple[str, str | None, str]],
    label: Category,
    rationale: str,
    required_evidence: list[str],
    risk_level: RiskLevel,
) -> SyntheticCase:
    """Удобный конструктор для шаблонного fallback-корпуса."""
    return SyntheticCase(
        case_id=case_id,
        scenario_family=scenario_family,
        process_type=process_type,
        crm_text=crm_text,
        chat_messages=[
            ChatMessage(author=author, role=role, text=text) for author, role, text in chat
        ],
        label=label,
        label_rationale=rationale,
        required_evidence=required_evidence,
        risk_level=risk_level,
    )

