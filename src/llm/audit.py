"""Независимый LLM-аудит кейса.

На вход аудитору подаются только ID, тип процесса, CRM-текст и сообщения чата.
ML-предсказание, вероятности, margin, правила merge и ожидаемые ответы
контрольных кейсов сюда не передаются.
"""
from __future__ import annotations

from pydantic import ValidationError

from ..config import Settings
from ..schemas import (
    ComponentStatus,
    LLMAnalysisResult,
    LLMAudit,
    NormalizedCase,
    PROCESS_TYPE_TITLES_RU,
)
from .client import DeepSeekClient, LLMNotConfiguredError, LLMResponseError

SYSTEM_PROMPT = """Ты — независимый аудитор клиентских кейсов службы поддержки.
Твоя задача — по CRM-тексту и сообщениям рабочего чата определить, нужен ли
срочный приоритет, проверка/вмешательство или дополнительных действий не требуется.

Обязательные правила:
1. Разделяй источники. CRM-запись и сообщение чата — разные источники; не смешивай их.
2. Слова сотрудника, курьера, клиента или руководителя в чате НЕ являются формальным
   подтверждением действия в CRM. «Я отправил», «мы обсудили», «я в курсе» — это не
   подтверждение получения, проверки, согласования или возврата.
3. Явно учитывай сроки: «сегодня», «завтра», истёкший срок, начало смены.
   Неподтверждённое обязательное действие в критический срок — сильный признак priority_1.
4. Фиксируй противоречия между CRM и чатом, а также противоречия внутри источника.
5. Не додумывай отсутствующие факты. Если данных нет — вынеси их в missing_information.
6. Недостаток информации сам по себе НЕ означает no_intervention. В таком случае выбирай
   requires_intervention и needs_human_review=true.
7. no_intervention допустим только если все обязательные подтверждения присутствуют,
   противоречий нет и нет причин для ручной проверки.
8. checks и recommended_actions должны быть конкретными: что именно проверить, у кого,
   в какой системе, какой документ/подтверждение запросить.
9. Каждый known_fact обязан иметь source_type и точную source_reference.
   Для CRM используй reference «CRM». Для чата — «Чат, сообщение N: Автор (Роль)»,
   ровно как в нумерованном списке.
10. Ответ — только валидный JSON без markdown. Все тексты — на русском языке.

Категории:
- priority_1 — критический риск уже возник или срок сегодня/истёк, обязательное действие
  не выполнено; возможны безопасность, утрата имущества/карт, срыв смены или выплаты.
- requires_intervention — есть противоречие, неподтверждённое обязательное действие,
  неясный ответственный или неполные данные, но нет оснований для немедленного priority_1.
- no_intervention — обязательные шаги подтверждены, противоречий и открытых рисков нет.

Формат JSON:
{
  "category": "priority_1 | requires_intervention | no_intervention",
  "rationale": "краткое обоснование со ссылкой на сроки, подтверждения и противоречия",
  "known_facts": [
    {"fact": "что известно", "source_type": "crm | chat",
     "source_reference": "CRM или Чат, сообщение N: Автор (Роль)",
     "source_quote": "короткая цитата или null"}
  ],
  "contradictions": [
    {"description": "в чём противоречие", "sources": ["CRM", "Чат, сообщение 2: ..."],
     "resolution_check": "как проверить"}
  ],
  "missing_information": ["чего не хватает"],
  "checks": ["конкретная проверка"],
  "recommended_actions": ["конкретное следующее действие"],
  "start_with": "одно первое действие",
  "needs_human_review": true,
  "uncertainty_reasons": ["почему нужна ручная проверка"]
}"""


def build_audit_prompts(case: NormalizedCase) -> tuple[str, str]:
    """Собирает prompts строго из исходных данных кейса."""
    lines = [
        f"ID кейса: {case.case_id}",
        f"Тип процесса: {case.process_type.value} — {PROCESS_TYPE_TITLES_RU[case.process_type]}",
        "",
        "CRM-ТЕКСТ:",
        case.crm_text.strip() or "[CRM-текст отсутствует]",
        "",
        "СООБЩЕНИЯ ЧАТА (нумерация с 1):",
    ]
    if case.chat_messages:
        for index, message in enumerate(case.chat_messages, start=1):
            role = f" ({message.role})" if message.role else ""
            timestamp = f" [{message.timestamp}]" if message.timestamp else ""
            lines.append(f"{index}. {message.author}{role}{timestamp}: {message.text}")
    else:
        lines.append("[сообщения чата отсутствуют]")
    lines.extend(
        [
            "",
            "Проверь расхождения между CRM и чатом, сроки и наличие формальных",
            "подтверждений. Верни только JSON по заданной схеме.",
        ]
    )
    return SYSTEM_PROMPT, "\n".join(lines)


def parse_audit(payload: dict) -> LLMAudit:
    try:
        return LLMAudit.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise LLMResponseError(
            f"LLM вернул JSON, не соответствующий схеме: {location}: {first.get('msg', 'ошибка')}"
        ) from exc


def audit_case(
    case: NormalizedCase,
    settings: Settings,
    client: DeepSeekClient | None = None,
) -> LLMAnalysisResult:
    """Возвращает независимый аудит или явный статус недоступности."""
    if not settings.has_deepseek_key:
        return LLMAnalysisResult(
            status=ComponentStatus.NOT_CONFIGURED,
            error=(
                "DEEPSEEK_API_KEY не задан. Независимый LLM-аудит не выполнялся; "
                "итоговая категория требует ручной проверки."
            ),
            warnings=["LLM-аудит недоступен: отсутствует DEEPSEEK_API_KEY."],
        )

    client = client or DeepSeekClient(settings)
    system_prompt, user_prompt = build_audit_prompts(case)
    try:
        payload = client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    except LLMNotConfiguredError as exc:
        return LLMAnalysisResult(
            status=ComponentStatus.NOT_CONFIGURED,
            error=str(exc),
            warnings=["LLM-аудит недоступен: отсутствует DEEPSEEK_API_KEY."],
        )
    except LLMResponseError as exc:
        return LLMAnalysisResult(
            status=ComponentStatus.ERROR,
            error=str(exc),
            warnings=["LLM-аудит не получен; требуется ручная проверка."],
        )
    except Exception as exc:  # noqa: BLE001 - не маскируем сбой под результат
        return LLMAnalysisResult(
            status=ComponentStatus.ERROR,
            error=f"Неожиданная ошибка LLM-аудита: {exc}",
            warnings=["LLM-аудит не получен; требуется ручная проверка."],
        )

    try:
        audit = parse_audit(payload)
    except LLMResponseError as exc:
        return LLMAnalysisResult(
            status=ComponentStatus.ERROR,
            error=str(exc),
            warnings=["Ответ LLM не прошёл валидацию; требуется ручная проверка."],
            details={"raw_keys": sorted(payload.keys())},
        )

    warnings: list[str] = []
    if audit.contradictions and not audit.needs_human_review:
        warnings.append(
            "LLM указал противоречия, но не отметил необходимость ручной проверки; "
            "merge принудительно направит кейс на проверку."
        )
    if audit.category.value == "priority_1" and not audit.needs_human_review:
        warnings.append("LLM выбрал priority_1 без отметки ручной проверки; приоритет сохраняется.")
    return LLMAnalysisResult(
        status=ComponentStatus.OK,
        audit=audit,
        warnings=warnings,
        details={
            "model": settings.deepseek_model,
            "base_url": settings.deepseek_base_url,
            "temperature": 0.0,
        },
    )


