"""Формирование входа Sentence Transformer.

E5-модели обучены с префиксами: ``query:`` для запросов/классифицируемого
текста и ``passage:`` для документов. Для классификации мы используем
``query: `` и при обучении, и при inference — ровно один и тот же формат.
"""
from __future__ import annotations

from ..schemas import ChatMessage, NormalizedCase, ProcessType

DEFAULT_E5_PREFIX = "query: "


def format_chat_message(index: int, message: ChatMessage) -> str:
    role = f" ({message.role})" if message.role else ""
    timestamp = f" [{message.timestamp}]" if message.timestamp else ""
    return f"[{index}] {message.author}{role}{timestamp}: {message.text}"


def build_ml_text(
    *,
    process_type: ProcessType,
    crm_text: str,
    chat_messages: list[ChatMessage],
    e5_prefix: str = DEFAULT_E5_PREFIX,
) -> str:
    """Собирает единый вход ML: тип процесса + CRM + чат с маркировкой."""
    chat_part = " || ".join(
        format_chat_message(index, message) for index, message in enumerate(chat_messages, start=1)
    )
    if not chat_part:
        chat_part = "нет сообщений"
    crm_part = crm_text.strip() if crm_text.strip() else "CRM не заполнен"
    return (
        f"{e5_prefix}Тип процесса: {process_type.value}\n"
        f"CRM: {crm_part}\n"
        f"CHAT: {chat_part}"
    )


def build_ml_text_for_case(case: NormalizedCase, e5_prefix: str = DEFAULT_E5_PREFIX) -> str:
    return build_ml_text(
        process_type=case.process_type,
        crm_text=case.crm_text,
        chat_messages=case.chat_messages,
        e5_prefix=e5_prefix,
    )
