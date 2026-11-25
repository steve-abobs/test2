"""Нормализация ручного ввода: CRM-текст и обычный текст чата.

Пользователь не обязан вставлять JSON. Поддерживаются форматы:
    Иван: текст
    Иван (Руководитель): текст
    [10:15] Курьер: текст
    Курьер — текст
Строки без «автора» считаются продолжением предыдущего сообщения; если
сообщений ещё нет — создаётся сообщение с автором «Неизвестно» и
предупреждением.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from ..schemas import CaseInput, ChatMessage, NormalizedCase
from ..utils import stable_hash, truncate_text

_HEADER_RE = re.compile(
    r"^\s*(?:\[(?P<timestamp>[^\]]{1,40})\]\s*)?"
    r"(?P<author>[^:\n]{1,80}?)\s*"
    r"(?:\((?P<role>[^()\n]{1,80})\))?\s*"
    r"[:—–-]\s*(?P<text>.+?)\s*$"
)


def _clean_block(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
            cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(line.strip())
    return "\n".join(cleaned).strip()


def _looks_like_author(author: str) -> bool:
    author = author.strip()
    if not author or not any(char.isalpha() for char in author):
        return False
    lowered = author.casefold()
    if lowered.startswith(("http://", "https://", "www.")):
        return False
    if "://" in author or author.count("/") > 1:
        return False
    # Строка вида «10:30» не должна становиться автором.
    if all(char.isdigit() or char in ".:- " for char in author):
        return False
    return True


def _match_header(line: str) -> re.Match[str] | None:
    match = _HEADER_RE.match(line)
    if not match:
        return None
    if not _looks_like_author(match.group("author")):
        return None
    if not (match.group("text") or "").strip():
        return None
    return match


def parse_chat_text(raw: str, *, max_message_chars: int = 20_000) -> tuple[list[ChatMessage], list[str]]:
    """Парсит свободный текст чата в список сообщений."""
    warnings: list[str] = []
    raw = (raw or "").strip()
    if not raw:
        return [], warnings

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    messages: list[ChatMessage] = []
    current_author: str | None = None
    current_role: str | None = None
    current_timestamp: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_author, current_role, current_timestamp, current_lines
        if current_author is None:
            current_lines = []
            return
        text = "\n".join(line for line in current_lines).strip()
        if text:
            text, truncated = truncate_text(text, max_message_chars)
            if truncated:
                warnings.append(
                    f"Сообщение от «{current_author}» обрезано до {max_message_chars} символов."
                )
            messages.append(
                ChatMessage(
                    author=current_author.strip() or "Неизвестно",
                    role=(current_role or "").strip() or None,
                    text=text,
                    timestamp=(current_timestamp or "").strip() or None,
                )
            )
        current_author, current_role, current_timestamp, current_lines = None, None, None, []

    any_header = False
    for line in lines:
        stripped = line.strip()
        match = _match_header(stripped)
        if match:
            any_header = True
            flush()
            current_author = match.group("author")
            current_role = match.group("role")
            current_timestamp = match.group("timestamp")
            current_lines = [match.group("text")]
            continue
        if not stripped:
            continue
        if current_author is None:
            current_author = "Неизвестно"
            current_lines = [stripped]
            warnings.append(
                "В чате найдена строка без автора; она сохранена как сообщение «Неизвестно»."
            )
        else:
            current_lines.append(stripped)
    flush()

    if not messages:
        # Ни одного заголовка: делим по пустым строкам, иначе по строкам.
        blocks = re.split(r"\n\s*\n", raw)
        if len(blocks) == 1:
            blocks = [line for line in lines if line.strip()]
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            text, truncated = truncate_text(block, max_message_chars)
            if truncated:
                warnings.append(f"Фрагмент чата обрезан до {max_message_chars} символов.")
            messages.append(ChatMessage(author="Неизвестно", text=text))
        if messages:
            warnings.append(
                "Авторы сообщений не распознаны: используйте формат «Автор: текст». "
                "Все сообщения сохранены как «Неизвестно»."
            )

    if not any_header and len(messages) > 1:
        warnings.append(
            "Формат «Автор: текст» не найден. Проверьте, что авторы сообщений указаны явно."
        )
    return messages, warnings


def normalize_case(case_input: CaseInput, *, max_input_chars: int = 20_000) -> NormalizedCase:
    """Нормализует ввод, не теряя предупреждения о качестве данных."""
    warnings: list[str] = []

    crm_text = _clean_block(case_input.crm_text)
    if crm_text:
        crm_text, crm_truncated = truncate_text(crm_text, max_input_chars)
        if crm_truncated:
            warnings.append(f"CRM-текст обрезан до {max_input_chars} символов.")
    else:
        warnings.append("CRM-текст не заполнен: вывод LLM и merge ограничены сообщениями чата.")

    messages: list[ChatMessage] = []
    for message in case_input.chat_messages:
        clean_text = _clean_block(message.text)
        if not clean_text:
            continue
        clean_text, truncated = truncate_text(clean_text, max_input_chars)
        if truncated:
            warnings.append(f"Сообщение от «{message.author}» обрезано до {max_input_chars} символов.")
        author = message.author.strip() or "Неизвестно"
        if author.casefold() in {"неизвестно", "unknown", "неизвестный"}:
            warnings.append("В чате есть сообщение без автора: источник указан как «Неизвестно».")
        messages.append(
            ChatMessage(
                author=author,
                role=message.role,
                text=clean_text,
                timestamp=message.timestamp,
            )
        )

    if not messages:
        warnings.append("Сообщения чата не заполнены: подтверждения из чата отсутствуют.")

    case_id = case_input.case_id
    if not case_id:
        seed = f"{case_input.process_type.value}|{crm_text}|" + "|".join(
            f"{item.author}:{item.text}" for item in messages
        )
        case_id = f"MANUAL-{stable_hash(seed, prefix='manual')[:10]}"
        warnings.append("ID кейса не указан; сгенерирован технический ID.")

    return NormalizedCase(
        case_id=case_id,
        process_type=case_input.process_type,
        crm_text=crm_text,
        chat_messages=messages,
        warnings=warnings,
    )


def default_case_id(process_type_value: str) -> str:
    """Технический ID для UI до вызова normalize_case."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{process_type_value.upper()}-{stamp}"
