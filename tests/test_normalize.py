"""Нормализация ручного ввода."""
from __future__ import annotations

from src.pipeline.normalize import normalize_case, parse_chat_text
from src.schemas import CaseInput, ChatMessage, ProcessType


def test_parse_chat_with_timestamp_role_and_continuation() -> None:
    raw = (
        "[10:15] Анна (Руководитель): Документы получила\n"
        "Иван: Спасибо\n"
        "продолжение сообщения"
    )
    messages, warnings = parse_chat_text(raw)

    assert warnings == []
    assert [(m.author, m.role, m.timestamp) for m in messages] == [
        ("Анна", "Руководитель", "10:15"),
        ("Иван", None, None),
    ]
    assert messages[1].text == "Спасибо\nпродолжение сообщения"


def test_parse_chat_without_headers_is_marked_unknown() -> None:
    messages, warnings = parse_chat_text("просто текст без автора")

    assert len(messages) == 1
    assert messages[0].author == "Неизвестно"
    assert any("без автора" in warning or "не распознаны" in warning for warning in warnings)


def test_normalize_generates_stable_manual_id() -> None:
    case_input = CaseInput(
        process_type=ProcessType.CARD_DELIVERY,
        crm_text="CRM: доставка до 10:00, статус «В пути».",
        chat_messages=[ChatMessage(author="Курьер", text="Буду после 13:00.")],
    )

    first = normalize_case(case_input)
    second = normalize_case(case_input)

    assert first.case_id == second.case_id
    assert first.case_id.startswith("MANUAL-")
    assert any("ID кейса не указан" in warning for warning in first.warnings)


def test_normalize_truncates_long_crm_and_keeps_warning() -> None:
    case_input = CaseInput(
        case_id="LONG-1",
        process_type=ProcessType.ONBOARDING,
        crm_text="данные " * 200,
    )

    result = normalize_case(case_input, max_input_chars=100)

    assert "текст обрезан" in result.crm_text
    assert any("CRM-текст обрезан" in warning for warning in result.warnings)
    assert any("Сообщения чата не заполнены" in warning for warning in result.warnings)
