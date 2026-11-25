"""Клиент DeepSeek через OpenAI-совместимый API.

Ключ берётся только из настроек, никогда не логируется и не возвращается
в сообщениях об ошибках.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from ..config import Settings


class LLMNotConfiguredError(RuntimeError):
    """DEEPSEEK_API_KEY отсутствует."""


class LLMResponseError(RuntimeError):
    """API вернул ошибку, таймаут или невалидный JSON."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any]:
    """Достаёт JSON-объект из ответа, допуская ```json-обёртку и текст вокруг."""
    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("В ответе LLM не найден JSON-объект") from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Невалидный JSON от LLM: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError("Ожидался JSON-объект, получен другой тип")
    return payload


class DeepSeekClient:
    """Минимальная обёртка с ограниченными retry и без утечки ключа."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    @property
    def available(self) -> bool:
        return self.settings.has_deepseek_key

    def _get_client(self):
        if not self.available:
            raise LLMNotConfiguredError(
                "DEEPSEEK_API_KEY не задан: независимый LLM-аудит недоступен."
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise LLMResponseError(
                    "Пакет openai не установлен; выполните pip install -r requirements.txt"
                ) from exc
            self._client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                timeout=self.settings.deepseek_timeout_seconds,
                max_retries=0,
            )
        return self._client

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Возвращает разобранный JSON-объект или поднимает контролируемую ошибку."""
        client = self._get_client()
        last_error = "неизвестная ошибка"
        for attempt in range(self.settings.deepseek_max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.settings.deepseek_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise LLMResponseError("Пустой ответ LLM")
                return extract_json_object(content)
            except LLMResponseError:
                raise
            except Exception as exc:  # noqa: BLE001 - нормализуем ошибки SDK
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.settings.deepseek_max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
        raise LLMResponseError(
            f"DeepSeek API недоступен после {self.settings.deepseek_max_retries + 1} попыток: {last_error}"
        )
