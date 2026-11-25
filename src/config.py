"""Конфигурация приложения.

Ключ DeepSeek читается только из переменной окружения DEEPSEEK_API_KEY
(или локального .env). Ключ никогда не логируется и не попадает в UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть числом, получено: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    embedding_model: str
    ml_low_margin_threshold: float
    data_dir: Path
    artifacts_dir: Path
    generated_dir: Path
    reports_dir: Path
    reference_cases_path: Path
    deepseek_timeout_seconds: float
    deepseek_max_retries: int
    generation_batch_size: int
    max_input_chars: int
    e5_prefix: str = "query: "
    request_timeout_seconds: float = 60.0
    random_seed: int = 42
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def has_deepseek_key(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def llm_status_note(self) -> str:
        if self.has_deepseek_key:
            return f"LLM-аудит доступен (модель {self.deepseek_model})."
        return (
            "DEEPSEEK_API_KEY не задан: независимый LLM-аудит недоступен. "
            "Итоговая категория требует ручной проверки."
        )

    def public_summary(self) -> dict[str, object]:
        """Безопасная информация для UI/логов: ключ не раскрывается."""
        return {
            "deepseek_configured": self.has_deepseek_key,
            "deepseek_base_url": self.deepseek_base_url,
            "deepseek_model": self.deepseek_model,
            "embedding_model": self.embedding_model,
            "ml_low_margin_threshold": self.ml_low_margin_threshold,
            "artifacts_dir": str(self.artifacts_dir),
            "data_dir": str(self.data_dir),
            "max_input_chars": self.max_input_chars,
            "e5_prefix": self.e5_prefix,
        }

    def validate(self) -> None:
        if not (0.0 <= self.ml_low_margin_threshold <= 1.0):
            raise ValueError("ML_LOW_MARGIN_THRESHOLD должен быть в диапазоне [0, 1]")
        if self.deepseek_max_retries < 0:
            raise ValueError("DEEPSEEK_MAX_RETRIES не может быть отрицательным")
        if self.deepseek_timeout_seconds <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT_SECONDS должен быть положительным")
        if self.max_input_chars < 500:
            raise ValueError("MAX_INPUT_CHARS слишком мал")


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Загружает .env (если есть) и переменные окружения.

    ``load_dotenv`` не перезаписывает уже заданные переменные окружения.
    """
    if env_file is None:
        env_file = PROJECT_ROOT / ".env"
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path, override=False)

    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    generated_dir = Path(os.getenv("GENERATED_DATA_DIR", data_dir / "generated"))

    settings = Settings(
        deepseek_api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip() or None,
        deepseek_base_url=(os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip(),
        deepseek_model=(os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip(),
        embedding_model=(os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-small").strip(),
        ml_low_margin_threshold=_env_float("ML_LOW_MARGIN_THRESHOLD", 0.15),
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        generated_dir=generated_dir,
        reports_dir=Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports")),
        reference_cases_path=Path(
            os.getenv("REFERENCE_CASES_PATH", data_dir / "reference_cases.json")
        ),
        deepseek_timeout_seconds=_env_float("DEEPSEEK_TIMEOUT_SECONDS", 60.0),
        deepseek_max_retries=_env_int("DEEPSEEK_MAX_RETRIES", 3),
        generation_batch_size=_env_int("GENERATION_BATCH_SIZE", 8),
        max_input_chars=_env_int("MAX_INPUT_CHARS", 20_000),
        request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 60.0),
        random_seed=_env_int("RANDOM_SEED", 42),
    )
    settings.validate()
    return settings
