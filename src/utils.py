"""Небольшие общие утилиты без побочных эффектов."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def stable_hash(text: str, *, prefix: str = "") -> str:
    digest = hashlib.sha256(f"{prefix}\x1f{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def normalize_for_dedup(text: str) -> str:
    """Нормализация для поиска точных дублей (регистр, пробелы, пунктуация)."""
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def case_dedup_key(case: Any) -> str:
    """Ключ дедупликации по процессу, CRM и всем сообщениям чата."""
    chat = " || ".join(
        f"{getattr(message, 'author', '')}:{getattr(message, 'text', '')}"
        for message in getattr(case, "chat_messages", [])
    )
    payload = f"{getattr(case, 'process_type', '')}|{case.crm_text}|{chat}"
    return stable_hash(normalize_for_dedup(payload), prefix="case")


def dedupe_preserve_order(items: Iterable[T], key_func) -> tuple[list[T], int]:
    seen: set[Any] = set()
    result: list[T] = []
    removed = 0
    for item in items:
        key = key_func(item)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        result.append(item)
    return result, removed


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    ensure_dir(path.parent)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n…[текст обрезан]", True
