"""Сборка датасета: генерация/fallback, исключение контрольных кейсов,
дедупликация, grouped split и сохранение метаданных.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..schemas import (
    Category,
    DatasetMetadata,
    ProcessType,
    ReferenceCase,
    SplitBundle,
    SyntheticCase,
)
from ..utils import case_dedup_key, ensure_dir, read_json, write_json, write_jsonl
from .dedup import DedupResult, remove_near_duplicates
from .generator import DeepSeekDatasetGenerator, GenerationStats
from .offline_generator import generate_offline_dataset
from .split import distribution, split_by_family

REFERENCE_CASE_IDS = ("C-17", "C-24", "P-08", "P-09")


class DatasetError(RuntimeError):
    pass


@dataclass
class DatasetBuildResult:
    cases: list[SyntheticCase]
    split: SplitBundle
    metadata: DatasetMetadata
    split_distribution: dict[str, dict[str, int]]
    dedup: DedupResult
    generation_stats: GenerationStats
    paths: dict[str, Path] = field(default_factory=dict)


def load_reference_cases(path: Path) -> list[ReferenceCase]:
    if not path.exists():
        raise DatasetError(f"Файл контрольных кейсов не найден: {path}")
    raw = read_json(path)
    if not isinstance(raw, list):
        raise DatasetError("reference_cases.json должен содержать список")
    return [ReferenceCase.model_validate(item) for item in raw]


def exclude_reference_cases(
    cases: list[SyntheticCase],
    references: list[ReferenceCase],
) -> tuple[list[SyntheticCase], list[str]]:
    """Убирает точные совпадения с контрольными кейсами по ID и тексту."""
    reference_ids = {case.case_id for case in references}
    reference_keys = {case_dedup_key(case) for case in references}
    excluded: list[str] = []
    kept: list[SyntheticCase] = []
    for case in cases:
        if case.case_id in reference_ids or case_dedup_key(case) in reference_keys:
            excluded.append(case.case_id)
            continue
        kept.append(case)
    return kept, excluded


def harmonize_families(cases: list[SyntheticCase]) -> tuple[list[SyntheticCase], list[str]]:
    """Гарантирует, что семья не смешивает label/process.

    Если генератор переиспользовал имя семьи, к нему добавляется суффикс,
    чтобы grouped split не создавал ложную утечку между классами.
    """
    families: dict[str, set[tuple[str, str]]] = {}
    for case in cases:
        families.setdefault(case.scenario_family, set()).add(
            (case.label.value, case.process_type.value)
        )
    warnings: list[str] = []
    result: list[SyntheticCase] = []
    for case in cases:
        if len(families[case.scenario_family]) > 1:
            new_family = f"{case.scenario_family}__{case.label.value}__{case.process_type.value}"
            warnings.append(
                f"scenario_family «{case.scenario_family}» смешивала классы/процессы; "
                f"кейс {case.case_id} переименован в «{new_family}»."
            )
            result.append(case.model_copy(update={"scenario_family": new_family}))
        else:
            result.append(case)
    return result, warnings


def ensure_balance(cases: list[SyntheticCase]) -> None:
    labels = {case.label for case in cases}
    missing = set(Category) - labels
    if missing:
        raise DatasetError(
            "В датасете нет всех классов: " + ", ".join(sorted(item.value for item in missing))
        )
    counts = Counter(case.label for case in cases)
    smallest = min(counts.values())
    largest = max(counts.values())
    if largest > smallest * 4 and len(cases) >= 30:
        raise DatasetError(
            "Сильный дисбаланс классов: "
            + ", ".join(f"{label.value}={counts[label]}" for label in Category)
        )


def build_dataset(
    settings: Settings,
    *,
    engine: str = "auto",
    target_size: int = 1200,
    batch_size: int | None = None,
    max_per_family: int = 10,
) -> DatasetBuildResult:
    """Создаёт датасет и сохраняет все артефакты.

    ``engine``: ``deepseek`` — только API, ``offline`` — только шаблоны,
    ``auto`` — DeepSeek при наличии ключа, иначе явный offline fallback.
    """
    if engine not in {"auto", "deepseek", "offline"}:
        raise ValueError("engine должен быть auto, deepseek или offline")

    references = load_reference_cases(settings.reference_cases_path)
    output_dir = ensure_dir(settings.generated_dir)
    raw_path = output_dir / "dataset_raw.jsonl"
    errors_path = output_dir / "generation_errors.jsonl"
    dataset_path = output_dir / "dataset.jsonl"
    split_path = output_dir / "split.json"
    metadata_path = output_dir / "dataset_metadata.json"
    warnings_path = output_dir / "dataset_warnings.json"

    if engine == "deepseek" and not settings.has_deepseek_key:
        raise DatasetError(
            "DEEPSEEK_API_KEY не задан. Задайте ключ или используйте --engine offline."
        )
    selected = "deepseek" if engine in {"auto", "deepseek"} and settings.has_deepseek_key else "offline"
    if selected == "deepseek":
        cases, stats = DeepSeekDatasetGenerator(settings).generate(
            target_size=target_size, batch_size=batch_size
        )
        engine_note = "Кейсы сгенерированы DeepSeek API и прошли Pydantic-валидацию."
        model = settings.deepseek_model
    else:
        cases, stats = generate_offline_dataset(max_per_family=max_per_family)
        engine_note = (
            "OFFLINE FALLBACK: использованы шаблонные кейсы без DeepSeek. "
            "Это bootstrap-корпус для проверки пайплайна, а не замена генеративной "
            "модели; метрики на нём не являются свидетельством реального качества."
        )
        model = None
        if engine == "auto":
            engine_note += " DEEPSEEK_API_KEY не задан, поэтому auto выбрал offline."

    write_jsonl(raw_path, cases)
    write_jsonl(
        errors_path,
        [{"error": message} for message in stats.error_messages],
    )

    kept, excluded = exclude_reference_cases(cases, references)
    dedup = remove_near_duplicates(kept)
    cases = dedup.kept
    cases, family_warnings = harmonize_families(cases)
    ensure_balance(cases)

    bundle, family_counts = split_by_family(cases)
    split_sizes = {
        "train": len(bundle.train),
        "validation": len(bundle.validation),
        "test": len(bundle.test),
    }
    write_jsonl(dataset_path, cases)
    write_json(split_path, bundle.model_dump(mode="json"))

    label_counts = Counter(case.label.value for case in cases)
    process_counts = Counter(case.process_type.value for case in cases)
    metadata = DatasetMetadata(
        engine=selected,
        engine_note=engine_note,
        model=model,
        target_size=target_size if selected == "deepseek" else len(cases),
        actual_size=len(cases),
        label_counts={label.value: label_counts.get(label.value, 0) for label in Category},
        process_type_counts={
            process.value: process_counts.get(process.value, 0) for process in ProcessType
        },
        family_count=len({case.scenario_family for case in cases}),
        split_sizes=split_sizes,
        split_family_counts=family_counts,
        excluded_reference_cases=list(excluded),
        dedup_removed=dedup.removed_total,
        generation_errors=stats.errors,
        source_files={
            "dataset": str(dataset_path),
            "raw_dataset": str(raw_path),
            "split": str(split_path),
            "reference_cases": str(settings.reference_cases_path),
            "generation_errors": str(errors_path),
            "warnings": str(warnings_path),
        },
    )
    write_json(metadata_path, metadata.model_dump(mode="json"))
    write_json(
        warnings_path,
        {
            "family_warnings": family_warnings,
            "near_duplicate_examples": dedup.near_duplicate_examples,
            "generation_errors": stats.error_messages,
        },
    )

    return DatasetBuildResult(
        cases=cases,
        split=bundle,
        metadata=metadata,
        split_distribution=distribution(cases, bundle),
        dedup=dedup,
        generation_stats=stats,
        paths={
            "dataset": dataset_path,
            "raw_dataset": raw_path,
            "split": split_path,
            "metadata": metadata_path,
            "errors": errors_path,
            "warnings": warnings_path,
        },
    )

