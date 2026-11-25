"""Сборка offline-датасета, split и исключение контрольных кейсов."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_settings
from src.data_generation.dataset import build_dataset, exclude_reference_cases
from src.reference_cases import load_reference_cases
from src.schemas import Category, ProcessType, RiskLevel, SyntheticCase
from src.utils import case_dedup_key


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("dataset")
    base = load_settings()
    settings = replace(
        base,
        deepseek_api_key=None,
        data_dir=root / "data",
        generated_dir=root / "data" / "generated",
        artifacts_dir=root / "artifacts",
        reports_dir=root / "reports",
        reference_cases_path=PROJECT_ROOT / "data" / "reference_cases.json",
    )
    return settings, build_dataset(settings, engine="offline", max_per_family=3)


def test_offline_dataset_is_explicit_and_covers_classes(built_dataset) -> None:
    _, result = built_dataset

    assert result.metadata.engine == "offline"
    assert "OFFLINE FALLBACK" in result.metadata.engine_note
    assert {case.label for case in result.cases} == set(Category)
    assert {case.process_type for case in result.cases} == set(ProcessType)
    assert len(result.cases) >= 30


def test_reference_cases_are_not_in_dataset_or_split(built_dataset) -> None:
    settings, result = built_dataset
    references = load_reference_cases(settings.reference_cases_path)
    reference_ids = {case.case_id for case in references}
    reference_keys = {case_dedup_key(case) for case in references}
    dataset_ids = {case.case_id for case in result.cases}

    assert not (dataset_ids & reference_ids)
    assert not ({case_dedup_key(case) for case in result.cases} & reference_keys)
    for split_ids in (result.split.train, result.split.validation, result.split.test):
        assert not (set(split_ids) & reference_ids)


def test_split_has_no_family_leakage_and_covers_dataset(built_dataset) -> None:
    _, result = built_dataset
    by_id = {case.case_id: case for case in result.cases}
    split_ids = {
        "train": set(result.split.train),
        "validation": set(result.split.validation),
        "test": set(result.split.test),
    }

    assert set().union(*split_ids.values()) == set(by_id)
    families = {
        name: {by_id[case_id].scenario_family for case_id in ids}
        for name, ids in split_ids.items()
    }
    assert not (families["train"] & families["validation"])
    assert not (families["train"] & families["test"])
    assert not (families["validation"] & families["test"])


def test_dataset_metadata_matches_written_files(built_dataset) -> None:
    settings, result = built_dataset

    assert (settings.generated_dir / "dataset.jsonl").exists()
    assert (settings.generated_dir / "split.json").exists()
    assert (settings.generated_dir / "dataset_metadata.json").exists()
    assert result.metadata.actual_size == len(result.cases)
    assert sum(result.metadata.split_sizes.values()) == len(result.cases)


def test_exclude_reference_cases_matches_by_dedup_key() -> None:
    reference = load_reference_cases(PROJECT_ROOT / "data" / "reference_cases.json")[0]
    clone = SyntheticCase(
        case_id="REF-CLONE-1",
        scenario_family="reference_clone",
        process_type=reference.process_type,
        crm_text=reference.crm_text,
        chat_messages=reference.chat_messages,
        label=Category.REQUIRES_INTERVENTION,
        label_rationale="Контрольный кейс исключается из обучения по совпадению текста.",
        required_evidence=["подтверждение", "срок"],
        risk_level=RiskLevel.MEDIUM,
    )

    kept, excluded = exclude_reference_cases([clone], [reference])

    assert kept == []
    assert excluded == ["REF-CLONE-1"]
