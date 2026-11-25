#!/usr/bin/env python
"""Аудит датасета: дубликаты, leakage, дисбаланс, шаблонность, логика.

Отчёт не является подтверждением production readiness.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data_generation.dedup import near_duplicate_pairs
from src.schemas import Category, DatasetMetadata, SplitBundle, SyntheticCase
from src.utils import case_dedup_key, ensure_dir, read_json, read_jsonl

STOPWORDS = {
    "который", "которая", "которые", "сегодня", "завтра", "сотрудник", "сотрудника",
    "клиент", "клиента", "нужно", "нужен", "нужна", "будет", "после", "через", "также",
    "сообщение", "сообщения", "подтверждение", "подтверждения", "процесс", "процесса",
    "руководитель", "руководителя", "карты", "карта", "курбер", "курьер", "курьера",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[а-яёa-z]{5,}", text.casefold())
        if token not in STOPWORDS
    }


def label_keyword_report(cases: list[SyntheticCase]) -> dict[str, list[str]]:
    token_sets = [
        (case.label.value, tokenize(f"{case.crm_text} {' '.join(m.text for m in case.chat_messages)}"))
        for case in cases
    ]
    counts = Counter(case.label.value for case in cases)
    result: dict[str, list[str]] = {}
    for label in (Category.PRIORITY_1.value, Category.REQUIRES_INTERVENTION.value, Category.NO_INTERVENTION.value):
        other_total = max(1, len(cases) - counts[label])
        scores: list[tuple[float, str]] = []
        vocabulary = set().union(*(tokens for _, tokens in token_sets))
        for token in vocabulary:
            in_label = sum(1 for item_label, tokens in token_sets if item_label == label and token in tokens)
            in_other = sum(1 for item_label, tokens in token_sets if item_label != label and token in tokens)
            share_label = in_label / max(1, counts[label])
            share_other = in_other / other_total
            if share_label >= 0.5 and share_other <= 0.05 and in_label >= 3:
                scores.append((share_label - share_other, token))
        scores.sort(reverse=True)
        result[label] = [token for _, token in scores[:8]]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-threshold", type=float, default=0.95)
    args = parser.parse_args()

    settings = load_settings()
    cases = [SyntheticCase.model_validate(row) for row in read_jsonl(settings.generated_dir / "dataset.jsonl")]
    split = SplitBundle.model_validate(read_json(settings.generated_dir / "split.json"))
    metadata = DatasetMetadata.model_validate(
        read_json(settings.generated_dir / "dataset_metadata.json")
    )
    by_id = {case.case_id: case for case in cases}
    failures: list[str] = []
    warnings: list[str] = []

    keys = Counter(case_dedup_key(case) for case in cases)
    exact_duplicates = sum(count - 1 for count in keys.values() if count > 1)
    if exact_duplicates:
        warnings.append(f"Найдено точных дублей: {exact_duplicates}.")

    near_pairs = near_duplicate_pairs(cases, threshold=args.near_threshold, max_pairs=25)
    if near_pairs:
        warnings.append(f"Найдено почти дублей (>{args.near_threshold}): {len(near_pairs)}+.")

    split_ids = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
    }
    for name, ids in split_ids.items():
        unknown = ids - set(by_id)
        if unknown:
            failures.append(f"{name}: ID отсутствуют в датасете: {sorted(unknown)[:5]}")
    all_split_ids = set().union(*split_ids.values())
    uncovered = set(by_id) - all_split_ids
    if uncovered:
        failures.append(f"Вне split осталось кейсов: {len(uncovered)}")
    overlap_ids = (
        (split_ids["train"] & split_ids["validation"])
        | (split_ids["train"] & split_ids["test"])
        | (split_ids["validation"] & split_ids["test"])
    )
    if overlap_ids:
        failures.append(f"ID встречаются в нескольких split: {sorted(overlap_ids)[:5]}")

    family_sets = {
        name: {by_id[case_id].scenario_family for case_id in ids if case_id in by_id}
        for name, ids in split_ids.items()
    }
    family_overlap = (
        (family_sets["train"] & family_sets["validation"])
        | (family_sets["train"] & family_sets["test"])
        | (family_sets["validation"] & family_sets["test"])
    )
    if family_overlap:
        failures.append(f"Leakage scenario_family между split: {sorted(family_overlap)[:5]}")

    family_labels: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        family_labels[case.scenario_family].add(case.label.value)
    mixed_families = {name: sorted(labels) for name, labels in family_labels.items() if len(labels) > 1}
    if mixed_families:
        failures.append(f"Семьи смешивают label: {list(mixed_families)[:5]}")

    counts = Counter(case.label.value for case in cases)
    smallest = min(counts.values())
    largest = max(counts.values())
    imbalance_ratio = round(largest / max(1, smallest), 2)
    if imbalance_ratio > 2.0:
        warnings.append(f"Дисбаланс классов: {imbalance_ratio} (max/min).")

    logical_issues: list[str] = []
    for case in cases:
        if case.label == Category.PRIORITY_1 and case.risk_level.value != "high":
            logical_issues.append(f"{case.case_id}: priority_1 без risk_level=high")
        if case.label == Category.NO_INTERVENTION and case.risk_level.value == "high":
            logical_issues.append(f"{case.case_id}: no_intervention с risk_level=high")
        if len(case.required_evidence) < 2:
            logical_issues.append(f"{case.case_id}: меньше двух required_evidence")
        if len(case.label_rationale.strip()) < 20:
            logical_issues.append(f"{case.case_id}: слишком короткий label_rationale")
    if logical_issues:
        failures.append(f"Логические несоответствия: {len(logical_issues)}")

    crm_counts = Counter(case.crm_text.strip() for case in cases)
    repeated_crm = [
        {"crm_text": text[:120], "count": count}
        for text, count in crm_counts.most_common(5)
        if count > 1
    ]
    repeated_share = (
        sum(count for count in crm_counts.values() if count > 1) / max(1, len(cases))
    )
    if repeated_share > 0.2:
        warnings.append(
            f"Высокая шаблонность CRM-текста: {repeated_share:.0%} кейсов используют повторяющиеся CRM."
        )

    keywords = label_keyword_report(cases)
    if any(keywords.values()):
        warnings.append("Обнаружены потенциальные label-маркеры; проверьте вручную.")

    status = "fail" if failures else ("warn" if warnings else "pass")
    report = {
        "status": status,
        "dataset_engine": metadata.engine,
        "dataset_engine_note": metadata.engine_note,
        "dataset_size": len(cases),
        "label_counts": dict(counts),
        "process_type_counts": dict(Counter(case.process_type.value for case in cases)),
        "family_count": len(family_labels),
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "split_family_counts": {name: len(values) for name, values in family_sets.items()},
        "checks": {
            "exact_duplicates": exact_duplicates,
            "near_duplicate_pairs": [list(pair) for pair in near_pairs],
            "family_leakage": sorted(family_overlap),
            "mixed_label_families": mixed_families,
            "imbalance_ratio": imbalance_ratio,
            "logical_issues": logical_issues,
            "repeated_crm_top": repeated_crm,
            "repeated_crm_share": round(repeated_share, 4),
            "potential_label_keywords": keywords,
        },
        "failures": failures,
        "warnings": warnings,
        "manual_checks": [
            "Выборочно прочитать по 10 кейсов каждого класса и проверить rationale/evidence.",
            "Проверить, что priority_1 действительно связан с истёкшим/сегодняшним сроком или риском.",
            "Проверить no_intervention на наличие формальных подтверждений в CRM.",
            "Регулярно перегенерировать DeepSeek-корпус и повторять аудит.",
            "Для реальных выводов нужны обезличенные реальные кейсы и разметка экспертов.",
        ],
        "production_ready": False,
        "disclaimer": (
            "Отчёт проверяет синтетический корпус и не подтверждает production readiness. "
            "Синтетические метрики не доказывают качество на реальных данных."
        ),
    }
    reports_dir = ensure_dir(settings.reports_dir)
    json_path = reports_dir / "dataset_quality.json"
    md_path = reports_dir / "dataset_quality.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Аудит качества датасета",
        "",
        f"- Статус автоматических проверок: **{status}**",
        f"- Движок: `{metadata.engine}`",
        f"- Размер: {len(cases)} кейсов, семей: {len(family_labels)}",
        f"- Классы: {dict(counts)}",
        f"- Split: {', '.join(f'{name}={len(ids)}' for name, ids in split_ids.items())}",
        "",
        "## Ошибки",
        *(f"- {item}" for item in (failures or ["нет"])),
        "",
        "## Предупреждения",
        *(f"- {item}" for item in (warnings or ["нет"])),
        "",
        "## Ручные проверки",
        *(f"- {item}" for item in report["manual_checks"]),
        "",
        "> Отчёт не подтверждает production readiness. Синтетические метрики не доказывают",
        "> качество на реальных данных; контрольные кейсы не являются тестовой выборкой.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


