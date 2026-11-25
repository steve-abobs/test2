"""Grouped train/validation/test split по scenario_family."""
from __future__ import annotations

from collections import Counter

from sklearn.model_selection import GroupShuffleSplit

from ..schemas import SplitBundle, SyntheticCase


class SplitError(RuntimeError):
    pass


def split_by_family(
    cases: list[SyntheticCase],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_attempts: int = 100,
) -> tuple[SplitBundle, dict[str, int]]:
    """Делит кейсы по scenario_family, чтобы близкие варианты не утекали.

    Делает несколько детерминированных попыток, пока во всех split не
    окажутся все три класса. Семья целиком попадает только в один split.
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Доли split должны давать 1.0")
    if len(cases) < 15:
        raise SplitError("Слишком мало кейсов для grouped split")
    labels = {case.label.value for case in cases}
    if len(labels) < 3:
        raise SplitError(f"В датасете не все классы: {sorted(labels)}")

    idx = list(range(len(cases)))
    groups = [case.scenario_family for case in cases]
    y = [case.label.value for case in cases]

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        outer = GroupShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=current_seed
        )
        train_val_idx, test_idx = next(outer.split(idx, y, groups))
        train_val_groups = [groups[i] for i in train_val_idx]
        validation_share = validation_ratio / (train_ratio + validation_ratio)
        inner = GroupShuffleSplit(
            n_splits=1, test_size=validation_share, random_state=current_seed + 10_000
        )
        train_rel, validation_rel = next(
            inner.split(train_val_idx, [y[i] for i in train_val_idx], train_val_groups)
        )
        train_idx = [train_val_idx[i] for i in train_rel]
        validation_idx = [train_val_idx[i] for i in validation_rel]

        split_labels = {
            "train": {y[i] for i in train_idx},
            "validation": {y[i] for i in validation_idx},
            "test": {y[i] for i in test_idx},
        }
        if all(len(values) == len(labels) for values in split_labels.values()):
            bundle = SplitBundle(
                train=[cases[i].case_id for i in train_idx],
                validation=[cases[i].case_id for i in validation_idx],
                test=[cases[i].case_id for i in test_idx],
            )
            family_counts = {
                "train": len({groups[i] for i in train_idx}),
                "validation": len({groups[i] for i in validation_idx}),
                "test": len({groups[i] for i in test_idx}),
            }
            _assert_no_family_leak(cases, bundle)
            _assert_no_reference_leak(cases, bundle)
            return bundle, family_counts

    raise SplitError(
        "Не удалось получить grouped split со всеми классами во всех частях. "
        "Увеличьте датасет или число scenario_family."
    )


def _assert_no_family_leak(cases: list[SyntheticCase], bundle: SplitBundle) -> None:
    by_id = {case.case_id: case for case in cases}
    family_sets = {
        name: {by_id[case_id].scenario_family for case_id in ids}
        for name, ids in (
            ("train", bundle.train),
            ("validation", bundle.validation),
            ("test", bundle.test),
        )
    }
    overlap = (
        (family_sets["train"] & family_sets["validation"])
        | (family_sets["train"] & family_sets["test"])
        | (family_sets["validation"] & family_sets["test"])
    )
    if overlap:
        raise SplitError(f"Leakage scenario_family между split: {sorted(overlap)}")


def _assert_no_reference_leak(cases: list[SyntheticCase], bundle: SplitBundle) -> None:
    by_id = {case.case_id: case for case in cases}
    reference_like = {"C-17", "C-24", "P-08", "P-09"}
    leaked = {
        case_id
        for name in ("train", "validation", "test")
        for case_id in getattr(bundle, name)
        if case_id in reference_like or case_id in by_id and by_id[case_id].case_id.startswith("REF-")
    }
    if leaked:
        raise SplitError(f"Контрольные кейсы попали в split: {sorted(leaked)}")


def distribution(cases: list[SyntheticCase], bundle: SplitBundle) -> dict[str, dict[str, int]]:
    by_id = {case.case_id: case for case in cases}
    result: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        counts = Counter(by_id[case_id].label.value for case_id in getattr(bundle, name))
        result[name] = dict(counts)
    return result
