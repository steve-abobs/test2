"""Дедупликация датасета: точные и почти точные дубликаты."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schemas import SyntheticCase
from ..utils import case_dedup_key


@dataclass
class DedupResult:
    kept: list[SyntheticCase]
    removed_exact: int = 0
    removed_near: int = 0
    near_duplicate_examples: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def removed_total(self) -> int:
        return self.removed_exact + self.removed_near


def remove_exact_duplicates(cases: list[SyntheticCase]) -> tuple[list[SyntheticCase], int]:
    seen: set[str] = set()
    kept: list[SyntheticCase] = []
    removed = 0
    for case in cases:
        key = case_dedup_key(case)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(case)
    return kept, removed


def remove_near_duplicates(
    cases: list[SyntheticCase],
    *,
    threshold: float = 0.98,
    max_examples: int = 20,
) -> DedupResult:
    """Удаляет почти одинаковые тексты по char TF-IDF cosine.

    Порог намеренно высокий, чтобы не удалять разные сценарии одной семьи.
  """
    exact_kept, removed_exact = remove_exact_duplicates(cases)
    if len(exact_kept) < 2:
        return DedupResult(kept=exact_kept, removed_exact=removed_exact)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        texts = [
            f"{case.process_type.value} {case.crm_text} "
            + " ".join(message.text for message in case.chat_messages)
            for case in exact_kept
        ]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
        matrix = vectorizer.fit_transform(texts)
        similarity = linear_kernel(matrix)
    except Exception:
        # Дедупликация не должна блокировать сбор датасета.
        return DedupResult(kept=exact_kept, removed_exact=removed_exact)

    removed_indices: set[int] = set()
    examples: list[tuple[str, str, float]] = []
    for first in range(len(exact_kept)):
        if first in removed_indices:
            continue
        for second in range(first + 1, len(exact_kept)):
            if second in removed_indices:
                continue
            score = float(similarity[first, second])
            if score >= threshold:
                removed_indices.add(second)
                if len(examples) < max_examples:
                    examples.append(
                        (exact_kept[first].case_id, exact_kept[second].case_id, round(score, 4))
                    )
    kept = [case for index, case in enumerate(exact_kept) if index not in removed_indices]
    return DedupResult(
        kept=kept,
        removed_exact=removed_exact,
        removed_near=len(removed_indices),
        near_duplicate_examples=examples,
    )


def near_duplicate_pairs(
    cases: list[SyntheticCase], *, threshold: float = 0.95, max_pairs: int = 50
) -> list[tuple[str, str, float]]:
    """Только для аудита: находит пары похожих текстов, ничего не удаляя."""
    if len(cases) < 2:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        texts = [
            f"{case.process_type.value} {case.crm_text} "
            + " ".join(message.text for message in case.chat_messages)
            for case in cases
        ]
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
        similarity = linear_kernel(matrix)
    except Exception:
        return []
    pairs: list[tuple[str, str, float]] = []
    upper = np.triu_indices(len(cases), k=1)
    scores = similarity[upper]
    order = np.argsort(scores)[::-1]
    for position in order:
        if len(pairs) >= max_pairs:
            break
        score = float(scores[position])
        if score < threshold:
            break
        first, second = int(upper[0][position]), int(upper[1][position])
        pairs.append((cases[first].case_id, cases[second].case_id, round(score, 4)))
    return pairs
