#!/usr/bin/env python
"""Генерация датасета и grouped split.

Примеры:
    python scripts/generate_dataset.py --engine offline
    python scripts/generate_dataset.py --engine auto --target-size 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data_generation.dataset import DatasetError, build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["auto", "deepseek", "offline"], default="auto")
    parser.add_argument("--target-size", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-per-family", type=int, default=10)
    args = parser.parse_args()

    settings = load_settings()
    try:
        result = build_dataset(
            settings,
            engine=args.engine,
            target_size=args.target_size,
            batch_size=args.batch_size,
            max_per_family=args.max_per_family,
        )
    except DatasetError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2

    payload = {
        "status": "ok",
        "engine": result.metadata.engine,
        "engine_note": result.metadata.engine_note,
        "actual_size": result.metadata.actual_size,
        "label_counts": result.metadata.label_counts,
        "process_type_counts": result.metadata.process_type_counts,
        "family_count": result.metadata.family_count,
        "split_sizes": result.metadata.split_sizes,
        "split_family_counts": result.metadata.split_family_counts,
        "split_distribution": result.split_distribution,
        "dedup_removed": result.metadata.dedup_removed,
        "excluded_reference_cases": result.metadata.excluded_reference_cases,
        "generation_errors": result.metadata.generation_errors,
        "paths": {name: str(path) for name, path in result.paths.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
