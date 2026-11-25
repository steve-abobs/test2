#!/usr/bin/env python
"""Прогон контрольных кейсов через полный пайплайн.

Контрольные кейсы не входят в обучение и не являются независимой тестовой
выборкой; скрипт не считает по ним accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.reference_cases import run_reference_cases
from src.utils import ensure_dir


def main() -> int:
    settings = load_settings()
    rows = run_reference_cases(settings)
    reports_dir = ensure_dir(settings.reports_dir)
    json_path = reports_dir / "reference_cases_results.json"
    md_path = reports_dir / "reference_cases_results.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Контрольные кейсы",
        "",
        "> Это качественная проверка сценариев, а не тестовая выборка и не оценка accuracy.",
        f"> LLM-аудит: {'доступен' if settings.has_deepseek_key else 'НЕ НАСТРОЕН (нужен DEEPSEEK_API_KEY)'}.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']} — {row['final_category']}",
                "",
                f"- Ручная проверка: {'да' if row['human_review_required'] else 'нет'}",
                f"- ML: {row['ml_category']} (margin={row['ml_margin']})",
                f"- LLM: {row['llm_category'] or '—'} (status={row['llm_status']})",
                f"- Правила: {', '.join(row['rule_codes'])}",
                f"- С чего начать: {row['start_with']}",
                "",
                "**Фокус ручной проверки (задан в контрольном кейсе, не вывод модели):**",
                *(f"- {item}" for item in row["expected_review_focus"]),
                "",
            ]
        )
        if row["component_warnings"]:
            lines.append("**Предупреждения:**")
            lines.extend(f"- {item}" for item in row["component_warnings"])
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "llm_configured": settings.has_deepseek_key,
                "results": [
                    {
                        "case_id": row["case_id"],
                        "final_category": row["final_category"],
                        "human_review_required": row["human_review_required"],
                        "ml_category": row["ml_category"],
                        "llm_category": row["llm_category"],
                        "rule_codes": row["rule_codes"],
                    }
                    for row in rows
                ],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
