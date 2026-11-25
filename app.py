"""Streamlit UI «Support Case AI»: ручной анализ, контрольные кейсы, качество.

Приложение не подключается к CRM и не выполняет внешних действий.
"""
from __future__ import annotations

import streamlit as st

from src.config import load_settings
from src.ml import artifacts
from src.pipeline.normalize import parse_chat_text
from src.pipeline.orchestrator import AnalysisPipeline
from src.reference_cases import load_reference_cases, reference_to_input
from src.schemas import (
    CATEGORY_TITLES_RU,
    PROCESS_TYPE_TITLES_RU,
    AnalysisResult,
    CaseInput,
    Category,
    ComponentStatus,
    ProcessType,
)


st.set_page_config(page_title="Support Case AI", page_icon="🗂️", layout="wide")

try:
    SETTINGS = load_settings()
except Exception as exc:  # noqa: BLE001 - конфигурация может быть некорректной
    st.error(f"Ошибка конфигурации: {exc}")
    st.stop()

PROCESS_LABELS = {
    f"{title} ({process.value})": process
    for process, title in PROCESS_TYPE_TITLES_RU.items()
}
STATUS_TITLES = {
    ComponentStatus.OK: "успешно",
    ComponentStatus.NOT_CONFIGURED: "не настроен",
    ComponentStatus.UNAVAILABLE: "недоступен",
    ComponentStatus.ERROR: "ошибка",
}


@st.cache_resource(show_spinner=False)
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(SETTINGS)


def get_ml_state(pipeline: AnalysisPipeline) -> tuple[bool, str | None]:
    try:
        _ = pipeline.ml.metadata
        return True, None
    except Exception as exc:  # noqa: BLE001 - показываем причину в UI
        return False, str(exc)


def render_status_banners() -> None:
    if not SETTINGS.has_deepseek_key:
        st.warning(
            "DEEPSEEK_API_KEY не задан. Независимый LLM-аудит недоступен; "
            "любой результат потребует ручной проверки. Это не ошибка интерфейса."
        )
    pipeline = get_pipeline()
    ready, error = get_ml_state(pipeline)
    if not ready:
        st.info(
            f"ML-модель не загружена: {error}\n\n"
            "Обучите её командами:\n"
            "1. `python scripts/generate_dataset.py --engine offline`\n"
            "2. `python scripts/train_model.py`"
        )


def render_category(result: AnalysisResult) -> None:
    category = result.merge.final_category
    if category == Category.PRIORITY_1:
        st.error(f"### {CATEGORY_TITLES_RU[category]}")
    elif category == Category.REQUIRES_INTERVENTION:
        st.warning(f"### {CATEGORY_TITLES_RU[category]}")
    else:
        st.success(f"### {CATEGORY_TITLES_RU[category]}")
    review = "да" if result.merge.human_review_required else "нет"
    st.write(
        f"**Ручная проверка:** {review}  \n"
        f"**Достаточность подтверждений:** "
        f"{'достаточно' if result.merge.evidence_sufficient else 'недостаточно'}"
    )


def render_components(result: AnalysisResult) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("ML-компонент")
        st.write(f"Статус: **{STATUS_TITLES[result.ml.status]}**")
        prediction = result.ml.prediction
        if prediction is None:
            st.write(result.ml.error or "Предсказание отсутствует.")
        else:
            st.write(f"Категория: **{CATEGORY_TITLES_RU[prediction.category]}**")
            st.write(f"Margin: **{prediction.margin:.3f}**")
            for category in (
                Category.PRIORITY_1,
                Category.REQUIRES_INTERVENTION,
                Category.NO_INTERVENTION,
            ):
                value = prediction.probability(category)
                st.progress(min(1.0, max(0.0, value)), text=f"{category.value}: {value:.3f}")
            if prediction.low_margin:
                st.warning("Низкий margin: нужна ручная проверка.")
    with right:
        st.subheader("LLM-аудит (DeepSeek)")
        st.write(f"Статус: **{STATUS_TITLES[result.llm.status]}**")
        audit = result.llm.audit
        if audit is None:
            st.write(result.llm.error or "Аудит отсутствует.")
        else:
            st.write(f"Категория: **{CATEGORY_TITLES_RU[audit.category]}**")
            st.write(f"Ручная проверка по мнению LLM: {'да' if audit.needs_human_review else 'нет'}")
            st.write(audit.rationale)
        for warning in result.llm.warnings:
            st.info(warning)


def render_details(result: AnalysisResult) -> None:
    summary = result.summary
    st.subheader("Сводка")
    st.write(f"**ID / процесс:** {summary.case_id} / {summary.process_type_title}")
    st.write(f"**С чего начать:** {summary.start_with}")
    st.write("**Почему такой приоритет:**")
    for reason in summary.priority_rationale:
        st.write(f"- {reason}")

    if summary.known_facts:
        st.write("**Известные факты с источниками:**")
        st.dataframe(
            [
                {
                    "Факт": fact.fact,
                    "Источник": fact.source_type.value,
                    "Ссылка": fact.source_reference,
                    "Цитата": fact.source_quote or "",
                }
                for fact in summary.known_facts
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("**Известные факты:** LLM-аудит недоступен, факты не подтверждены.")

    if summary.contradictions:
        st.write("**Противоречия:**")
        for item in summary.contradictions:
            st.write(f"- {item.description}")
            if item.sources:
                st.caption("Источники: " + "; ".join(item.sources))
            if item.resolution_check:
                st.caption(f"Как проверить: {item.resolution_check}")
    else:
        st.write("**Противоречия:** не зафиксированы.")

    for title, values in (
        ("Недостающая информация", summary.missing_information),
        ("Что проверить", summary.checks),
        ("Следующие действия", summary.recommended_actions),
    ):
        st.write(f"**{title}:**")
        if values:
            for item in values:
                st.write(f"- {item}")
        else:
            st.write("- нет данных")

    if summary.component_warnings:
        with st.expander("Технические предупреждения", expanded=False):
            for warning in summary.component_warnings:
                st.write(f"- {warning}")

    with st.expander("Технический JSON (без секретов)", expanded=False):
        st.json(result.technical_json())


def run_pipeline(case_input: CaseInput, *, spinner_text: str) -> AnalysisResult | None:
    pipeline = get_pipeline()
    try:
        with st.spinner(spinner_text):
            return pipeline.analyze(case_input)
    except Exception as exc:  # noqa: BLE001 - компонент мог упасть, не изображаем успех
        st.error(f"Анализ не выполнен: {exc}")
        return None


def manual_tab() -> None:
    st.header("Ручной анализ")
    st.caption(
        "Вставьте обычный текст CRM и чата. JSON не требуется. "
        "Приложение не подключается к CRM и ничего не отправляет наружу, кроме запроса к DeepSeek при наличии ключа."
    )
    case_id = st.text_input("ID кейса (необязательно)", value="", max_chars=120)
    process_label = st.selectbox("Тип процесса", list(PROCESS_LABELS))
    crm_text = st.text_area("Данные из CRM", height=170, max_chars=50_000)
    chat_text = st.text_area(
        "Сообщения из рабочих чатов",
        height=220,
        max_chars=50_000,
        placeholder="Иван (Руководитель): Документы получил\nСотрудник: Отправил вчера",
    )

    if st.button("Проанализировать", type="primary"):
        chat_messages, parse_warnings = parse_chat_text(chat_text)
        try:
            case_input = CaseInput(
                case_id=case_id.strip() or None,
                process_type=PROCESS_LABELS[process_label],
                crm_text=crm_text,
                chat_messages=chat_messages,
            )
        except Exception as exc:  # noqa: BLE001 - показываем ошибку ввода
            st.session_state["manual_state"] = None
            st.error(f"Проверьте ввод: {exc}")
            return
        result = run_pipeline(case_input, spinner_text="ML и LLM анализируют кейс…")
        if result is not None:
            st.session_state["manual_state"] = {
                "result": result,
                "parse_warnings": parse_warnings,
            }

    state = st.session_state.get("manual_state")
    if state:
        if state["parse_warnings"]:
            st.warning("Замечания к разбору чата:")
            for warning in state["parse_warnings"]:
                st.write(f"- {warning}")
        result = state["result"]
        render_category(result)
        render_components(result)
        render_details(result)


def reference_tab() -> None:
    st.header("Контрольные кейсы")
    st.warning(
        "Это контрольные примеры из тестового задания, а не независимая тестовая "
        "выборка и не оценка accuracy. Они не входят в обучение."
    )
    try:
        references = load_reference_cases(SETTINGS.reference_cases_path)
    except Exception as exc:  # noqa: BLE001 - файл мог отсутствовать/быть повреждён
        st.error(f"Не удалось загрузить контрольные кейсы: {exc}")
        return
    labels = {f"{case.case_id} — {PROCESS_TYPE_TITLES_RU[case.process_type]}": case for case in references}
    selected_label = st.selectbox("Выберите кейс", list(labels))
    reference = labels[selected_label]

    st.write("**CRM:**")
    st.code(reference.crm_text, language=None)
    st.write("**Сообщения чата:**")
    for message in reference.chat_messages:
        role = f" ({message.role})" if message.role else ""
        st.write(f"- **{message.author}{role}:** {message.text}")

    if st.button("Запустить пайплайн на контрольном кейсе", type="primary", key="run_reference"):
        result = run_pipeline(
            reference_to_input(reference),
            spinner_text=f"Анализ контрольного кейса {reference.case_id}…",
        )
        if result is not None:
            st.session_state.setdefault("reference_results", {})[reference.case_id] = result

    results = st.session_state.get("reference_results", {})
    result = results.get(reference.case_id)
    if result:
        render_category(result)
        render_components(result)
        render_details(result)

    st.subheader("Ожидаемые смысловые акценты (не эталонная метка)")
    for item in reference.expected_review_focus:
        st.write(f"- {item}")
    st.caption(
        "Расхождение пайплайна с ожидаемыми акцентами отображается для анализа. "
        "Подгонять модель под четыре примера нельзя."
    )


def quality_tab() -> None:
    st.header("Качество модели")
    st.warning(
        "Метрики получены на синтетическом датасете и не подтверждают качество "
        "на реальных кейсах. Production readiness не заявляется."
    )
    try:
        metadata = artifacts.load_metadata(SETTINGS.artifacts_dir)
        metrics_payload = artifacts.load_metrics(SETTINGS.artifacts_dir)
    except FileNotFoundError:
        st.info(
            "Артефакты модели не найдены. Сначала выполните:\n\n"
            "```\npython scripts/generate_dataset.py --engine offline\n"
            "python scripts/train_model.py\n```\n\n"
            "Датасет при открытии приложения автоматически не генерируется."
        )
        return
    except Exception as exc:  # noqa: BLE001 - показываем причину
        st.error(f"Не удалось загрузить метрики: {exc}")
        return

    test = metadata.test_metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", f"{test.accuracy:.3f}")
    col2.metric("Test macro-F1", f"{test.macro_f1:.3f}")
    col3.metric("Test weighted-F1", f"{test.weighted_f1:.3f}")
    col4.metric("Размер test", test.sample_size)

    st.write(
        f"Embedding: `{metadata.embedding_model}`, классификатор: `{metadata.classifier}`, "
        f"C={metadata.selected_c:g}, low margin={metadata.low_margin_threshold:g}"
    )
    st.write(
        f"Датасет: `{metadata.dataset_engine}`, размер {metadata.dataset_size}, "
        f"split {metadata.split_sizes}"
    )
    if metadata.dataset_engine != "deepseek":
        st.warning(
            "Модель обучена на offline-шаблонном корпусе (DeepSeek недоступен). "
            "Это проверка пайплайна, а не свидетельство качества."
        )
    for warning in metadata.warnings:
        st.info(warning)

    st.subheader("Per-class метрики (test)")
    rows = []
    for label, values in test.per_class.items():
        rows.append(
            {
                "Класс": label,
                "Precision": round(values["precision"], 4),
                "Recall": round(values["recall"], 4),
                "F1": round(values["f1"], 4),
                "Support": int(values["support"]),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    st.subheader("Confusion matrix (test)")
    matrix_rows = []
    for label, values in zip(test.labels, test.confusion_matrix):
        matrix_rows.append({"Истина \\ Прогноз": label, **dict(zip(test.labels, values))})
    st.dataframe(matrix_rows, hide_index=True, width="stretch")

    st.subheader("Сравнение с TF-IDF baseline")
    comparison = [
        {
            "Модель": "E5 + LogisticRegression",
            "Validation macro-F1": round(metadata.validation_metrics.macro_f1, 4),
            "Test macro-F1": round(metadata.test_metrics.macro_f1, 4),
        },
        {
            "Модель": "TF-IDF + LogisticRegression",
            "Validation macro-F1": round(metadata.baseline_validation_metrics.macro_f1, 4),
            "Test macro-F1": round(metadata.baseline_test_metrics.macro_f1, 4),
        },
    ]
    st.dataframe(comparison, hide_index=True, width="stretch")

    st.subheader("Калибровка (test)")
    st.json(metadata.test_metrics.calibration)
    with st.expander("Полный metrics.json", expanded=False):
        st.json(metrics_payload)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Support Case AI")
        st.caption("Учебный прототип: ML-черновик + независимый LLM-аудит + правила.")
        summary = SETTINGS.public_summary()
        st.write(f"DeepSeek: {'настроен' if summary['deepseek_configured'] else 'не настроен'}")
        st.write(f"Модель LLM: `{summary['deepseek_model']}`")
        st.write(f"Embedding: `{summary['embedding_model']}`")
        st.write(f"Low margin: `{summary['ml_low_margin_threshold']}`")
        pipeline = get_pipeline()
        ready, error = get_ml_state(pipeline)
        st.write(f"ML-модель: {'готова' if ready else 'не загружена'}")
        if not ready and error:
            st.caption(error)
        st.divider()
        st.caption(
            "Никаких подключений к CRM и внешних действий. "
            "Итог является черновиком для сотрудника, а не автоматическим решением."
        )


def main() -> None:
    render_sidebar()
    render_status_banners()
    st.title("Support Case AI")
    st.caption(
        "Ручной ввод → нормализация → ML → независимый LLM-аудит → детерминированный merge → сводка."
    )
    manual, reference, quality = st.tabs(
        ["Ручной анализ", "Контрольные кейсы", "Качество модели"]
    )
    with manual:
        manual_tab()
    with reference:
        reference_tab()
    with quality:
        quality_tab()


main()



