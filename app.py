"""
app.py
------
# * Streamlit entry point — run with: streamlit run app.py

# ? Two-phase pipeline to handle Streamlit's rerun model:
# ?   Phase 1 (Analyse clicked) -> Intent + clarifying questions -> save to session
# ?   Phase 2 (Generate Dashboard clicked) -> SQL + DB + chart -> save to session
# ?   Phase done -> render dashboard from session state
"""

import html
import traceback

import streamlit as st

from config import (
    APP_TITLE,
    CODER_MODEL,
    FAST_MODEL,
    PG_TABLE_NAME,
    USE_MOCK_DATA,
    validate_config,
)
from utils.benchmark import QueryBenchmark
from utils.logger import get_logger

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

log = get_logger(__name__)


def _render_hero(title: str, description: str, chips: list[tuple[str, str]], badge: str):
    chip_html = "".join(
        f"""
        <div class="ld-chip">
            <small>{html.escape(label)}</small>
            <strong>{html.escape(value)}</strong>
        </div>
        """
        for label, value in chips
        if value
    )
    st.markdown(
        f"""
        <section class="ld-hero">
            <div class="ld-badge">{html.escape(badge)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
            <div class="ld-chip-row">{chip_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_status_row(title: str, status_text: str, detail: str, tone: str) -> str:
    return f"""
    <div class="ld-status-row">
        <span class="ld-status-dot {tone}"></span>
        <div>
            <span class="ld-status-title">{html.escape(title)} &middot; {html.escape(status_text)}</span>
            <span class="ld-status-detail">{html.escape(detail)}</span>
        </div>
    </div>
    """


def _note_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="ld-note-card">
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Starting up...")
def _startup() -> dict:
    from charts.theme import apply_theme
    from core.dictionary import get_dictionary
    from database.connection import get_connection
    from models.ollama_client import get_client as get_ollama

    apply_theme()
    status = {}
    status["config_warnings"] = validate_config()

    dictionary = get_dictionary()
    status["dictionary_ready"] = dictionary.is_ready

    db = get_connection()
    status["db_connected"] = db.is_ready
    status["db_mock_mode"] = USE_MOCK_DATA

    ollama = get_ollama()
    summary = ollama.status_summary()
    status.update(
        {
            "ollama_running": summary["ollama_running"],
            "coder_available": summary["coder_available"],
            "fast_available": summary["fast_available"],
        }
    )

    try:
        if not USE_MOCK_DATA and status.get("db_connected"):
            status["db_row_count"] = int(
                db.execute(f"SELECT COUNT(*) FROM {PG_TABLE_NAME}").iloc[0, 0]
            )
        else:
            status["db_row_count"] = 89255
    except Exception:
        status["db_row_count"] = 0

    log.info(f"[app] Startup complete | {status}")
    return status


def _render_sidebar(session, status: dict):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="ld-sidebar-card">
                <strong>{html.escape(APP_TITLE)}</strong>
                <span>Live portfolio analytics with AI-assisted SQL, filters, charting, and export controls.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status.get("db_mock_mode"):
            db_state = ("Database", "Mock Mode", "89,255 synthetic rows available", "ok")
        elif status.get("db_connected"):
            row_count = status.get("db_row_count", 0)
            db_state = ("Database", "Connected", f"{int(row_count):,} rows in {PG_TABLE_NAME}", "ok")
        else:
            db_state = ("Database", "Unavailable", "Check Docker and .env PG_* values", "bad")

        if status.get("ollama_running"):
            model_detail = (
                f"Coder: {CODER_MODEL} | Fast: {FAST_MODEL}"
                if status.get("coder_available") or status.get("fast_available")
                else "Models configured but not reachable"
            )
            model_state = ("Models", "Running", model_detail, "ok")
        else:
            model_state = ("Models", "Offline", "Run `ollama serve` in terminal", "warn")

        if status.get("dictionary_ready"):
            context_state = ("Context", "Loaded", "Prompt context and retrieval are ready", "ok")
        else:
            context_state = ("Context", "Not Ready", "Context index could not be loaded", "warn")

        st.markdown(
            f"""
            <div class="ld-sidebar-card">
                <strong>System Status</strong>
                <span>Health of the data, model, and context stack.</span>
                {_sidebar_status_row(*db_state)}
                {_sidebar_status_row(*model_state)}
                {_sidebar_status_row(*context_state)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        for warning in status.get("config_warnings", []):
            st.warning(f"Warning: {warning}")

        st.markdown(
            """
            <div class="ld-sidebar-card">
                <strong>Controls</strong>
                <span>Reset the current flow, clear the session, or reveal timings while tuning prompts.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New Query", use_container_width=True):
                _reset_pipeline(session)
                st.rerun()
        with c2:
            if st.button("Clear All", use_container_width=True):
                session.full_reset()
                _reset_pipeline(session)
                st.rerun()

        session.set("show_debug", st.checkbox("Show Timings", value=False))

        if USE_MOCK_DATA:
            _note_card("Demo Mode", "You are exploring the synthetic 89K-loan sample dataset.")


def _reset_pipeline(session):
    """# * Clear all pipeline state — called on New Query."""
    session.set("pipeline_phase", None)
    session.set("pending_intent", None)
    session.set("pending_clarifier", None)
    session.set("pending_slicers", {})
    session.reset_query_state()


def _get_phase(session) -> str:
    return session.get("pipeline_phase", None)


def _run_phase1(question: str, session):
    """
    # * Run intent analysis and generate clarifying questions.
    # * Saves results to session then reruns to show filter widgets.
    """
    from models.analyzer import get_analyzer
    from models.clarifier import get_clarifier

    qb = QueryBenchmark(question)

    with st.status("Understanding your question...", expanded=True) as status_box:
        try:
            qb.start("intent_analysis")
            intent = get_analyzer().analyze(question)
            qb.end("intent_analysis")

            if intent.failed:
                status_box.update(label=f"Failed: {intent.error}", state="error")
                st.error(f"Could not understand: {intent.error}")
                return

            session.set_intent(intent.to_dict())
            session.set(
                "pending_intent",
                {
                    "success": True,
                    "raw_question": intent.raw_question,
                    "intent_summary": intent.intent_summary,
                    "metric": intent.metric,
                    "metric_key": intent.metric_key,
                    "columns_needed": intent.columns_needed,
                    "slicer_candidates": intent.slicer_candidates,
                    "aggregation": intent.aggregation,
                    "group_by": intent.group_by,
                    "time_involved": intent.time_involved,
                    "granularity": intent.granularity,
                    "sql_pattern_hint": intent.sql_pattern_hint,
                    "confidence": intent.confidence,
                },
            )
            status_box.update(
                label=f"Metric: {intent.metric} (confidence: {intent.confidence})",
                state="complete",
            )
        except Exception as exc:
            qb.end("intent_analysis", error=str(exc))
            st.error(f"Analysis error: {exc}")
            log.error(traceback.format_exc())
            return

    with st.status("Generating filter options...", expanded=True) as status_box:
        try:
            qb.start("clarifying_questions")
            clarifier_result = get_clarifier().generate(intent)
            qb.end("clarifying_questions")
            session.set("pending_clarifier", clarifier_result)
            status_box.update(label="Filters ready", state="complete")
        except Exception as exc:
            qb.end("clarifying_questions", error=str(exc))
            session.set("pending_clarifier", None)
            status_box.update(label="No filters found — using all data", state="complete")

    qb.report()
    session.record_benchmark(qb)
    session.set("pipeline_phase", "clarifying")
    st.rerun()


def _render_clarifying_ui(session) -> bool:
    """
    # * Render filter widgets and return True when Generate is clicked.
    # * Saves widget answers to session state on every rerun.
    """
    from models.clarifier import get_clarifier

    intent_dict = session.get("pending_intent", {})
    clarifier_result = session.get("pending_clarifier")
    metric = intent_dict.get("metric", "your query")

    _note_card(
        f"Metric in focus: {metric}",
        "Confirm the filters below so the dashboard is generated against the right segment.",
    )

    if clarifier_result and clarifier_result.success and clarifier_result.questions:
        st.markdown("#### Refine the scope")
        answers = get_clarifier().collect_answers(clarifier_result)
        session.set("pending_slicers", answers)
        st.divider()

    generate_clicked = st.button(
        "Generate Dashboard",
        type="primary",
        use_container_width=True,
        key="gen_btn",
    )

    if not generate_clicked:
        st.info("Pick your filters, then generate the dashboard.")

    return generate_clicked


def _run_phase2(session) -> bool:
    """
    # * Generate SQL, fetch data, decide chart.
    # * Reads intent and slicer answers from session state.
    """
    from core.filters import FilterManager, FilterSelection
    from database.client import get_client as get_db
    from models.analyzer import IntentResult
    from models.chart_decider import get_chart_decider
    from models.sql_generator import get_sql_generator

    intent_dict = session.get("pending_intent", {})
    if not intent_dict:
        st.error("Intent data was lost. Please start a new query.")
        return False

    intent = IntentResult(**intent_dict)
    slicer_answers = session.get("pending_slicers", {})
    session.set_slicers(slicer_answers)

    qb = QueryBenchmark(intent.raw_question)

    with st.status("Generating SQL query...", expanded=True) as status_box:
        try:
            qb.start("sql_generation")
            sql_result = get_sql_generator().generate(intent, slicer_answers)
            qb.end("sql_generation")

            if sql_result.failed:
                status_box.update(label=f"SQL failed: {sql_result.error}", state="error")
                st.error(f"SQL error: {sql_result.error}")
                return False

            session.set_sql(sql_result.sql)
            status_box.update(label="SQL generated", state="complete")
        except Exception as exc:
            qb.end("sql_generation", error=str(exc))
            st.error(f"SQL error: {exc}")
            log.error(traceback.format_exc())
            return False

    with st.status("Fetching data from the database...", expanded=True) as status_box:
        try:
            qb.start("db_fetch")
            db_result = get_db().run_query(sql_result.sql)
            qb.end("db_fetch")

            if db_result.failed:
                status_box.update(label=f"Database error: {db_result.error}", state="error")
                st.error(f"Database error: {db_result.error}")
                return False

            if db_result.is_empty:
                status_box.update(label="No data returned", state="complete")
                st.warning("Query returned no rows. Try different filters.")
                return False

            df = db_result.dataframe
            status_box.update(
                label=f"{db_result.row_count:,} rows loaded from {db_result.source}",
                state="complete",
            )
        except Exception as exc:
            qb.end("db_fetch", error=str(exc))
            st.error(f"Fetch error: {exc}")
            log.error(traceback.format_exc())
            return False

    fm = FilterManager()
    selections = [
        FilterSelection(field=field, value=value)
        for field, value in slicer_answers.items()
        if value and str(value).lower() not in ("all", "")
    ]
    df = fm.apply_filters(df, selections)
    filter_summary = fm.get_summary(selections)
    session.store_dataframe(df)

    with st.status("Choosing the best chart type...", expanded=True) as status_box:
        try:
            qb.start("chart_decision")
            chart_cfg = get_chart_decider().decide(df, intent, filter_summary)
            qb.end("chart_decision")
            cfg_dict = chart_cfg.to_dict()
            session.set_chart_config(cfg_dict)
            status_box.update(
                label=f"{chart_cfg.chart_type} selected | x={chart_cfg.x_col} | y={chart_cfg.y_col}",
                state="complete",
            )
        except Exception as exc:
            qb.end("chart_decision", error=str(exc))
            cfg_dict = {}
            status_box.update(label="Using the default chart", state="complete")
            log.error(traceback.format_exc())

    session.set(
        "dashboard_charts",
        [
            {
                "chart_config": session.get_current_chart_config(),
                "filter_summary": filter_summary,
                "sql": sql_result.sql,
                "metric": intent.metric,
                "question": intent.raw_question,
            }
        ],
    )

    qb.report()
    session.record_benchmark(qb)
    session.add_chat_message("assistant", f"Here is your {intent.metric} dashboard.")
    return True


def _render_dashboard(session):
    """# * Render the full dashboard — KPIs + slicer bar + tabs."""
    from ui.chat import render_chat_history
    from ui.dashboard_tabs import render_dashboard_tabs
    from ui.kpis import render_kpis
    from ui.sidebar import render_slicer_bar

    df = session.get_dataframe()
    charts = session.get("dashboard_charts", [])
    cfg_raw = session.get_current_chart_config()

    if df is None or df.empty or not charts:
        return

    data = charts[0]
    metric = data.get("metric", "")
    filter_summary = data.get("filter_summary", "All Data")
    sql = data.get("sql", "")

    _render_hero(
        title=metric or "Dashboard",
        description="Interactive results with KPI cards, slicers, chart exploration, raw data, SQL traceability, and export tools.",
        chips=[
            ("Visible Rows", f"{len(df):,}"),
            ("Filters", filter_summary if filter_summary and filter_summary != "All Data" else "All data"),
            ("Chart Type", (cfg_raw or {}).get("chart_type", "auto")),
        ],
        badge="Dashboard Ready",
    )

    df, filter_summary = render_slicer_bar(df, max_slicers=4, key_prefix="dash")
    render_kpis(df, max_kpis=4)

    if filter_summary and filter_summary != "All Data":
        st.caption(f"Active filters: {filter_summary}")

    render_dashboard_tabs(
        df=df,
        chart_config=cfg_raw or {},
        metric=metric,
        filter_summary=filter_summary,
        sql=sql,
        question=data.get("question", metric),
    )

    render_chat_history(session)


def _render_debug(session):
    if not session.get("show_debug", False):
        return
    last = session.get_last_benchmark()
    if not last:
        return

    with st.expander("Query Timings", expanded=False):
        steps = last.get("steps", {})
        total = last.get("total_ms", 0)
        if steps:
            n_cols = min(len(steps) + 1, 6)
            cols = st.columns(n_cols)
            for i, (step, ms) in enumerate(list(steps.items())[: n_cols - 1]):
                cols[i].metric(step.replace("_", " ").title(), f"{ms}ms")
            cols[-1].metric("Total", f"{total}ms")


def _render_home():
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """
        <div class="ld-card">
            <h3>What You Can Ask</h3>
            <ul>
                <li>Bounce rate by branch</li>
                <li>Resolution % by TL</li>
                <li>NPA by region</li>
                <li>Coverage analysis</li>
                <li>Bucket movement</li>
                <li>FE scorecard</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        """
        <div class="ld-card">
            <h3>How The Flow Works</h3>
            <ul>
                <li>Ask a question in plain English</li>
                <li>The model identifies the metric and needed fields</li>
                <li>You confirm the relevant filters</li>
                <li>SQL runs against the database</li>
                <li>The chart and insights are generated automatically</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        """
        <div class="ld-card">
            <h3>Popular Metrics</h3>
            <ul>
                <li>Bounce % and Bounce Count</li>
                <li>Resolution %</li>
                <li>Coverage %</li>
                <li>Visit Intensity</li>
                <li>Portfolio Outstanding</li>
                <li>Bucket Distribution</li>
                <li>NPA Count</li>
                <li>FE Scorecard</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    from charts.theme import inject_app_css
    from core.session import get_session

    status = _startup()
    inject_app_css()

    session = get_session()
    session.cleanup_all_stale_files()

    _render_sidebar(session, status)

    data_source_label = (
        "Mock dataset · 89,255 synthetic loans"
        if USE_MOCK_DATA
        else f"Live table · {int(status.get('db_row_count', 0)):,} rows"
    )
    _render_hero(
        title=APP_TITLE,
        description="Ask portfolio questions in plain English and get structured filters, generated SQL, interactive visuals, and export-ready outputs from the same workflow.",
        chips=[
            ("Data Source", data_source_label),
            ("Coder Model", CODER_MODEL if status.get("coder_available") else "Unavailable"),
            ("Fast Model", FAST_MODEL if status.get("fast_available") else "Unavailable"),
            ("Context", "Ready" if status.get("dictionary_ready") else "Unavailable"),
        ],
        badge="Collection Intelligence Studio",
    )
    _note_card(
        "Ask a question",
        "Describe the metric, group-by, or segment you want. The app will understand intent, ask for missing filters, then build the dashboard.",
    )

    with st.expander("Example questions — click to load", expanded=False):
        examples = [
            "Show bounce rate by branch",
            "Which team leader has the best coverage?",
            "Show NPA distribution by region",
            "Show bucket movement matrix",
            "Which branches have the highest resolution rate?",
            "Show DPD distribution",
            "Compare bounce rate by portfolio",
            "Show FE scorecard",
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
                _reset_pipeline(session)
                st.session_state["_q"] = example
                st.rerun()

    question = st.text_input(
        "question",
        placeholder="e.g. Show bounce rate by branch for Rajasthan",
        key="_q",
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 5])
    ask_clicked = ask_col.button(
        "Analyse",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

    phase = _get_phase(session)

    if ask_clicked and question.strip():
        _reset_pipeline(session)
        session.set_query(question.strip())
        session.add_chat_message("user", question.strip())

        st.divider()
        st.markdown("### Processing")
        _run_phase1(question.strip(), session)

    elif phase == "clarifying":
        st.divider()
        generate_clicked = _render_clarifying_ui(session)

        if generate_clicked:
            st.divider()
            st.markdown("### Building dashboard")
            success = _run_phase2(session)

            if success:
                session.set("pipeline_phase", "done")
                st.rerun()
            else:
                session.set("pipeline_phase", "clarifying")

    elif phase == "done":
        df = session.get_dataframe()
        charts = session.get("dashboard_charts", [])

        if df is not None and not df.empty and charts:
            st.divider()
            _render_dashboard(session)
            _render_debug(session)
        else:
            st.error("Dashboard data is missing. Please start a new query.")
            _reset_pipeline(session)

    else:
        _render_home()


if __name__ == "__main__":
    main()
