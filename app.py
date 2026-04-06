"""
app.py
------
# * Streamlit entry point — run with: streamlit run app.py

# ? Two-phase pipeline to handle Streamlit's rerun model:
# ?   Phase 1 (Analyse clicked) → Intent + Clarifying questions → save to session
# ?   Phase 2 (Generate Dashboard clicked) → SQL + DB + Chart → save to session
# ?   Phase done → render dashboard from session state
"""

import streamlit as st

st.set_page_config(
    page_title            = "Loan Collection Analytics",
    page_icon             = "📊",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

import traceback
from config import (
    APP_TITLE, CODER_MODEL, FAST_MODEL,
    USE_MOCK_DATA, validate_config, PG_TABLE_NAME,
)
from utils.logger    import get_logger
from utils.benchmark import QueryBenchmark

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * STARTUP — cached once per app lifetime
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🚀 Starting up...")
def _startup() -> dict:
    from charts.theme         import apply_theme
    from core.dictionary      import get_dictionary
    from database.connection  import get_connection
    from models.ollama_client import get_client as get_ollama

    apply_theme()
    status = {}
    status["config_warnings"] = validate_config()

    d = get_dictionary()
    status["dictionary_ready"] = d.is_ready

    db = get_connection()
    status["db_connected"] = db.is_ready
    status["db_mock_mode"] = USE_MOCK_DATA

    ollama = get_ollama()
    s = ollama.status_summary()
    status.update({
        "ollama_running":  s["ollama_running"],
        "coder_available": s["coder_available"],
        "fast_available":  s["fast_available"],
    })

    # * Row count — runs once at startup, cached
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


# ──────────────────────────────────────────────────────────────────────────────
# * SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def _render_sidebar(session, status: dict):
    with st.sidebar:
        st.markdown(f"## 📊 {APP_TITLE}")
        st.divider()

        st.markdown("### 🖥️ System Status")

        # * Database
        if status.get("db_mock_mode"):
            st.success("🗄️ Database: **Mock Mode**")
            st.caption("📋 89,255 rows (synthetic)")
        elif status.get("db_connected"):
            row_count = status.get("db_row_count", 0)
            st.success("🗄️ Database: Connected")
            st.caption(f"📋 {int(row_count):,} rows in {PG_TABLE_NAME}")
        else:
            st.error("🗄️ Database: Not Connected")
            st.caption("Check Docker and .env PG_* values.")

        # * Ollama
        if status.get("ollama_running"):
            st.success("🤖 Ollama: Running")
            c1, c2 = st.columns(2)
            c1.success("✅ Coder" if status.get("coder_available") else "❌ Coder")
            c2.success("✅ Fast"  if status.get("fast_available")  else "❌ Fast")
            st.caption(f"Coder: `{CODER_MODEL}`")
            st.caption(f"Fast:  `{FAST_MODEL}`")
        else:
            st.error("🤖 Ollama: Not running")
            st.caption("Run `ollama serve` in terminal.")

        # * Context / ChromaDB
        if status.get("dictionary_ready"):
            st.success("📚 Context: Loaded")
        else:
            st.warning("📚 Context: Not ready")

        # * Config warnings
        for w in status.get("config_warnings", []):
            st.warning(f"⚠️ {w}")

        st.divider()

        # * Controls
        st.markdown("### ⚙️ Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 New Query", use_container_width=True):
                _reset_pipeline(session)
                st.rerun()
        with c2:
            if st.button("🗑️ Clear All", use_container_width=True):
                session.full_reset()
                _reset_pipeline(session)
                st.rerun()

        session.set("show_debug", st.checkbox("🔬 Show Timings", value=False))

        if USE_MOCK_DATA:
            st.divider()
            st.info("🎭 **Demo Mode** — 89K synthetic loans.")


# ──────────────────────────────────────────────────────────────────────────────
# * PIPELINE STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _reset_pipeline(session):
    """# * Clear all pipeline state — called on New Query."""
    session.set("pipeline_phase",    None)
    session.set("pending_intent",    None)
    session.set("pending_clarifier", None)
    session.set("pending_slicers",   {})
    session.reset_query_state()


def _get_phase(session) -> str:
    return session.get("pipeline_phase", None)


# ──────────────────────────────────────────────────────────────────────────────
# * PHASE 1 — Intent + Clarifying questions
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase1(question: str, session):
    """
    # * Run intent analysis and generate clarifying questions.
    # * Saves results to session then reruns to show filter widgets.
    """
    from models.analyzer  import get_analyzer
    from models.clarifier import get_clarifier

    qb = QueryBenchmark(question)

    # * Step 1 — Intent
    with st.status("🧠 Understanding your question...", expanded=True) as s:
        try:
            qb.start("intent_analysis")
            intent = get_analyzer().analyze(question)
            qb.end("intent_analysis")

            if intent.failed:
                s.update(label=f"❌ {intent.error}", state="error")
                st.error(f"Could not understand: {intent.error}")
                return

            session.set_intent(intent.to_dict())
            session.set("pending_intent", {
                "success":           True,
                "raw_question":      intent.raw_question,
                "intent_summary":    intent.intent_summary,
                "metric":            intent.metric,
                "metric_key":        intent.metric_key,
                "columns_needed":    intent.columns_needed,
                "slicer_candidates": intent.slicer_candidates,
                "aggregation":       intent.aggregation,
                "group_by":          intent.group_by,
                "time_involved":     intent.time_involved,
                "granularity":       intent.granularity,
                "sql_pattern_hint":  intent.sql_pattern_hint,
                "confidence":        intent.confidence,
            })
            s.update(
                label = f"✅ Metric: **{intent.metric}** (Confidence: {intent.confidence})",
                state = "complete",
            )
        except Exception as e:
            qb.end("intent_analysis", error=str(e))
            st.error(f"Analysis error: {e}")
            log.error(traceback.format_exc())
            return

    # * Step 2 — Clarifying questions
    with st.status("❓ Generating filter options...", expanded=True) as s:
        try:
            qb.start("clarifying_questions")
            clarifier_result = get_clarifier().generate(intent)
            qb.end("clarifying_questions")
            session.set("pending_clarifier", clarifier_result)
            s.update(label="✅ Filters ready", state="complete")
        except Exception as e:
            qb.end("clarifying_questions", error=str(e))
            session.set("pending_clarifier", None)
            s.update(label="⚠️ No filters — will use all data", state="complete")

    qb.report()
    session.record_benchmark(qb)
    session.set("pipeline_phase", "clarifying")
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# * CLARIFYING UI
# ──────────────────────────────────────────────────────────────────────────────

def _render_clarifying_ui(session) -> bool:
    """
    # * Render filter widgets and return True when Generate is clicked.
    # * Saves widget answers to session state on every rerun.
    """
    from models.clarifier import get_clarifier

    intent_dict      = session.get("pending_intent", {})
    clarifier_result = session.get("pending_clarifier")
    metric           = intent_dict.get("metric", "your query")

    st.markdown(f"### 🧠 Metric: **{metric}**")
    st.divider()

    if clarifier_result and clarifier_result.success and clarifier_result.questions:
        st.markdown("#### 🔍 Help me narrow this down:")
        answers = get_clarifier().collect_answers(clarifier_result)
        # * Save answers on EVERY rerun so they're available in phase 2
        session.set("pending_slicers", answers)
        st.divider()

    generate_clicked = st.button(
        "🚀 Generate Dashboard",
        type                = "primary",
        use_container_width = True,
        key                 = "gen_btn",
    )

    if not generate_clicked:
        st.info("👆 Pick your filters then click **Generate Dashboard**.")

    return generate_clicked


# ──────────────────────────────────────────────────────────────────────────────
# * PHASE 2 — SQL + DB fetch + Chart
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase2(session) -> bool:
    """
    # * Generate SQL, fetch data, decide chart.
    # * Reads intent and slicer answers from session state.
    """
    from core.filters         import FilterManager, FilterSelection
    from database.client      import get_client as get_db
    from models.analyzer      import IntentResult
    from models.sql_generator import get_sql_generator
    from models.chart_decider import get_chart_decider

    intent_dict = session.get("pending_intent", {})
    if not intent_dict:
        st.error("Intent data lost — please start a new query.")
        return False

    intent = IntentResult(**intent_dict)

    # * Read saved slicer answers — DO NOT render widgets again
    slicer_answers = session.get("pending_slicers", {})
    session.set_slicers(slicer_answers)

    qb = QueryBenchmark(intent.raw_question)

    # * Step 3 — SQL generation
    with st.status("⚙️ Generating SQL query...", expanded=True) as s:
        try:
            qb.start("sql_generation")
            sql_result = get_sql_generator().generate(intent, slicer_answers)
            qb.end("sql_generation")

            if sql_result.failed:
                s.update(label=f"❌ SQL failed: {sql_result.error}", state="error")
                st.error(f"SQL error: {sql_result.error}")
                return False

            session.set_sql(sql_result.sql)
            s.update(label="✅ SQL generated", state="complete")

        except Exception as e:
            qb.end("sql_generation", error=str(e))
            st.error(f"SQL error: {e}")
            log.error(traceback.format_exc())
            return False

    # * Step 4 — Fetch data
    with st.status("🗄️ Fetching data from database...", expanded=True) as s:
        try:
            qb.start("db_fetch")
            db_result = get_db().run_query(sql_result.sql)
            qb.end("db_fetch")

            if db_result.failed:
                s.update(label=f"❌ DB error: {db_result.error}", state="error")
                st.error(f"Database error: {db_result.error}")
                return False

            if db_result.is_empty:
                s.update(label="⚠️ No data returned", state="complete")
                st.warning("Query returned no rows — try different filters.")
                return False

            df = db_result.dataframe
            s.update(
                label = f"✅ {db_result.row_count:,} rows | source: {db_result.source}",
                state = "complete",
            )

        except Exception as e:
            qb.end("db_fetch", error=str(e))
            st.error(f"Fetch error: {e}")
            log.error(traceback.format_exc())
            return False

    # * Step 5 — Apply post-fetch filters
    fm   = FilterManager()
    sels = [
        FilterSelection(field=k, value=v)
        for k, v in slicer_answers.items()
        if v and str(v).lower() not in ("all", "")
    ]
    df             = fm.apply_filters(df, sels)
    filter_summary = fm.get_summary(sels)
    session.store_dataframe(df)

    # * Step 6 — Chart decision
    with st.status("🎨 Choosing best chart type...", expanded=True) as s:
        try:
            qb.start("chart_decision")
            chart_cfg = get_chart_decider().decide(df, intent, filter_summary)
            qb.end("chart_decision")
            cfg_dict  = chart_cfg.to_dict()
            session.set_chart_config(cfg_dict)
            s.update(
                label = (
                    f"✅ {chart_cfg.chart_type} | "
                    f"x={chart_cfg.x_col} | y={chart_cfg.y_col}"
                ),
                state = "complete",
            )
        except Exception as e:
            qb.end("chart_decision", error=str(e))
            cfg_dict = {}
            s.update(label="⚠️ Using default chart", state="complete")
            log.error(traceback.format_exc())

    # * Save dashboard data
    session.set("dashboard_charts", [{
        "chart_config":   session.get_current_chart_config(),
        "filter_summary": filter_summary,
        "sql":            sql_result.sql,
        "metric":         intent.metric,
        "question":       intent.raw_question,
    }])

    qb.report()
    session.record_benchmark(qb)
    session.add_chat_message("assistant",
                             f"Here is your **{intent.metric}** dashboard.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# * DASHBOARD RENDER
# ──────────────────────────────────────────────────────────────────────────────

def _render_dashboard(session):
    """# * Render the full dashboard — KPIs + slicer bar + tabs."""
    from ui.kpis           import render_kpis
    from ui.sidebar        import render_slicer_bar
    from ui.dashboard_tabs import render_dashboard_tabs
    from ui.chat           import render_chat_history

    df      = session.get_dataframe()
    charts  = session.get("dashboard_charts", [])
    cfg_raw = session.get_current_chart_config()

    if df is None or df.empty or not charts:
        return

    data           = charts[0]
    metric         = data.get("metric", "")
    filter_summary = data.get("filter_summary", "All Data")
    sql            = data.get("sql", "")

    st.markdown(f"### 📊 {metric}")

    # * Post-result slicer bar at top
    st.divider()
    df, filter_summary = render_slicer_bar(df, max_slicers=4, key_prefix="dash")
    st.divider()

    # * KPI cards
    render_kpis(df, max_kpis=4)

    if filter_summary and filter_summary != "All Data":
        st.caption(f"Active filters: {filter_summary}")

    st.divider()

    # * Chart + data + SQL + export tabs
    render_dashboard_tabs(
        df             = df,
        chart_config   = cfg_raw or {},
        metric         = metric,
        filter_summary = filter_summary,
        sql            = sql,
    )

    st.divider()

    # * Query history at bottom
    render_chat_history(session)


# ──────────────────────────────────────────────────────────────────────────────
# * DEBUG TIMINGS
# ──────────────────────────────────────────────────────────────────────────────

def _render_debug(session):
    if not session.get("show_debug", False):
        return
    last = session.get_last_benchmark()
    if not last:
        return
    with st.expander("🔬 Query Timings", expanded=False):
        steps = last.get("steps", {})
        total = last.get("total_ms", 0)
        if steps:
            n    = min(len(steps) + 1, 6)
            cols = st.columns(n)
            for i, (step, ms) in enumerate(list(steps.items())[:n-1]):
                cols[i].metric(step.replace("_", " ").title(), f"{ms}ms")
            cols[-1].metric("Total", f"{total}ms")


# ──────────────────────────────────────────────────────────────────────────────
# * HOME SCREEN
# ──────────────────────────────────────────────────────────────────────────────

def _render_home():
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "#### 📋 What can I ask?\n"
        "- Bounce rate by branch\n"
        "- Resolution % by TL\n"
        "- NPA by region\n"
        "- Coverage analysis\n"
        "- Bucket movement\n"
        "- FE scorecard"
    )
    c2.markdown(
        "#### 🔧 How it works\n"
        "1. Type your question\n"
        "2. AI finds the metric\n"
        "3. Pick your filters\n"
        "4. SQL runs against DB\n"
        "5. Chart builds automatically"
    )
    c3.markdown(
        "#### 📊 Available Metrics\n"
        "- Bounce % (Tech / Non-Tech)\n"
        "- Resolution %\n"
        "- Coverage %\n"
        "- Visit Intensity\n"
        "- Portfolio Outstanding\n"
        "- Bucket Distribution\n"
        "- NPA Count\n"
        "- FE Scorecard"
    )


# ──────────────────────────────────────────────────────────────────────────────
# * MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    from core.session import get_session

    status  = _startup()
    session = get_session()
    session.cleanup_all_stale_files()

    _render_sidebar(session, status)

    # * Header
    st.title("📊 Loan Collection Analytics")
    st.caption("Ask any question about your portfolio in plain English.")

    if USE_MOCK_DATA:
        st.info("🎭 **Demo Mode** — 89,255 synthetic loan accounts.")

    st.divider()
    st.markdown("### 💬 Ask a question")

    # * Example questions
    with st.expander("💡 Example questions — click to load", expanded=False):
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
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                _reset_pipeline(session)
                st.session_state["_q"] = ex
                st.rerun()

    # * Question input
    question = st.text_input(
        "question",
        placeholder      = "e.g. Show bounce rate by branch for Rajasthan",
        key              = "_q",
        label_visibility = "collapsed",
    )

    ask_col, _ = st.columns([1, 5])
    ask_clicked = ask_col.button(
        "🔍 Analyse",
        type                = "primary",
        use_container_width = True,
        disabled            = not question.strip(),
    )

    # ── Pipeline state machine ─────────────────────────────────────────────
    phase = _get_phase(session)

    if ask_clicked and question.strip():
        _reset_pipeline(session)
        session.set_query(question.strip())
        session.add_chat_message("user", question.strip())

        st.divider()
        st.markdown("### ⚙️ Processing...")
        _run_phase1(question.strip(), session)
        # * _run_phase1 ends with st.rerun() so nothing below executes

    elif phase == "clarifying":
        st.divider()
        generate_clicked = _render_clarifying_ui(session)

        if generate_clicked:
            st.divider()
            st.markdown("### ⚙️ Building dashboard...")
            success = _run_phase2(session)

            if success:
                session.set("pipeline_phase", "done")
                st.rerun()
            else:
                # * Keep on clarifying so user can retry
                session.set("pipeline_phase", "clarifying")

    elif phase == "done":
        df     = session.get_dataframe()
        charts = session.get("dashboard_charts", [])

        if df is not None and not df.empty and charts:
            st.divider()
            _render_dashboard(session)
            _render_debug(session)
        else:
            st.error("Dashboard data missing. Please start a new query.")
            _reset_pipeline(session)

    else:
        # * No phase — show home screen
        _render_home()


if __name__ == "__main__":
    main()