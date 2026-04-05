"""
app.py
------
Streamlit entry point — run with: streamlit run app.py

Two-phase pipeline:
  Phase 1 (Analyse clicked)            → Intent + Clarifying questions → save to session
  Phase 2 (Generate Dashboard clicked) → SQL + DB fetch + Chart → save to session
  Phase done                           → Render dashboard from session state
"""

import streamlit as st

st.set_page_config(
    page_title="Loan Collection Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import traceback
from config import (
    APP_TITLE, CODER_MODEL, FAST_MODEL, PG_TABLE_NAME,
    USE_MOCK_DATA, validate_config,
)
from utils.logger    import get_logger
from utils.helpers   import format_number, format_currency, format_percent, build_filter_summary
from utils.benchmark import QueryBenchmark

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP — cached, runs once per app lifetime
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

    # Row count — after both status={} and db= are defined
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
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def _render_sidebar(session, status: dict):
    with st.sidebar:
        st.markdown(f"## 📊 {APP_TITLE}")
        st.divider()

        st.markdown("### 🖥️ System Status")

        if status.get("db_mock_mode"):
            st.success("🗄️ Database: **Mock Mode**")
            st.caption(f"📋 {89255:,} rows (synthetic)")
        elif status.get("db_connected"):
            row_count = status.get("db_row_count", 0)
            st.success("🗄️ Database: Connected")
            st.caption(f"📋 {int(row_count):,} rows in {PG_TABLE_NAME}")
        else:
            st.error("🗄️ Database: Not Connected")
            st.caption("Check Docker and .env PG_* values.")

        if status.get("ollama_running"):
            st.success("🤖 Ollama: Running")
            c1, c2 = st.columns(2)
            c1.success("✅ Coder" if status.get("coder_available") else "❌ Coder")
            c2.success("✅ Fast"  if status.get("fast_available")  else "❌ Fast")
        else:
            st.error("🤖 Ollama: Not running")
            st.caption("Run `ollama serve` in terminal.")

        if status.get("dictionary_ready"):
            st.success("📚 Context: Loaded")
        else:
            st.warning("📚 Context: Not ready")

        for w in status.get("config_warnings", []):
            st.warning(f"⚠️ {w}")

        st.divider()
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
# PIPELINE STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _reset_pipeline(session):
    """Clear all pipeline phase state — called on New Query or example click."""
    session.set("pipeline_phase",    None)
    session.set("pending_intent",    None)
    session.set("pending_clarifier", None)
    session.set("pending_slicers",   {})
    session.reset_query_state()


def _get_phase(session) -> str:
    return session.get("pipeline_phase", None)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Intent analysis + clarifying questions
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase1(question: str, session):
    """
    Phase 1: Analyse the question and generate clarifying filter questions.
    Saves results to session then reruns to show the filter widgets cleanly.
    """
    from models.analyzer  import get_analyzer
    from models.clarifier import get_clarifier

    qb = QueryBenchmark(question)

    # Step 1 — Intent analysis
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
                label=f"✅ Metric: **{intent.metric}** (Confidence: {intent.confidence})",
                state="complete"
            )

        except Exception as e:
            qb.end("intent_analysis", error=str(e))
            st.error(f"Analysis error: {e}")
            log.error(traceback.format_exc())
            return

    # Step 2 — Clarifying questions
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
# CLARIFYING UI — shown between phase 1 and phase 2
# ──────────────────────────────────────────────────────────────────────────────

def _render_clarifying_ui(session) -> bool:
    """
    Render the filter widgets and Generate Dashboard button.
    
    KEY FIX: Saves widget answers to session.set("pending_slicers") every render.
    Phase 2 reads from pending_slicers — never calls collect_answers() again
    (which would cause DuplicateWidgetID error).
    
    Returns True when Generate Dashboard is clicked.
    """
    from models.clarifier import get_clarifier

    clarifier_result = session.get("pending_clarifier")
    intent_dict      = session.get("pending_intent", {})
    metric           = intent_dict.get("metric", "your query")

    st.markdown(f"### 🧠 Metric identified: **{metric}**")
    st.divider()

    # Render clarifying question widgets and collect current answers
    if clarifier_result and clarifier_result.success and clarifier_result.questions:
        st.markdown("#### 🔍 Help me narrow this down:")
        
        # collect_answers() renders widgets and returns current values
        answers = get_clarifier().collect_answers(clarifier_result)
        
        # Save answers EVERY render — phase 2 reads from here
        # This is critical: when Generate Dashboard is clicked, Streamlit
        # reruns BEFORE phase 2 executes, so we must persist answers in session
        session.set("pending_slicers", answers)
        
        st.divider()

    generate_clicked = st.button(
        "🚀 Generate Dashboard",
        type="primary",
        use_container_width=True,
        key="gen_btn",
    )

    if not generate_clicked:
        st.info("👆 Pick your filters then click **Generate Dashboard**.")

    return generate_clicked


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — SQL generation, DB fetch, chart decision
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase2(session) -> bool:
    """
    Phase 2: Generate SQL, fetch data, decide chart type.
    Reads intent and slicer answers from session state (set in phase 1).
    Returns True on success.
    """
    from core.filters         import FilterManager, FilterSelection
    from database.client      import get_client as get_db
    from models.analyzer      import IntentResult
    from models.sql_generator import get_sql_generator
    from models.chart_decider import get_chart_decider

    # Rebuild IntentResult from stored dict
    intent_dict = session.get("pending_intent", {})
    if not intent_dict:
        st.error("Intent data lost — please start a new query.")
        return False

    intent = IntentResult(**intent_dict)

    # Read slicer answers saved by _render_clarifying_ui()
    # DO NOT call collect_answers() here — that causes DuplicateWidgetID
    slicer_answers = session.get("pending_slicers", {})
    session.set_slicers(slicer_answers)

    log.info(f"[app] Phase 2 | metric={intent.metric} | slicers={slicer_answers}")

    qb = QueryBenchmark(intent.raw_question)

    # Step 3 — SQL generation
    with st.status("⚙️ Generating SQL query...", expanded=True) as s:
        try:
            qb.start("sql_generation")
            sql_result = get_sql_generator().generate(intent, slicer_answers)
            qb.end("sql_generation")

            if sql_result.failed:
                s.update(label=f"❌ SQL failed: {sql_result.error}", state="error")
                st.error(f"Could not generate SQL: {sql_result.error}")
                return False

            session.set_sql(sql_result.sql)
            s.update(label="✅ SQL generated", state="complete")

        except Exception as e:
            qb.end("sql_generation", error=str(e))
            st.error(f"SQL error: {e}")
            log.error(traceback.format_exc())
            return False

    # Step 4 — Fetch data
    with st.status("🗄️ Fetching data...", expanded=True) as s:
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
                label=f"✅ {db_result.row_count:,} rows | source: {db_result.source}",
                state="complete"
            )

        except Exception as e:
            qb.end("db_fetch", error=str(e))
            st.error(f"Fetch error: {e}")
            log.error(traceback.format_exc())
            return False

    # Step 5 — Apply post-fetch slicer filters to the DataFrame
    # (These are IN-MEMORY filters on top of what the SQL already filtered)
    active_sels = [
        FilterSelection(field=k, value=v)
        for k, v in slicer_answers.items()
        if v and str(v).lower() not in ("all", "")
        and not (isinstance(v, list) and len(v) == 0)
    ]
    fm = FilterManager()
    if active_sels:
        fm.detect_filters(df)
        df = fm.apply_filters(df, active_sels)

    filter_summary = fm.get_summary(active_sels) if active_sels else build_filter_summary(
        {k: v for k, v in slicer_answers.items()
         if v and str(v).lower() not in ("all", "")}
    )
    session.store_dataframe(df)

    # Step 6 — Chart decision
    with st.status("🎨 Choosing best chart...", expanded=True) as s:
        try:
            qb.start("chart_decision")
            chart_cfg = get_chart_decider().decide(df, intent, filter_summary)
            qb.end("chart_decision")
            cfg_dict = chart_cfg.to_dict()
            session.set_chart_config(cfg_dict)
            s.update(
                label=f"✅ {chart_cfg.chart_type} | x={chart_cfg.x_col} | y={chart_cfg.y_col}",
                state="complete"
            )
        except Exception as e:
            qb.end("chart_decision", error=str(e))
            cfg_dict = {}
            s.update(label="⚠️ Using default chart", state="complete")
            log.error(traceback.format_exc())

    # Save everything for dashboard rendering
    session.set("dashboard_charts", [{
        "chart_config":   session.get_current_chart_config(),
        "filter_summary": filter_summary,
        "sql":            sql_result.sql,
        "metric":         intent.metric,
        "question":       intent.raw_question,
    }])

    qb.report()
    session.record_benchmark(qb)
    session.add_chat_message("assistant", f"Here is your **{intent.metric}** dashboard.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────

def _render_kpis(df):
    import pandas as pd
    if df is None or df.empty:
        return

    kpis       = []
    cols_lower = {c.lower(): c for c in df.columns}

    checks = [
        ("bounce %",          "🔴 Bounce Rate",  "pct"),
        ("resolution %",      "✅ Resolution",   "pct"),
        ("coverage %",        "🏃 Coverage",     "pct"),
        ("bounce count",      "🔴 Bounces",      "num"),
        ("resolved count",    "✅ Resolved",     "num"),
        ("customer count",    "👥 Customers",    "num"),
        ("loan count",        "📋 Loans",        "num"),
        ("balance principal", "💰 Portfolio",    "cur"),
        ("total overdue",     "⚠️ Overdue",      "cur"),
        ("npa count",         "🚨 NPA",          "num"),
        ("avg visits",        "🏃 Intensity",    "num"),
    ]

    for pattern, label, fmt in checks:
        for col_lower, col_actual in cols_lower.items():
            if pattern in col_lower:
                try:
                    val = df[col_actual].iloc[0] if len(df) == 1 else df[col_actual].sum()
                    if pd.isna(val):
                        continue
                    val     = float(val)
                    display = (
                        format_percent(val)  if fmt == "pct" else
                        format_currency(val) if fmt == "cur" else
                        format_number(val)
                    )
                    kpis.append((label, display))
                    break
                except Exception:
                    continue
        if len(kpis) >= 4:
            break

    if not kpis:
        return

    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        col.metric(label=label, value=value)


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD RENDER
# ──────────────────────────────────────────────────────────────────────────────

def _render_dashboard(session):
    from charts.renderer import render_chart

    df      = session.get_dataframe()
    charts  = session.get("dashboard_charts", [])
    cfg_raw = session.get_current_chart_config()

    if df is None or df.empty or not charts:
        return

    data           = charts[0]
    metric         = data.get("metric", "")
    filter_summary = data.get("filter_summary", "All Data")
    sql            = data.get("sql", "")

    _render_kpis(df)
    st.markdown(f"### 📊 {metric}")
    if filter_summary and filter_summary != "All Data":
        st.caption(f"Filters: {filter_summary}")

    tab_chart, tab_data, tab_sql = st.tabs(["📈 Chart", "📋 Data", "🔍 SQL"])

    with tab_chart:
        if not cfg_raw:
            st.dataframe(df, use_container_width=True)
            return

        # Translate ChartConfig keys → what renderer.py expects
        render_config = {
            "chart_type":      cfg_raw.get("chart_type", "heatmap"),
            "x_axis":          cfg_raw.get("x_col", ""),
            "y_axis":          cfg_raw.get("y_col", ""),
            "hue":             cfg_raw.get("hue_col"),
            "color_palette":   cfg_raw.get("palette", "Set2"),
            "title":           cfg_raw.get("title", metric),
            "filters_applied": [filter_summary] if filter_summary != "All Data" else [],
        }

        try:
            result = render_chart(df, render_config)
            if result.success and result.figure:
                st.pyplot(result.figure, use_container_width=True)
            else:
                st.warning(f"Chart issue: {result.error}")
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            log.error(f"[app] Render error: {e}\n{traceback.format_exc()}")
            st.warning("Chart failed — showing raw data.")
            st.dataframe(df, use_container_width=True)

    with tab_data:
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{metric.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

    with tab_sql:
        st.markdown("**Generated SQL:**")
        st.code(sql, language="sql")
        st.caption("Generated by Qwen Coder 14B from your question and filter selections.")


# ──────────────────────────────────────────────────────────────────────────────
# DEBUG TIMINGS
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
            cols = st.columns(min(len(steps) + 1, 6))
            for i, (step, ms) in enumerate(steps.items()):
                if i < len(cols) - 1:
                    cols[i].metric(step.replace("_", " ").title(), f"{ms}ms")
            cols[-1].metric("Total", f"{total}ms")


# ──────────────────────────────────────────────────────────────────────────────
# HOME SCREEN
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
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    from core.session import get_session

    status  = _startup()
    session = get_session()
    session.cleanup_all_stale_files()

    _render_sidebar(session, status)

    st.title("📊 Loan Collection Analytics")
    st.caption("Ask any question about your portfolio in plain English.")

    if USE_MOCK_DATA:
        st.info("🎭 **Demo Mode** — 89,255 synthetic loan accounts.")

    st.divider()
    st.markdown("### 💬 Ask a question")

    # Example question buttons
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

    # Question input
    question = st.text_input(
        "question",
        placeholder="e.g. Show bounce rate by branch for Pune",
        key="_q",
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 5])
    ask_clicked = ask_col.button(
        "🔍 Analyse",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

    # ── Pipeline state machine ────────────────────────────────────────────────
    phase = _get_phase(session)

    if ask_clicked and question.strip():
        _reset_pipeline(session)
        session.set_query(question.strip())
        session.add_chat_message("user", question.strip())
        st.divider()
        st.markdown("### ⚙️ Processing...")
        _run_phase1(question.strip(), session)
        # _run_phase1 ends with st.rerun() — nothing below executes

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
                # Phase 2 failed — stay on clarifying screen so user can retry
                session.set("pipeline_phase", "clarifying")
                st.rerun()

    elif phase == "done":
        df     = session.get_dataframe()
        charts = session.get("dashboard_charts", [])

        if df is not None and not df.empty and charts:
            st.divider()
            _render_dashboard(session)
            _render_debug(session)
        else:
            st.error("Dashboard data missing. Please try again.")
            _reset_pipeline(session)
            st.rerun()

    else:
        _render_home()


if __name__ == "__main__":
    main()