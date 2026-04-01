"""
app.py
------
Streamlit entry point — run with: streamlit run app.py
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
    APP_TITLE, CODER_MODEL, FAST_MODEL,
    USE_MOCK_DATA, validate_config,
)
from utils.logger   import get_logger
from utils.helpers  import format_number, format_currency, format_percent, build_filter_summary
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
        elif status.get("db_connected"):
            st.success("🗄️ Database: Connected")
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

        st.success("📚 Context: Loaded") if status.get("dictionary_ready") else st.warning("📚 Context: Not ready")

        for w in status.get("config_warnings", []):
            st.warning(f"⚠️ {w}")

        st.divider()
        st.markdown("### ⚙️ Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 New Query", use_container_width=True):
                session.reset_query_state()
                st.rerun()
        with c2:
            if st.button("🗑️ Clear All", use_container_width=True):
                session.full_reset()
                st.rerun()

        session.set("show_debug", st.checkbox("🔬 Show Timings", value=False))

        if USE_MOCK_DATA:
            st.divider()
            st.info("🎭 **Demo Mode** — mock data (89K synthetic loans).")


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
        ("bounce %",          "🔴 Bounce Rate",     "pct"),
        ("resolution %",      "✅ Resolution",      "pct"),
        ("coverage %",        "🏃 Coverage",        "pct"),
        ("bounce count",      "🔴 Bounces",         "num"),
        ("resolved count",    "✅ Resolved",        "num"),
        ("customer count",    "👥 Customers",       "num"),
        ("loan count",        "📋 Loans",           "num"),
        ("balance principal", "💰 Portfolio",       "cur"),
        ("total overdue",     "⚠️ Overdue",         "cur"),
        ("npa count",         "🚨 NPA",             "num"),
        ("npa customers",     "🚨 NPA Customers",   "num"),
    ]

    for pattern, label, fmt in checks:
        for col_lower, col_actual in cols_lower.items():
            if pattern in col_lower:
                try:
                    val = df[col_actual].iloc[0] if len(df) == 1 else df[col_actual].sum()
                    if pd.isna(val):
                        continue
                    val = float(val)
                    display = (format_percent(val) if fmt == "pct"
                               else format_currency(val) if fmt == "cur"
                               else format_number(val))
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
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(question: str, session, qb: QueryBenchmark) -> bool:
    from core.filters         import FilterManager, FilterSelection
    from database.client      import get_client as get_db
    from models.analyzer      import get_analyzer
    from models.clarifier     import get_clarifier
    from models.sql_generator import get_sql_generator
    from models.chart_decider import get_chart_decider

    # Step 1 — Intent
    with st.status("🧠 Understanding your question...", expanded=False) as s:
        try:
            qb.start("intent_analysis")
            intent = get_analyzer().analyze(question)
            qb.end("intent_analysis")
            if intent.failed:
                s.update(label=f"❌ {intent.error}", state="error")
                st.error(f"Could not understand: {intent.error}")
                return False
            session.set_intent(intent.to_dict())
            s.update(label=f"✅ Metric: **{intent.metric}** ({intent.confidence})", state="complete")
        except Exception as e:
            qb.end("intent_analysis", error=str(e))
            st.error(f"Analysis error: {e}")
            log.error(traceback.format_exc())
            return False

    # Step 2 — Clarifying questions
    clarifier_result = None
    with st.status("❓ Generating filter options...", expanded=False) as s:
        try:
            qb.start("clarifying_questions")
            clarifier_result = get_clarifier().generate(intent)
            qb.end("clarifying_questions")
            s.update(label="✅ Filters ready", state="complete")
        except Exception as e:
            qb.end("clarifying_questions", error=str(e))
            s.update(label="⚠️ Skipping filters", state="complete")

    # Step 3 — Collect answers
    slicer_answers = {}
    if clarifier_result and clarifier_result.success and clarifier_result.questions:
        st.markdown("---")
        slicer_answers = get_clarifier().collect_answers(clarifier_result)
        session.set_slicers(slicer_answers)
        st.markdown("---")
        if not st.button("🚀 Generate Dashboard", type="primary",
                         use_container_width=True, key="gen_btn"):
            st.info("👆 Pick your filters then click **Generate Dashboard**.")
            return False

    # Step 4 — SQL
    with st.status("⚙️ Building query...", expanded=False) as s:
        try:
            qb.start("sql_generation")
            sql_result = get_sql_generator().generate(intent, slicer_answers)
            qb.end("sql_generation")
            if sql_result.failed:
                s.update(label="❌ Query failed", state="error")
                st.error(f"SQL error: {sql_result.error}")
                return False
            session.set_sql(sql_result.sql)
            s.update(label=f"✅ Query ready", state="complete")
        except Exception as e:
            qb.end("sql_generation", error=str(e))
            st.error(f"SQL error: {e}")
            log.error(traceback.format_exc())
            return False

    # Step 5 — Fetch data
    with st.status("🗄️ Fetching data...", expanded=False) as s:
        try:
            qb.start("db_fetch")
            result = get_db().run_query(sql_result.sql)
            qb.end("db_fetch")
            if result.failed:
                s.update(label="❌ Fetch failed", state="error")
                st.error(f"DB error: {result.error}")
                return False
            if result.is_empty:
                s.update(label="⚠️ No data", state="complete")
                st.warning("No data for these filters — try broader selections.")
                return False
            df = result.dataframe
            session.store_dataframe(df)
            s.update(
                label=f"✅ {format_number(result.row_count)} rows × {result.col_count} cols "
                      f"({'mock' if result.source == 'mock' else 'live'})",
                state="complete"
            )
        except Exception as e:
            qb.end("db_fetch", error=str(e))
            st.error(f"Data error: {e}")
            log.error(traceback.format_exc())
            return False

    # Step 6 — Apply any sidebar filters
    filter_summary = build_filter_summary(
        {k: v for k, v in slicer_answers.items()
         if v and str(v).lower() not in ("all", "")}
    )
    active = session.get("filter_selections", {})
    if active:
        fm   = FilterManager()
        fm.detect_filters(df)
        sels = [FilterSelection(field=k, value=v) for k, v in active.items()]
        df   = fm.apply_filters(df, sels)
        filter_summary = fm.get_summary(sels)
        session.store_dataframe(df)

    # Step 7 — Chart decision
    cfg_dict = {}
    with st.status("🎨 Choosing chart type...", expanded=False) as s:
        try:
            qb.start("chart_decision")
            chart_cfg = get_chart_decider().decide(df, intent, filter_summary)
            qb.end("chart_decision")
            cfg_dict = chart_cfg.to_dict()
            session.set_chart_config(cfg_dict)
            s.update(
                label=f"✅ {chart_cfg.chart_type} | x={chart_cfg.x_col} y={chart_cfg.y_col}",
                state="complete"
            )
        except Exception as e:
            qb.end("chart_decision", error=str(e))
            s.update(label="⚠️ Default chart", state="complete")
            log.error(traceback.format_exc())

    session.set("dashboard_charts", [{
        "chart_config":   cfg_dict,
        "filter_summary": filter_summary,
        "sql":            sql_result.sql,
        "metric":         intent.metric,
        "question":       question,
    }])
    return True


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD
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

        # Translate ChartConfig field names → what renderer.py expects
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
        cols  = st.columns(len(steps) + 1) if steps else st.columns(1)
        for i, (step, ms) in enumerate(steps.items()):
            cols[i].metric(step.replace("_", " ").title(), f"{ms}ms")
        cols[-1].metric("Total", f"{total}ms")


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
                st.session_state["_q"] = ex
                st.rerun()

    question = st.text_input(
        "question",
        placeholder="e.g. Show bounce rate by branch for Pune",
        key="_q",
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 5])
    ask_clicked = ask_col.button(
        "🔍 Analyse", type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

    if ask_clicked and question.strip():
        session.set_query(question.strip())
        session.add_chat_message("user", question.strip())
        qb = QueryBenchmark(question.strip())

        st.divider()
        st.markdown("### ⚙️ Processing...")
        success = _run_pipeline(question.strip(), session, qb)
        qb.report()
        session.record_benchmark(qb)

        if success:
            session.add_chat_message(
                "assistant",
                f"Here is your **{session.get_current_intent().get('metric','analysis')}** dashboard."
            )
            st.rerun()

    df     = session.get_dataframe()
    charts = session.get("dashboard_charts", [])

    if df is not None and not df.empty and charts:
        st.divider()
        _render_dashboard(session)
        _render_debug(session)
    elif not ask_clicked:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown("#### 📋 What can I ask?\n"
                    "- Bounce rate by branch\n- Resolution % by TL\n"
                    "- NPA by region\n- Coverage analysis\n"
                    "- Bucket movement\n- FE scorecard")
        c2.markdown("#### 🔧 How it works\n"
                    "1. Type your question\n2. AI finds the metric\n"
                    "3. Pick filters\n4. SQL ~runs\n5. Chart builds")
        c3.markdown("#### 📊 Metrics\n"
                    "- Bounce % (Tech/Non-Tech)\n- Resolution %\n"
                    "- Coverage %\n- Visit Intensity\n"
                    "- Portfolio Outstanding\n- Bucket Distribution\n"
                    "- NPA Count\n- FE Scorecard")


if __name__ == "__main__":
    main()