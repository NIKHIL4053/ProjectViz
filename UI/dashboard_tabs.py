"""
ui/dashboard_tabs.py
--------------------
# * Dashboard tab layout — Chart | Insights | Data | SQL | Export.
# * Uses Plotly for interactive charts (hover, zoom, pan).
# * Shows AI-generated insights below the chart.
"""

import html
import io

import pandas as pd
import streamlit as st

from charts.renderer import RenderResult, render_chart
from utils.logger import get_logger

log = get_logger(__name__)


def render_dashboard_tabs(
    df: pd.DataFrame,
    chart_config: dict,
    metric: str = "",
    filter_summary: str = "All Data",
    sql: str = "",
    question: str = "",
):
    """
    # * Render dashboard with tabs: Chart | Insights | Data | SQL | Export.
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return

    tab_chart, tab_insights, tab_data, tab_sql, tab_export = st.tabs(
        ["Chart", "Insights", "Data", "SQL", "Export"]
    )

    with tab_chart:
        _render_chart_tab(df, chart_config, metric, filter_summary)

    with tab_insights:
        _render_insights_tab(df, metric, question, filter_summary)

    with tab_data:
        _render_data_tab(df, metric, filter_summary)

    with tab_sql:
        _render_sql_tab(sql)

    with tab_export:
        _render_export_tab(df, chart_config, metric, filter_summary)


def _render_chart_tab(df, chart_config, metric, filter_summary):
    """# * Render Plotly chart with chart-type switcher."""
    render_config = {
        "chart_type": chart_config.get("chart_type", "horizontal_bar"),
        "x_col": chart_config.get("x_col", ""),
        "y_col": chart_config.get("y_col", ""),
        "hue_col": chart_config.get("hue_col"),
        "palette": chart_config.get("palette", "teal"),
        "title": chart_config.get("title", metric),
        "top_n": chart_config.get("top_n", 0),
        "sort_desc": chart_config.get("sort_desc", True),
        "filters_applied": [filter_summary] if filter_summary != "All Data" else [],
    }

    st.markdown(
        """
        <div class="ld-note-card">
            <strong>Visual Explorer</strong>
            <span>Use the chart controls below to switch views without rerunning the query.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        result: RenderResult = render_chart(df, render_config)
        if result.success and result.figure:
            st.plotly_chart(result.figure, use_container_width=True)
        else:
            st.warning(f"Chart could not render: {result.error}")
            st.info("Showing raw data instead.")
            st.dataframe(df, use_container_width=True)
    except Exception as exc:
        log.error(f"[dashboard_tabs] Chart render exception: {exc}")
        st.warning("Chart failed — showing raw data.")
        st.dataframe(df, use_container_width=True)

    chart_options = [
        "horizontal_bar",
        "heatmap",
        "line",
        "area",
        "kde",
        "scatter",
        "boxplot",
        "treemap",
    ]
    current_type = chart_config.get("chart_type", "horizontal_bar")
    top_n_val = int(chart_config.get("top_n", 0))

    st.markdown("##### Chart Controls")
    c1, c2, c3 = st.columns([2.2, 1, 2.8])
    with c1:
        new_type = st.selectbox(
            "Chart type",
            options=chart_options,
            index=chart_options.index(current_type) if current_type in chart_options else 0,
            key="chart_type_override",
        )
    with c2:
        st.write("")
        if st.button("Apply", key="apply_chart_type", use_container_width=True):
            chart_config["chart_type"] = new_type
            st.rerun()
    with c3:
        new_top_n = st.number_input(
            "Show top N rows (0 = all)",
            min_value=0,
            max_value=100,
            value=top_n_val,
            key="top_n_override",
        )
        if new_top_n != top_n_val:
            chart_config["top_n"] = new_top_n
            st.rerun()


def _render_insights_tab(df, metric, question, filter_summary):
    """# * Generate and display AI insights from the data."""
    from models.insight_generator import get_insight_generator

    st.markdown(
        """
        <div class="ld-note-card">
            <strong>AI Narrative</strong>
            <span>These insights are generated from the returned dataset, not from assumptions.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cache_key = f"insights_{hash(str(df.shape) + metric + filter_summary)}"

    if cache_key not in st.session_state:
        with st.spinner("Generating insights..."):
            result = get_insight_generator().generate(
                df=df,
                metric=metric,
                question=question or metric,
                filter_summary=filter_summary,
            )
        st.session_state[cache_key] = result

    result = st.session_state.get(cache_key)

    if result and result.success and result.insights:
        for i, insight in enumerate(result.insights, 1):
            st.markdown(
                f"""
                <div class="ld-insight-card">
                    <strong>Insight {i}</strong>
                    <span>{html.escape(insight)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Regenerate Insights", key="regen_insights"):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()
    else:
        st.info("Could not generate insights. Check the Data tab for raw results.")


def _render_data_tab(df, metric, filter_summary):
    st.markdown(
        f"""
        <div class="ld-note-card">
            <strong>Dataset Snapshot</strong>
            <span>{len(df):,} rows x {len(df.columns)} columns{"" if filter_summary == "All Data" else f" | Filters: {html.escape(filter_summary)}"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(df, use_container_width=True, height=420)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    safe_name = _safe_name(metric or "data")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{safe_name}.csv",
        mime="text/csv",
        key="download_csv",
    )


def _render_sql_tab(sql):
    st.markdown(
        """
        <div class="ld-note-card">
            <strong>Generated SQL</strong>
            <span>This is the exact query sent to PostgreSQL for the current dashboard.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sql:
        st.code(sql, language="sql")
        st.caption(
            "Generated by Qwen Coder from your question and filters, then executed against PostgreSQL."
        )
        with st.expander("Copy as plain text"):
            st.text_area(
                "SQL",
                value=sql,
                height=200,
                label_visibility="collapsed",
                key="sql_plain",
            )
    else:
        st.info("SQL not available.")


def _render_export_tab(df, chart_config, metric, filter_summary):
    st.markdown(
        """
        <div class="ld-note-card">
            <strong>Export Center</strong>
            <span>Download the current slice as Excel, CSV, or a chart image for reporting.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    safe_name = _safe_name(metric or "export")

    with c1:
        st.markdown(
            """
            <div class="ld-card">
                <h3>Excel Export</h3>
                <p>Includes a data sheet and a small metadata sheet with metric, filters, and shape.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
                meta = pd.DataFrame(
                    {
                        "Property": ["Metric", "Filters", "Rows", "Columns"],
                        "Value": [metric, filter_summary, len(df), len(df.columns)],
                    }
                )
                meta.to_excel(writer, index=False, sheet_name="Info")
            st.download_button(
                "Download Excel",
                data=excel_buf.getvalue(),
                file_name=f"{safe_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel",
            )
        except Exception as exc:
            st.error(f"Excel export failed: {exc}")
            st.info("Install `xlsxwriter` to enable Excel export.")

    with c2:
        st.markdown(
            """
            <div class="ld-card">
                <h3>Chart Image</h3>
                <p>Generate a PNG snapshot of the chart currently shown in the dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate PNG", key="gen_png"):
            try:
                result = render_chart(
                    df,
                    {
                        "chart_type": chart_config.get("chart_type", "horizontal_bar"),
                        "x_col": chart_config.get("x_col", ""),
                        "y_col": chart_config.get("y_col", ""),
                        "hue_col": chart_config.get("hue_col"),
                        "palette": chart_config.get("palette", "teal"),
                        "title": chart_config.get("title", metric),
                        "top_n": chart_config.get("top_n", 0),
                        "sort_desc": chart_config.get("sort_desc", True),
                    },
                )
                if result.success and result.figure:
                    img_bytes = result.figure.to_image(format="png", scale=2)
                    st.download_button(
                        "Download PNG",
                        data=img_bytes,
                        file_name=f"{safe_name}.png",
                        mime="image/png",
                        key="download_png",
                    )
            except Exception as exc:
                st.error(f"PNG export failed: {exc}")
                st.info("Install `kaleido` to enable PNG export.")


def _safe_name(text: str) -> str:
    return text.lower().replace(" ", "_").replace("%", "pct") or "export"
