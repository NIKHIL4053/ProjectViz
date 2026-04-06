"""
ui/dashboard_tabs.py
--------------------
# * Dashboard tab layout — chart, data, SQL, export.
# * Receives the filtered DataFrame and chart config.
# * Renders the full dashboard section below KPI cards.
"""

import io
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from charts.renderer import render_chart, RenderResult
from utils.logger import get_logger

log = get_logger(__name__)


def render_dashboard_tabs(
    df:             pd.DataFrame,
    chart_config:   dict,
    metric:         str         = "",
    filter_summary: str         = "All Data",
    sql:            str         = "",
):
    """
    # * Render the full dashboard with tabs: Chart | Data | SQL | Export.

    Args:
        df             : Filtered DataFrame to visualise
        chart_config   : Dict from ChartConfig.to_dict()
        metric         : Metric name for display
        filter_summary : Active filters string for caption
        sql            : Generated SQL string for SQL tab
    """
    if df is None or df.empty:
        st.warning("No data available to display.")
        return

    tab_chart, tab_data, tab_sql, tab_export = st.tabs([
        "📈 Chart",
        "📋 Data",
        "🔍 SQL",
        "📤 Export",
    ])

    # ── Chart tab ─────────────────────────────────────────────────────────────
    with tab_chart:
        _render_chart_tab(df, chart_config, metric, filter_summary)

    # ── Data tab ──────────────────────────────────────────────────────────────
    with tab_data:
        _render_data_tab(df, metric, filter_summary)

    # ── SQL tab ───────────────────────────────────────────────────────────────
    with tab_sql:
        _render_sql_tab(sql)

    # ── Export tab ────────────────────────────────────────────────────────────
    with tab_export:
        _render_export_tab(df, chart_config, metric, filter_summary, sql)


# ── Internal tab renderers ────────────────────────────────────────────────────

def _render_chart_tab(
    df:             pd.DataFrame,
    chart_config:   dict,
    metric:         str,
    filter_summary: str,
):
    """# * Render the chart inside the Chart tab."""

    render_config = {
        "chart_type":      chart_config.get("chart_type", "heatmap"),
        "x_axis":          chart_config.get("x_col", ""),
        "y_axis":          chart_config.get("y_col", ""),
        "hue":             chart_config.get("hue_col"),
        "color_palette":   chart_config.get("palette", "Set2"),
        "title":           chart_config.get("title", metric),
        "filters_applied": [filter_summary] if filter_summary != "All Data" else [],
    }

    try:
        result: RenderResult = render_chart(df, render_config)

        if result.success and result.figure:
            st.pyplot(result.figure, use_container_width=True)
            plt.close(result.figure)
        else:
            st.warning(f"Chart could not render: {result.error}")
            st.info("Showing raw data instead.")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        log.error(f"[dashboard_tabs] Chart render exception: {e}")
        st.warning("Chart failed — showing raw data.")
        st.dataframe(df, use_container_width=True)

    # * Chart type selector — let user override
    st.divider()
    st.markdown("##### 🔄 Try a different chart type")
    chart_options = ["heatmap", "line", "area", "kde", "scatter", "boxplot"]
    current_type  = chart_config.get("chart_type", "heatmap")

    c1, c2 = st.columns([2, 4])
    with c1:
        new_type = st.selectbox(
            "Chart type",
            options       = chart_options,
            index         = chart_options.index(current_type) if current_type in chart_options else 0,
            key           = "chart_type_override",
            label_visibility = "collapsed",
        )
    with c2:
        if st.button("Apply", key="apply_chart_type"):
            chart_config["chart_type"] = new_type
            st.rerun()


def _render_data_tab(
    df:             pd.DataFrame,
    metric:         str,
    filter_summary: str,
):
    """# * Render the data table and CSV download inside the Data tab."""

    st.markdown(f"**{len(df):,} rows** × **{len(df.columns)} columns**")
    if filter_summary and filter_summary != "All Data":
        st.caption(f"Filters applied: {filter_summary}")

    st.dataframe(df, use_container_width=True, height=380)

    # * CSV download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    safe_name = metric.lower().replace(" ", "_").replace("%", "pct") or "data"

    st.download_button(
        label     = "⬇️ Download CSV",
        data      = csv_bytes,
        file_name = f"{safe_name}.csv",
        mime      = "text/csv",
        key       = "download_csv",
    )


def _render_sql_tab(sql: str):
    """# * Render the generated SQL and explanation inside the SQL tab."""

    st.markdown("**Generated SQL Query:**")

    if sql:
        st.code(sql, language="sql")
        st.caption(
            "This SQL was generated by Qwen Coder 14B based on your question "
            "and filter selections. It runs against your PostgreSQL database."
        )

        # * Copy-friendly text area
        with st.expander("📋 Copy as plain text"):
            st.text_area("SQL", value=sql, height=200,
                         label_visibility="collapsed", key="sql_plain")
    else:
        st.info("SQL query not available.")


def _render_export_tab(
    df:             pd.DataFrame,
    chart_config:   dict,
    metric:         str,
    filter_summary: str,
    sql:            str,
):
    """# * Render PDF and Excel export options inside the Export tab."""

    st.markdown("#### 📤 Export Options")

    c1, c2 = st.columns(2)

    # ── Excel export ──────────────────────────────────────────────────────────
    with c1:
        st.markdown("**Excel (.xlsx)**")
        st.caption("Download the filtered data as an Excel file.")

        try:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
                # * Add filter metadata sheet
                meta = pd.DataFrame({
                    "Property": ["Metric", "Filters", "Rows", "Columns"],
                    "Value":    [metric, filter_summary, len(df), len(df.columns)],
                })
                meta.to_excel(writer, index=False, sheet_name="Info")

            excel_bytes = excel_buf.getvalue()
            safe_name   = metric.lower().replace(" ", "_").replace("%", "pct") or "export"

            st.download_button(
                label     = "⬇️ Download Excel",
                data      = excel_bytes,
                file_name = f"{safe_name}.xlsx",
                mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key       = "download_excel",
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")
            st.info("Install xlsxwriter: pip install xlsxwriter")

    # ── PDF export ────────────────────────────────────────────────────────────
    with c2:
        st.markdown("**PDF Report**")
        st.caption("Download the chart as a PDF report.")

        if st.button("📄 Generate PDF", key="gen_pdf"):
            try:
                pdf_bytes = _generate_pdf(df, chart_config, metric, filter_summary)
                safe_name = metric.lower().replace(" ", "_").replace("%", "pct") or "report"
                st.download_button(
                    label     = "⬇️ Download PDF",
                    data      = pdf_bytes,
                    file_name = f"{safe_name}_report.pdf",
                    mime      = "application/pdf",
                    key       = "download_pdf",
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")
                st.info("Install reportlab: pip install reportlab")


def _generate_pdf(
    df:             pd.DataFrame,
    chart_config:   dict,
    metric:         str,
    filter_summary: str,
) -> bytes:
    """
    # * Generate a PDF report with the chart and data summary.

    Returns:
        PDF as bytes.
    """
    import tempfile, os

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize    = landscape(A4),
        topMargin   = 30,
        bottomMargin= 30,
        leftMargin  = 40,
        rightMargin = 40,
    )

    styles  = getSampleStyleSheet()
    story   = []

    # * Title
    story.append(Paragraph(
        f"<b>Loan Collection Analytics — {metric}</b>",
        styles["Heading1"]
    ))
    story.append(Spacer(1, 6))

    # * Filter summary
    story.append(Paragraph(
        f"<i>Filters: {filter_summary}</i>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # * Render chart to temp image
    render_config = {
        "chart_type":      chart_config.get("chart_type", "heatmap"),
        "x_axis":          chart_config.get("x_col", ""),
        "y_axis":          chart_config.get("y_col", ""),
        "hue":             chart_config.get("hue_col"),
        "color_palette":   chart_config.get("palette", "Set2"),
        "title":           chart_config.get("title", metric),
        "filters_applied": [filter_summary] if filter_summary != "All Data" else [],
    }

    result = render_chart(df, render_config)
    if result.success and result.figure:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            result.figure.savefig(
                tmp.name,
                dpi         = 150,
                bbox_inches = "tight",
                facecolor   = "#1e1e2e",
            )
            plt.close(result.figure)
            tmp_path = tmp.name

        story.append(RLImage(tmp_path, width=680, height=300))
        story.append(Spacer(1, 12))
        os.unlink(tmp_path)

    # * Data summary
    story.append(Paragraph(
        f"<b>Data Summary</b> — {len(df):,} rows × {len(df.columns)} columns",
        styles["Heading3"]
    ))

    doc.build(story)
    return buf.getvalue()