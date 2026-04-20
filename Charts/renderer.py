"""
charts/renderer.py
------------------
# * Central Plotly chart router.
# * Routes to the correct chart module based on chart_type.
# * Returns a Plotly Figure — use st.plotly_chart() not st.pyplot().

Exports:
    - RenderResult  : Dataclass
    - render_chart(): Main entry point
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from utils.logger import get_logger, get_charts_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string

log        = get_logger(__name__)
charts_log = get_charts_logger(__name__)

# * Plotly dark theme template
PLOTLY_TEMPLATE = "plotly_dark"


@dataclass
class RenderResult:
    success:    bool
    figure:     Optional[go.Figure] = None
    chart_type: str                 = ""
    error:      Optional[str]       = None

    @property
    def failed(self) -> bool:
        return not self.success


def render_chart(df: pd.DataFrame, config: dict) -> RenderResult:
    """
    # * Route to the correct Plotly chart module.
    # * config uses keys: chart_type, x_col, y_col, hue_col,
    # *   palette, title, top_n, sort_desc
    # * Also accepts old keys x_axis/y_axis/hue for backwards compatibility.

    Returns:
        RenderResult with Plotly Figure.
        Use: st.plotly_chart(result.figure, use_container_width=True)
    """
    if df is None or df.empty:
        return RenderResult(success=False, error="Empty DataFrame")

    # * Normalise config keys (support both old and new key names)
    config = _normalise_config(config)

    chart_type = config.get("chart_type", "horizontal_bar").lower().strip()
    title      = config.get("title", "")

    charts_log.info(
        f"[renderer] chart_type={chart_type} | rows={len(df)} | "
        f"cols={list(df.columns)} | title='{truncate_string(title, 50)}'"
    )

    with benchmark("chart_render"):
        try:
            fig = _route(chart_type, df, config)
            charts_log.info(f"[renderer] OK | {chart_type}")
            return RenderResult(success=True, figure=fig, chart_type=chart_type)
        except Exception as e:
            charts_log.error(f"[renderer] FAILED | {chart_type} | {e}")
            try:
                fig = _error_figure(str(e), chart_type)
                return RenderResult(success=False, figure=fig,
                                    chart_type=chart_type, error=str(e))
            except Exception:
                return RenderResult(success=False,
                                    error=f"Render failed: {str(e)[:200]}")


def _normalise_config(config: dict) -> dict:
    """# * Translate old key names to new ones for backwards compatibility."""
    c = dict(config)
    if "x_axis" in c and "x_col" not in c:
        c["x_col"] = c.pop("x_axis")
    if "y_axis" in c and "y_col" not in c:
        c["y_col"] = c.pop("y_axis")
    if "hue" in c and "hue_col" not in c:
        c["hue_col"] = c.pop("hue")
    if "color_palette" in c and "palette" not in c:
        c["palette"] = c.pop("color_palette")
    return c


def _route(chart_type: str, df: pd.DataFrame, config: dict) -> go.Figure:
    """# * Import and call the correct chart module."""
    if chart_type == "horizontal_bar":
        from charts.bar import render
    elif chart_type == "line":
        from charts.line import render
    elif chart_type == "area":
        from charts.area import render
    elif chart_type == "heatmap":
        from charts.heatmap import render
    elif chart_type == "kde":
        from charts.kde import render
    elif chart_type == "scatter":
        from charts.scatter import render
    elif chart_type == "boxplot":
        from charts.boxplot import render
    elif chart_type == "treemap":
        from charts.treemap import render
    else:
        charts_log.warning(f"[renderer] Unknown '{chart_type}' → horizontal_bar")
        from charts.bar import render
    return render(df, config)


def _error_figure(error_msg: str, chart_type: str) -> go.Figure:
    """# * Returns a Plotly figure with the error message displayed."""
    fig = go.Figure()
    fig.add_annotation(
        text     = f"⚠ Could not render {chart_type} chart<br>"
                   f"<sub>{truncate_string(error_msg, 120)}</sub><br>"
                   f"<sub>Raw data is available in the 📋 Data tab</sub>",
        xref     = "paper", yref = "paper",
        x=0.5, y=0.5, showarrow=False,
        font     = {"size": 14, "color": "#fab387"},
        align    = "center",
    )
    fig.update_layout(
        template = PLOTLY_TEMPLATE,
        height   = 350,
        paper_bgcolor = "#1e1e2e",
        plot_bgcolor  = "#1e1e2e",
    )
    return fig
