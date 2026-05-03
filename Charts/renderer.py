"""
charts/renderer.py
------------------
# * Central Plotly chart router.
# * Routes to the correct chart module based on chart_type.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from charts.theme import COLORS, PLOTLY_TEMPLATE
from utils.benchmark import benchmark
from utils.helpers import truncate_string
from utils.logger import get_charts_logger, get_logger

log = get_logger(__name__)
charts_log = get_charts_logger(__name__)


@dataclass
class RenderResult:
    success: bool
    figure: Optional[go.Figure] = None
    chart_type: str = ""
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return not self.success


def render_chart(df: pd.DataFrame, config: dict) -> RenderResult:
    """
    # * Route to the correct Plotly chart module.
    """
    if df is None or df.empty:
        return RenderResult(success=False, error="Empty DataFrame")

    config = _normalise_config(config)
    chart_type = config.get("chart_type", "horizontal_bar").lower().strip()
    title = config.get("title", "")

    charts_log.info(
        f"[renderer] chart_type={chart_type} | rows={len(df)} | "
        f"cols={list(df.columns)} | title='{truncate_string(title, 50)}'"
    )

    with benchmark("chart_render"):
        try:
            fig = _route(chart_type, df, config)
            charts_log.info(f"[renderer] OK | {chart_type}")
            return RenderResult(success=True, figure=fig, chart_type=chart_type)
        except Exception as exc:
            charts_log.error(f"[renderer] FAILED | {chart_type} | {exc}")
            try:
                fig = _error_figure(str(exc), chart_type)
                return RenderResult(
                    success=False,
                    figure=fig,
                    chart_type=chart_type,
                    error=str(exc),
                )
            except Exception:
                return RenderResult(
                    success=False,
                    error=f"Render failed: {str(exc)[:200]}",
                )


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
        charts_log.warning(f"[renderer] Unknown '{chart_type}' -> horizontal_bar")
        from charts.bar import render
    return render(df, config)


def _error_figure(error_msg: str, chart_type: str) -> go.Figure:
    """# * Return a Plotly figure with the error message displayed."""
    fig = go.Figure()
    fig.add_annotation(
        text=(
            f"Could not render {chart_type}<br>"
            f"<sub>{truncate_string(error_msg, 120)}</sub><br>"
            f"<sub>Raw data is available in the Data tab</sub>"
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": COLORS.ACCENT_SECONDARY},
        align="center",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        paper_bgcolor=COLORS.BG_APP,
        plot_bgcolor=COLORS.BG_SURFACE,
    )
    return fig
