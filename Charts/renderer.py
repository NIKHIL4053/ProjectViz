"""
charts/renderer.py
------------------
# * Central chart router — delegates to individual chart files.
# * Each chart type has its own file with full implementation.

Exports:
    - RenderResult  : Dataclass wrapping render output
    - render_chart(): Main entry point
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from charts.theme import apply_theme, COLORS
from utils.logger import get_logger, get_charts_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

log        = get_logger(__name__)
charts_log = get_charts_logger(__name__)


@dataclass
class RenderResult:
    success:    bool
    figure:     Optional[plt.Figure] = None
    chart_type: str                  = ""
    error:      Optional[str]        = None

    @property
    def failed(self) -> bool:
        return not self.success


def render_chart(df: pd.DataFrame, config: dict) -> RenderResult:
    """
    # * Route to the correct chart module based on config["chart_type"].
    # * Always returns a RenderResult — never raises.
    """
    apply_theme()

    if df is None or df.empty:
        return RenderResult(success=False, error="Empty DataFrame — no data to render")

    chart_type = config.get("chart_type", "heatmap").lower().strip()
    title      = config.get("title", "")

    charts_log.info(
        f"[renderer] Routing | chart_type={chart_type} | "
        f"rows={len(df)} | cols={list(df.columns)} | "
        f"title='{truncate_string(title, 50)}'"
    )

    with benchmark("chart_render"):
        try:
            fig = _route(chart_type, df, config)
            charts_log.info(f"[renderer] OK | chart_type={chart_type}")
            return RenderResult(success=True, figure=fig, chart_type=chart_type)

        except Exception as e:
            charts_log.error(f"[renderer] FAILED | chart_type={chart_type} | error={e}")
            try:
                fig = _error_figure(str(e), chart_type)
                return RenderResult(success=False, figure=fig,
                                    chart_type=chart_type, error=str(e))
            except Exception:
                return RenderResult(success=False,
                                    error=f"Render failed: {str(e)[:200]}")


def _route(chart_type: str, df: pd.DataFrame, config: dict) -> plt.Figure:
    if chart_type == "line":
        from charts.line import render
        return render(df, config)
    elif chart_type == "area":
        from charts.area import render
        return render(df, config)
    elif chart_type == "heatmap":
        from charts.heatmap import render
        return render(df, config)
    elif chart_type == "kde":
        from charts.kde import render
        return render(df, config)
    elif chart_type == "scatter":
        from charts.scatter import render
        return render(df, config)
    elif chart_type == "boxplot":
        from charts.boxplot import render
        return render(df, config)
    else:
        charts_log.warning(f"[renderer] Unknown '{chart_type}' — falling back to heatmap")
        from charts.heatmap import render
        return render(df, config)


def _error_figure(error_msg: str, chart_type: str) -> plt.Figure:
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS.BG_MAIN)
    ax.set_facecolor(COLORS.BG_MAIN)
    ax.axis("off")
    ax.text(0.5, 0.65, f"⚠ Could not render {chart_type} chart",
            ha="center", va="center", color=COLORS.WARN,
            fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.38, truncate_string(error_msg, 150),
            ha="center", va="center", color=COLORS.TEXT_SECONDARY,
            fontsize=9, transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.15, "Raw data is available in the 📋 Data tab below.",
            ha="center", va="center", color=COLORS.TEXT_SECONDARY,
            fontsize=9, transform=ax.transAxes)
    plt.tight_layout()
    return fig