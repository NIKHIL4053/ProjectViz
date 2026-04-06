"""
charts/scatter.py
-----------------
# * Scatter plot — relationship between two numeric columns.
# * Adds linear regression trend line automatically.
# * Used for: credit score vs loan amount, visit count vs resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from charts.theme import COLORS, PALETTES, FIGSIZE, style_figure, apply_theme
from utils.helpers import get_column
from utils.logger import get_charts_logger
from config import SCATTER_SAMPLE_ROWS

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Render a scatter plot with optional trend line.

    Args:
        df     : DataFrame from database client
        config : dict with keys: x_axis, y_axis, hue, title,
                 color_palette, filters_applied

    Returns:
        matplotlib Figure
    """
    apply_theme()

    x       = get_column(df, config.get("x_axis"))
    y       = get_column(df, config.get("y_axis"))
    hue     = get_column(df, config.get("hue"))
    palette = config.get("color_palette", PALETTES.DIVERGING)
    title   = config.get("title", f"{x} vs {y}")
    filters = config.get("filters_applied", [])

    # * Both axes must be numeric — auto-detect if not
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x or not pd.api.types.is_numeric_dtype(df.get(x, pd.Series())):
        x = num_cols[0] if len(num_cols) >= 1 else df.columns[0]
    if not y or not pd.api.types.is_numeric_dtype(df.get(y, pd.Series())):
        y = num_cols[1] if len(num_cols) >= 2 else num_cols[0]

    if hue and hue not in df.columns:
        hue = None

    # * Optional size column
    size_col = None
    for col in num_cols:
        if col not in (x, y):
            size_col = col
            break

    # * Sample for rendering performance
    plot_df = df.sample(min(SCATTER_SAMPLE_ROWS, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=FIGSIZE.SCATTER)

    sns.scatterplot(
        data    = plot_df,
        x       = x,
        y       = y,
        hue     = hue,
        size    = size_col,
        sizes   = (30, 280),
        alpha   = 0.65,
        palette = palette,
        ax      = ax,
    )

    # * Linear trend line
    try:
        clean = plot_df[[x, y]].dropna()
        if len(clean) > 5:
            m, b  = np.polyfit(clean[x], clean[y], 1)
            xs    = np.linspace(clean[x].min(), clean[x].max(), 200)
            ax.plot(
                xs, m * xs + b,
                "--",
                color     = COLORS.TREND_LINE,
                linewidth = 1.8,
                alpha     = 0.85,
                label     = "Trend",
                zorder    = 5,
            )
    except Exception:
        pass

    ax.set_xlabel(x, labelpad=8, fontsize=11)
    ax.set_ylabel(y, labelpad=8, fontsize=11)

    if hue or size_col:
        legend = ax.legend(framealpha=0.2, labelcolor=COLORS.TEXT_PRIMARY)

    # * Correlation annotation
    try:
        corr = plot_df[x].corr(plot_df[y])
        ax.text(
            0.02, 0.96,
            f"r = {corr:.2f}",
            transform = ax.transAxes,
            fontsize  = 10,
            color     = COLORS.TEXT_SECONDARY,
            va        = "top",
        )
    except Exception:
        pass

    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(
        f"[scatter] Rendered | x={x} | y={y} | hue={hue} | "
        f"sample={len(plot_df)}/{len(df)}"
    )
    return style_figure(fig, title=title)