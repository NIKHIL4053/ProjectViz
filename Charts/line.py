"""
charts/line.py
--------------
# * Line chart — trends over time or across ordered categories.
# * Used for: bounce trend, resolution trend, NPA over time.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from charts.theme import COLORS, PALETTES, FIGSIZE, style_figure, apply_theme
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Render a line chart from DataFrame and config.

    Args:
        df     : DataFrame from database client
        config : dict with keys: x_axis, y_axis, hue, title,
                 color_palette, filters_applied

    Returns:
        matplotlib Figure
    """
    apply_theme()

    x   = get_column(df, config.get("x_axis")) or df.columns[0]
    y   = get_column(df, config.get("y_axis")) or next(
            (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x),
            df.columns[-1]
          )
    hue     = get_column(df, config.get("hue"))
    palette = config.get("color_palette", PALETTES.CATEGORICAL)
    title   = config.get("title", f"{y} over {x}")
    filters = config.get("filters_applied", [])

    # * Validate hue
    if hue and hue not in df.columns:
        hue = None

    # * Sort by x for clean line
    try:
        df = df.sort_values(by=x)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=FIGSIZE.LINE)

    sns.lineplot(
        data       = df,
        x          = x,
        y          = y,
        hue        = hue,
        marker     = "o",
        linewidth  = 2.5,
        markersize = 6,
        palette    = palette,
        ax         = ax,
    )

    # * Subtle fill under line when no grouping
    if not hue and pd.api.types.is_numeric_dtype(df[y]):
        try:
            ax.fill_between(
                range(len(df)),
                df[y].values,
                alpha = 0.12,
                color = COLORS.INFO,
            )
        except Exception:
            pass

    ax.set_xlabel(x, labelpad=8, fontsize=11)
    ax.set_ylabel(y, labelpad=8, fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)

    if hue:
        legend = ax.legend(
            title       = hue,
            framealpha  = 0.2,
            labelcolor  = COLORS.TEXT_PRIMARY,
            loc         = "best",
        )
        legend.get_title().set_color(COLORS.TEXT_PRIMARY)

    # * Filter caption
    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(f"[line] Rendered | x={x} | y={y} | hue={hue} | rows={len(df)}")
    return style_figure(fig, title=title)