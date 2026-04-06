"""
charts/kde.py
-------------
# * KDE (Kernel Density Estimate) — distribution shape of a numeric column.
# * Used for: DPD distribution, loan amount distribution, credit score spread.
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
    # * Render a KDE distribution chart.
    # * x_axis should be the numeric column to show distribution for.
    # * hue splits the distribution by a categorical column.

    Args:
        df     : DataFrame from database client
        config : dict with keys: x_axis, hue, title,
                 color_palette, filters_applied

    Returns:
        matplotlib Figure
    """
    apply_theme()

    x       = get_column(df, config.get("x_axis"))
    hue     = get_column(df, config.get("hue"))
    palette = config.get("color_palette", PALETTES.CATEGORICAL)
    title   = config.get("title", "Distribution")
    filters = config.get("filters_applied", [])

    # * x must be numeric — find first numeric column if not specified
    if not x or not pd.api.types.is_numeric_dtype(df[x]):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                x = col
                break

    if not x:
        raise ValueError(f"KDE requires at least one numeric column. Got: {list(df.columns)}")

    if hue and hue not in df.columns:
        hue = None

    # * Drop nulls for clean KDE
    plot_df = df[[x] + ([hue] if hue else [])].dropna()

    if len(plot_df) == 0:
        raise ValueError("No non-null values found for KDE")

    fig, ax = plt.subplots(figsize=FIGSIZE.KDE)

    sns.kdeplot(
        data      = plot_df,
        x         = x,
        hue       = hue,
        fill      = True,
        alpha     = 0.45,
        linewidth = 2.2,
        palette   = palette,
        ax        = ax,
    )

    # * Mean reference line
    mean_val = plot_df[x].mean()
    ax.axvline(
        mean_val,
        color     = COLORS.MEAN_LINE,
        linewidth = 1.5,
        linestyle = "--",
        alpha     = 0.85,
    )
    ylim = ax.get_ylim()
    ax.text(
        mean_val * 1.005,
        ylim[1] * 0.88,
        f" Mean: {mean_val:,.1f}",
        color    = COLORS.MEAN_LINE,
        fontsize = 9,
        va       = "top",
    )

    # * Median reference line
    median_val = plot_df[x].median()
    ax.axvline(
        median_val,
        color     = COLORS.PURPLE,
        linewidth = 1.2,
        linestyle = ":",
        alpha     = 0.7,
    )
    ax.text(
        median_val * 1.005,
        ylim[1] * 0.72,
        f" Median: {median_val:,.1f}",
        color    = COLORS.PURPLE,
        fontsize = 9,
        va       = "top",
    )

    ax.set_xlabel(x, labelpad=8, fontsize=11)
    ax.set_ylabel("Density", labelpad=8, fontsize=11)

    if hue:
        legend = ax.legend(
            title      = hue,
            framealpha = 0.2,
            labelcolor = COLORS.TEXT_PRIMARY,
        )
        if legend:
            legend.get_title().set_color(COLORS.TEXT_PRIMARY)

    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(f"[kde] Rendered | x={x} | hue={hue} | rows={len(plot_df)}")
    return style_figure(fig, title=title)