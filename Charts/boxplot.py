"""
charts/boxplot.py
-----------------
# * Box plot — spread and outliers per category.
# * Overlays strip plot for data point transparency.
# * Used for: loan amount by branch, DPD by bucket, visit count by FE.
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
    # * Render a box plot with strip overlay.
    # * x_axis = categorical grouping column.
    # * y_axis = numeric value column.

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
    palette = config.get("color_palette", PALETTES.CATEGORICAL)
    title   = config.get("title", f"{y} by {x}")
    filters = config.get("filters_applied", [])

    # * Ensure x is categorical and y is numeric
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x or x not in df.columns:
        x = cat_cols[0] if cat_cols else df.columns[0]
    if not y or not pd.api.types.is_numeric_dtype(df[y]):
        y = num_cols[0] if num_cols else df.columns[-1]

    # * Swap if x is numeric and y is categorical
    if pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_numeric_dtype(df[y]):
        x, y = y, x

    if hue and hue not in df.columns:
        hue = None

    # * Limit to top 12 categories by median to avoid overcrowding
    if df[x].nunique() > 12:
        try:
            top_cats = df.groupby(x)[y].median().nlargest(12).index
            df = df[df[x].isin(top_cats)].copy()
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=FIGSIZE.BOXPLOT)

    sns.boxplot(
        data       = df,
        x          = x,
        y          = y,
        hue        = hue,
        palette    = palette,
        width      = 0.55,
        linewidth  = 1.3,
        flierprops = {
            "marker":           "o",
            "markersize":       3,
            "alpha":            0.4,
            "markerfacecolor":  COLORS.TEXT_SECONDARY,
        },
        ax = ax,
    )

    # * Overlay individual data points (strip plot)
    sns.stripplot(
        data   = df,
        x      = x,
        y      = y,
        dodge  = bool(hue),
        size   = 2.5,
        alpha  = 0.25,
        color  = COLORS.TEXT_SECONDARY,
        ax     = ax,
    )

    # * De-duplicate legend entries from strip overlay
    handles, labels = ax.get_legend_handles_labels()
    if handles and hue:
        half = len(handles) // 2
        legend = ax.legend(
            handles[:half],
            labels[:half],
            title      = hue,
            framealpha = 0.2,
            labelcolor = COLORS.TEXT_PRIMARY,
        )
        if legend:
            legend.get_title().set_color(COLORS.TEXT_PRIMARY)
    elif ax.get_legend():
        ax.get_legend().remove()

    ax.set_xlabel(x, labelpad=8, fontsize=11)
    ax.set_ylabel(y, labelpad=8, fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)

    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(
        f"[boxplot] Rendered | x={x} | y={y} | hue={hue} | "
        f"categories={df[x].nunique()} | rows={len(df)}"
    )
    return style_figure(fig, title=title)