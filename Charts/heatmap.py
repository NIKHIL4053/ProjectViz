"""
charts/heatmap.py
-----------------
# * Heatmap — two categorical dimensions vs a numeric metric.
# * Used for: bounce by branch/region, bucket movement matrix.
# * Dynamic sizing based on pivot dimensions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from charts.theme import COLORS, PALETTES, style_figure, apply_theme
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Render a heatmap from DataFrame and config.
    # * x_axis = columns, y_axis = rows, value = first numeric column.

    Args:
        df     : DataFrame from database client
        config : dict with keys: x_axis, y_axis, title,
                 color_palette, filters_applied

    Returns:
        matplotlib Figure
    """
    apply_theme()

    x       = get_column(df, config.get("x_axis"))
    y       = get_column(df, config.get("y_axis"))
    palette = config.get("color_palette", PALETTES.RISK)
    title   = config.get("title", "Heatmap")
    filters = config.get("filters_applied", [])

    # * Auto-detect columns if not provided
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x and len(cat_cols) >= 1:
        x = cat_cols[0]
    if not y and len(cat_cols) >= 2:
        y = cat_cols[1]
    elif not y and len(cat_cols) == 1:
        y = cat_cols[0]

    # * Find value column (numeric, not x or y)
    value_col = None
    for col in num_cols:
        if col not in (x, y):
            value_col = col
            break
    if not value_col and num_cols:
        value_col = num_cols[0]

    if not value_col or not x:
        raise ValueError(f"Heatmap needs at least one numeric column and one categorical. Got: {list(df.columns)}")

    # * Build pivot table
    if x == y:
        y = x  # single-axis heatmap fallback

    try:
        pivot = df.pivot_table(
            values  = value_col,
            index   = y if y else x,
            columns = x,
            aggfunc = "sum",
        ).fillna(0)
    except Exception as e:
        # * Fallback — simple value heatmap
        pivot = df.set_index(y or x)[[value_col]] if y else df[[value_col]]

    # * Dynamic figure sizing
    n_cols = len(pivot.columns)
    n_rows = len(pivot.index)
    fw     = max(10, min(n_cols * 1.4, 20))
    fh     = max(5,  min(n_rows * 0.75, 15))

    fig, ax = plt.subplots(figsize=(fw, fh))

    sns.heatmap(
        pivot,
        annot      = True,
        fmt        = ".0f",
        cmap       = palette,
        linewidths = 0.5,
        linecolor  = COLORS.GRID,
        annot_kws  = {"size": max(7, 11 - n_cols), "color": COLORS.TEXT_PRIMARY},
        cbar_kws   = {"shrink": 0.8},
        ax         = ax,
    )

    ax.set_xlabel(x or "", labelpad=8, fontsize=11)
    ax.set_ylabel(y or "", labelpad=8, fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(
        f"[heatmap] Rendered | x={x} | y={y} | value={value_col} | "
        f"pivot={pivot.shape} | rows={len(df)}"
    )
    return style_figure(fig, title=title)