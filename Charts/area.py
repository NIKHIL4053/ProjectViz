"""
charts/area.py
--------------
# * Area chart — stacked trends showing composition over time.
# * Used for: bucket distribution over months, status composition.
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
    # * Render an area chart from DataFrame and config.

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

    if hue and hue not in df.columns:
        hue = None

    try:
        df = df.sort_values(by=x)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=FIGSIZE.AREA)

    if hue and hue in df.columns:
        try:
            pivot = df.pivot_table(
                index   = x,
                columns = hue,
                values  = y,
                aggfunc = "sum",
            ).fillna(0)
            colors = sns.color_palette(palette, n_colors=len(pivot.columns))
            pivot.plot.area(ax=ax, color=colors, alpha=0.75, linewidth=1.5)
            ax.legend(
                title       = hue,
                framealpha  = 0.2,
                labelcolor  = COLORS.TEXT_PRIMARY,
                loc         = "upper left",
            )
        except Exception:
            # * Fallback to line if pivot fails
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette)
    else:
        if pd.api.types.is_numeric_dtype(df[y]):
            ax.fill_between(
                range(len(df)),
                df[y].values,
                alpha     = 0.65,
                color     = COLORS.INFO,
                linewidth = 2,
            )
            ax.plot(range(len(df)), df[y].values, color=COLORS.INFO, linewidth=2)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x].astype(str), rotation=30, ha="right")

    ax.set_xlabel(x, labelpad=8, fontsize=11)
    ax.set_ylabel(y, labelpad=8, fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)

    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(0.5, -0.03, caption, ha="center",
                 fontsize=8, color=COLORS.TEXT_SECONDARY, style="italic")

    charts_log.info(f"[area] Rendered | x={x} | y={y} | hue={hue} | rows={len(df)}")
    return style_figure(fig, title=title)