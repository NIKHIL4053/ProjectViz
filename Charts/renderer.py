"""
charts/renderer.py
------------------
# * Central chart renderer for the Loan Dashboard.
# * Receives a ChartConfig from chart_decider.py and a DataFrame from the DB.
# * Routes to the correct chart function and returns a matplotlib Figure.
# * All 6 chart types are implemented here — no separate files needed.

# ? Why all charts in one file instead of separate files?
# ? For a prototype, one file is faster to maintain and easier to debug.
# ? The individual chart files (line.py, area.py etc.) stay empty for now
# ? and can be populated later if individual chart logic grows complex.

# ? Chart types supported:
# ?   line    — trends over time or across ordered categories
# ?   area    — stacked trends showing composition
# ?   heatmap — two categorical dimensions vs a numeric metric
# ?   kde     — distribution shape of a numeric column
# ?   scatter — relationship between two numeric columns
# ?   boxplot — spread and outliers per category

Exports:
    - RenderResult     : Dataclass wrapping render output
    - render_chart()   : Main entry point — takes config + df, returns figure
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from charts.theme import (
    apply_theme, style_figure,
    COLORS, PALETTES, FIGSIZE,
)
from utils.logger import get_logger, get_charts_logger
from utils.benchmark import benchmark
from utils.helpers import get_column, truncate_string

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# * Module level loggers
log        = get_logger(__name__)
charts_log = get_charts_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * RENDER RESULT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderResult:
    """
    # * Wraps every chart render response.

    Attributes:
        success    : True if chart rendered successfully
        figure     : matplotlib Figure — pass to st.pyplot()
        chart_type : Which chart type was rendered
        error      : Error message if success=False
    """
    success:    bool
    figure:     Optional[plt.Figure]  = None
    chart_type: str                   = ""
    error:      Optional[str]         = None

    @property
    def failed(self) -> bool:
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# * MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def render_chart(
    df:     pd.DataFrame,
    config: dict,
) -> RenderResult:
    """
    # * Route to the correct chart function based on config["chart_type"].
    # * Always returns a RenderResult — check .success before using .figure.

    Args:
        df     : DataFrame from database client (real or mock)
        config : ChartConfig.to_dict() from chart_decider.py
                 Keys: chart_type, x_axis, y_axis, hue, title,
                       color_palette, filters_applied

    Returns:
        RenderResult with matplotlib Figure.

    Example:
        result = render_chart(df, chart_config.to_dict())
        if result.success:
            st.pyplot(result.figure, use_container_width=True)
    """
    apply_theme()

    if df is None or df.empty:
        return RenderResult(
            success=False,
            error="Empty DataFrame — no data to render"
        )

    chart_type = config.get("chart_type", "line").lower().strip()
    title      = config.get("title", "")
    filters    = config.get("filters_applied", [])

    charts_log.info(
        f"[renderer] Rendering | "
        f"chart_type={chart_type} | "
        f"rows={len(df)} | "
        f"title='{truncate_string(title, 60)}'"
    )

    with benchmark("chart_render"):
        # * Route to correct chart function
        router = {
            "line":    _render_line,
            "area":    _render_area,
            "heatmap": _render_heatmap,
            "kde":     _render_kde,
            "scatter": _render_scatter,
            "boxplot": _render_boxplot,
        }

        render_fn = router.get(chart_type)

        if not render_fn:
            charts_log.warning(
                f"[renderer] Unknown chart type '{chart_type}' — falling back to line"
            )
            render_fn = _render_line

        try:
            fig = render_fn(df, config)
            charts_log.info(
                f"[renderer] OK | chart_type={chart_type} | title='{truncate_string(title, 60)}'"
            )
            return RenderResult(
                success    = True,
                figure     = fig,
                chart_type = chart_type,
            )

        except Exception as e:
            charts_log.error(
                f"[renderer] FAILED | chart_type={chart_type} | error={e}"
            )
            # * Try to return a fallback error figure instead of crashing
            try:
                fig = _render_error_figure(str(e), chart_type)
                return RenderResult(
                    success    = False,
                    figure     = fig,
                    chart_type = chart_type,
                    error      = str(e),
                )
            except Exception:
                return RenderResult(
                    success=False,
                    error=f"Chart render failed: {str(e)[:200]}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# * SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_axes(df: pd.DataFrame, config: dict) -> tuple:
    """
    # * Resolve x_axis, y_axis, hue column names from config.
    # * Uses case-insensitive lookup — handles Qwen returning wrong casing.
    # * Falls back to first available numeric column if y_axis not found.

    Returns:
        (x, y, hue) — actual column names from df.columns, or None.
    """
    x   = get_column(df, config.get("x_axis"))
    y   = get_column(df, config.get("y_axis"))
    hue = get_column(df, config.get("hue"))

    # * Fallback x — first non-numeric column
    if not x:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                x = col
                break
        if not x:
            x = df.columns[0]

    # * Fallback y — first numeric column that is not x
    if not y:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != x:
                y = col
                break
        if not y:
            y = df.columns[-1]

    # * Nullify invalid hue
    if hue and hue not in df.columns:
        hue = None
    if hue and str(hue).lower() in ("null", "none", ""):
        hue = None

    return x, y, hue


def _add_filter_caption(fig: plt.Figure, filters: list):
    """# * Add applied filters as a small footer caption on the figure."""
    if filters:
        caption = "Filters: " + "  |  ".join(str(f) for f in filters[:4])
        fig.text(
            0.5, -0.03,
            caption,
            ha       = "center",
            fontsize = 8,
            color    = COLORS.TEXT_SECONDARY,
            style    = "italic",
        )


def _get_palette(config: dict) -> str:
    """# * Get color palette from config with sensible default."""
    return config.get("color_palette", PALETTES.CATEGORICAL)


# ──────────────────────────────────────────────────────────────────────────────
# * LINE CHART
# ──────────────────────────────────────────────────────────────────────────────

def _render_line(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Line chart — best for trends over time or across ordered categories.
    # * Adds a subtle fill under the line when no hue grouping is used.
    """
    x, y, hue = _resolve_axes(df, config)
    palette   = _get_palette(config)

    # * Sort by x for clean line
    try:
        df = df.sort_values(by=x)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=FIGSIZE.LINE)

    sns.lineplot(
        data      = df,
        x         = x,
        y         = y,
        hue       = hue,
        marker    = "o",
        linewidth = 2.5,
        markersize= 6,
        palette   = palette,
        ax        = ax,
    )

    # * Subtle fill under line when no grouping
    if not hue and pd.api.types.is_numeric_dtype(df[y]):
        try:
            ax.fill_between(
                range(len(df)),
                df[y].values,
                alpha = COLORS.FILL_ALPHA,
                color = COLORS.INFO,
            )
        except Exception:
            pass

    ax.set_xlabel(x, labelpad=8)
    ax.set_ylabel(y, labelpad=8)
    plt.xticks(rotation=30, ha="right")

    if hue:
        ax.legend(title=hue, framealpha=0.2, labelcolor=COLORS.TEXT_PRIMARY)

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"{y} over {x}"))


# ──────────────────────────────────────────────────────────────────────────────
# * AREA CHART
# ──────────────────────────────────────────────────────────────────────────────

def _render_area(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Area chart — stacked areas showing composition over time.
    # * Falls back to filled line chart if pivot fails.
    """
    x, y, hue = _resolve_axes(df, config)
    palette   = _get_palette(config)

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
            # * Fallback to regular lineplot if pivot fails
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette)
    else:
        if pd.api.types.is_numeric_dtype(df[y]):
            ax.fill_between(range(len(df)), df[y].values, alpha=0.65, color=COLORS.INFO, linewidth=2)
            ax.plot(range(len(df)), df[y].values, color=COLORS.INFO, linewidth=2)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x].astype(str), rotation=30, ha="right")

    ax.set_xlabel(x, labelpad=8)
    ax.set_ylabel(y, labelpad=8)

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"{y} over {x}"))


# ──────────────────────────────────────────────────────────────────────────────
# * HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def _render_heatmap(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Heatmap — two categorical dimensions vs a numeric metric.
    # * Dynamically sizes figure based on pivot dimensions.
    # * Uses RdYlGn_r by default — red=bad, green=good for collection data.
    """
    x, y, _   = _resolve_axes(df, config)
    palette   = _get_palette(config) or PALETTES.RISK

    # * Find numeric value column (not x or y)
    value_col = None
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in (x, y):
            value_col = col
            break

    if not value_col:
        # * If no separate value col, try y as value
        if pd.api.types.is_numeric_dtype(df[y]):
            value_col = y
            y_axis    = x
            # * Need a second categorical — take first categorical col
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]) and col != x:
                    y = col
                    break

    if not value_col:
        raise ValueError("Heatmap requires at least one numeric column")

    # * Build pivot
    pivot = df.pivot_table(
        values  = value_col,
        index   = y,
        columns = x,
        aggfunc = "sum",
    ).fillna(0)

    # * Dynamic figure size
    fw = max(10, len(pivot.columns) * 1.3)
    fh = max(6,  len(pivot.index)   * 0.8)
    fig, ax = plt.subplots(figsize=(fw, fh))

    sns.heatmap(
        pivot,
        annot       = True,
        fmt         = ".0f",
        cmap        = palette,
        linewidths  = 0.5,
        linecolor   = COLORS.GRID,
        annot_kws   = {"size": 9, "color": COLORS.TEXT_PRIMARY},
        cbar_kws    = {"shrink": 0.8},
        ax          = ax,
    )

    ax.set_xlabel(x, labelpad=8)
    ax.set_ylabel(y, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"{value_col} by {x} and {y}"))


# ──────────────────────────────────────────────────────────────────────────────
# * KDE DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────

def _render_kde(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * KDE — shows distribution shape of a numeric column.
    # * Adds a mean reference line with label.
    # * Uses hue to split distribution by a categorical variable.
    """
    x, _, hue = _resolve_axes(df, config)
    palette   = _get_palette(config)

    # * x must be numeric for KDE
    x_num = None
    if x and pd.api.types.is_numeric_dtype(df[x]):
        x_num = x
    else:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                x_num = col
                break

    if not x_num:
        raise ValueError("KDE requires at least one numeric column")

    fig, ax = plt.subplots(figsize=FIGSIZE.KDE)

    sns.kdeplot(
        data      = df,
        x         = x_num,
        hue       = hue,
        fill      = True,
        alpha     = 0.45,
        linewidth = 2.2,
        palette   = palette,
        ax        = ax,
    )

    # * Mean reference line
    mean_val = df[x_num].mean()
    ax.axvline(mean_val, color=COLORS.MEAN_LINE, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(
        mean_val * 1.01,
        ax.get_ylim()[1] * 0.88,
        f" Mean: {mean_val:,.1f}",
        color    = COLORS.MEAN_LINE,
        fontsize = 9,
    )

    ax.set_xlabel(x_num, labelpad=8)
    ax.set_ylabel("Density", labelpad=8)

    if hue:
        ax.legend(title=hue, framealpha=0.2, labelcolor=COLORS.TEXT_PRIMARY)

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"Distribution of {x_num}"))


# ──────────────────────────────────────────────────────────────────────────────
# * SCATTER PLOT
# ──────────────────────────────────────────────────────────────────────────────

def _render_scatter(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Scatter plot — shows relationship between two numeric columns.
    # * Adds a linear regression trend line.
    # * Samples down to SCATTER_SAMPLE_ROWS to keep rendering fast.
    """
    from config import SCATTER_SAMPLE_ROWS
    x, y, hue = _resolve_axes(df, config)
    palette   = _get_palette(config)

    # * Both x and y must be numeric
    if not pd.api.types.is_numeric_dtype(df[x]):
        # * Find first numeric col
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != y:
                x = col
                break

    if not pd.api.types.is_numeric_dtype(df[y]):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != x:
                y = col
                break

    # * Sample for performance
    plot_df = df.sample(min(SCATTER_SAMPLE_ROWS, len(df)), random_state=42)

    # * Optional size column
    size_col = None
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in (x, y):
            size_col = col
            break

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

    # * Trend line
    try:
        m, b = np.polyfit(plot_df[x], plot_df[y], 1)
        xs   = np.linspace(plot_df[x].min(), plot_df[x].max(), 200)
        ax.plot(xs, m * xs + b, "--", color=COLORS.TREND_LINE,
                linewidth=1.5, alpha=0.8, label="Trend")
    except Exception:
        pass

    ax.set_xlabel(x, labelpad=8)
    ax.set_ylabel(y, labelpad=8)

    if hue or size_col:
        ax.legend(framealpha=0.2, labelcolor=COLORS.TEXT_PRIMARY)

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"{x} vs {y}"))


# ──────────────────────────────────────────────────────────────────────────────
# * BOX PLOT
# ──────────────────────────────────────────────────────────────────────────────

def _render_boxplot(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    # * Box plot — spread and outliers of a numeric column per category.
    # * Overlays a strip plot for data transparency.
    # * Limits to top 12 categories by median if too many.
    """
    x, y, hue = _resolve_axes(df, config)
    palette   = _get_palette(config)

    # * y must be numeric
    if not pd.api.types.is_numeric_dtype(df[y]):
        # * Swap x and y
        x, y = y, x

    # * Limit categories to top 12 by median
    if df[x].nunique() > 12:
        top_cats = df.groupby(x)[y].median().nlargest(12).index
        df = df[df[x].isin(top_cats)].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE.BOXPLOT)

    sns.boxplot(
        data        = df,
        x           = x,
        y           = y,
        hue         = hue,
        palette     = palette,
        width       = 0.6,
        linewidth   = 1.3,
        flierprops  = {"marker": "o", "markersize": 3, "alpha": 0.4, "markerfacecolor": COLORS.TEXT_SECONDARY},
        ax          = ax,
    )

    # * Overlay individual data points
    sns.stripplot(
        data    = df,
        x       = x,
        y       = y,
        dodge   = bool(hue),
        size    = 2.5,
        alpha   = 0.25,
        color   = COLORS.TEXT_SECONDARY,
        ax      = ax,
    )

    # * De-duplicate legend (stripplot adds duplicate entries)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        half = len(handles) // 2
        ax.legend(
            handles[:half], labels[:half],
            title      = hue,
            framealpha = 0.2,
            labelcolor = COLORS.TEXT_PRIMARY,
        )

    ax.set_xlabel(x, labelpad=8)
    ax.set_ylabel(y, labelpad=8)
    plt.xticks(rotation=30, ha="right")

    _add_filter_caption(fig, config.get("filters_applied", []))
    return style_figure(fig, title=config.get("title", f"{y} by {x}"))


# ──────────────────────────────────────────────────────────────────────────────
# * ERROR FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def _render_error_figure(error_msg: str, chart_type: str) -> plt.Figure:
    """
    # * Returns a figure with the error message displayed.
    # * Shown in the UI when chart rendering fails so the page doesn't crash.

    Args:
        error_msg  : Error string to display
        chart_type : Chart type that failed

    Returns:
        matplotlib Figure with error text.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(COLORS.BG_MAIN)
    ax.set_facecolor(COLORS.BG_MAIN)
    ax.axis("off")

    ax.text(
        0.5, 0.6,
        f"⚠ Could not render {chart_type} chart",
        ha         = "center",
        va         = "center",
        color      = COLORS.WARN,
        fontsize   = 13,
        fontweight = "bold",
        transform  = ax.transAxes,
    )
    ax.text(
        0.5, 0.35,
        truncate_string(error_msg, 150),
        ha        = "center",
        va        = "center",
        color     = COLORS.TEXT_SECONDARY,
        fontsize  = 9,
        transform = ax.transAxes,
        wrap      = True,
    )
    plt.tight_layout()
    return fig