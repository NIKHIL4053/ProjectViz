"""
charts/bar.py
-------------
# * Horizontal bar chart — best for ranking categorical data.
# * Used for: bounce count by branch, resolution % by TL, NPA by region.
# * Horizontal layout keeps long branch names readable.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    """
    # * Render a horizontal bar chart sorted by value.

    Args:
        df     : DataFrame from database client
        config : dict with keys: x_col (numeric), y_col (category),
                 palette, title, top_n, sort_desc

    Returns:
        Plotly Figure
    """
    x_col     = get_column(df, config.get("x_col")) or _first_numeric(df)
    y_col     = get_column(df, config.get("y_col")) or _first_categorical(df)
    hue_col   = get_column(df, config.get("hue_col"))
    palette   = config.get("palette", "teal")
    title     = config.get("title", f"{x_col} by {y_col}")
    top_n     = int(config.get("top_n", 0))
    sort_desc = bool(config.get("sort_desc", True))

    if not x_col or not y_col:
        raise ValueError(f"horizontal_bar needs numeric x and categorical y. Got: {list(df.columns)}")

    # * Ensure x is numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        # * Swap if needed
        if pd.api.types.is_numeric_dtype(df[y_col]):
            x_col, y_col = y_col, x_col

    plot_df = df.copy()

    # * Sort
    try:
        plot_df = plot_df.sort_values(by=x_col, ascending=not sort_desc)
    except Exception:
        pass

    # * Limit to top_n
    if top_n and top_n > 0:
        plot_df = plot_df.head(top_n)

    # * Build colour — uniform or by hue
    if hue_col and hue_col in plot_df.columns:
        fig = px.bar(
            plot_df,
            x           = x_col,
            y           = y_col,
            color       = hue_col,
            orientation = "h",
            title       = title,
            template    = TEMPLATE,
            color_continuous_scale = palette,
        )
    else:
        # * Colour bars by value (gradient)
        fig = px.bar(
            plot_df,
            x                      = x_col,
            y                      = y_col,
            orientation            = "h",
            title                  = title,
            template               = TEMPLATE,
            color                  = x_col,
            color_continuous_scale = palette,
        )

    # * Layout polish
    n_bars  = len(plot_df)
    height  = max(400, min(n_bars * 28, 900))

    fig.update_layout(
        height             = height,
        xaxis_title        = x_col,
        yaxis_title        = "",
        showlegend         = bool(hue_col),
        coloraxis_showscale= False,
        paper_bgcolor      = "#1e1e2e",
        plot_bgcolor       = "#1e1e2e",
        font               = {"color": "#cdd6f4", "size": 12},
        title_font         = {"size": 15, "color": "#89b4fa"},
        margin             = {"l": 10, "r": 30, "t": 50, "b": 30},
        yaxis              = {"categoryorder": "total ascending"
                              if sort_desc else "total descending"},
    )

    fig.update_traces(
        texttemplate = "%{x:,.1f}",
        textposition = "outside",
        textfont     = {"size": 10, "color": "#cdd6f4"},
    )

    charts_log.info(f"[bar] Rendered | x={x_col} | y={y_col} | bars={n_bars}")
    return fig


def _first_numeric(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1] if len(df.columns) else None


def _first_categorical(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0] if len(df.columns) else None


from typing import Optional
