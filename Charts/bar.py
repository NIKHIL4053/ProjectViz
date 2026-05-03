"""
charts/bar.py
-------------
# * Horizontal bar chart for ranking categorical data.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts.theme import (
    COLORS,
    PLOTLY_TEMPLATE,
    get_plotly_colorscale,
    get_plotly_sequence,
    style_plotly_figure,
)
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col = get_column(df, config.get("x_col")) or _first_numeric(df)
    y_col = get_column(df, config.get("y_col")) or _first_categorical(df)
    hue_col = get_column(df, config.get("hue_col"))
    palette = config.get("palette", "teal")
    title = config.get("title", f"{x_col} by {y_col}")
    top_n = int(config.get("top_n", 0))
    sort_desc = bool(config.get("sort_desc", True))

    if not x_col or not y_col:
        raise ValueError(f"horizontal_bar needs numeric x and categorical y. Got: {list(df.columns)}")

    if not pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        x_col, y_col = y_col, x_col

    plot_df = df.copy()
    try:
        plot_df = plot_df.sort_values(by=x_col, ascending=not sort_desc)
    except Exception:
        pass

    if top_n and top_n > 0:
        plot_df = plot_df.head(top_n)

    if hue_col and hue_col in plot_df.columns:
        common_args = {
            "data_frame": plot_df,
            "x": x_col,
            "y": y_col,
            "color": hue_col,
            "orientation": "h",
            "title": title,
            "template": PLOTLY_TEMPLATE,
        }
        if pd.api.types.is_numeric_dtype(plot_df[hue_col]):
            fig = px.bar(
                **common_args,
                color_continuous_scale=get_plotly_colorscale(palette),
            )
        else:
            fig = px.bar(
                **common_args,
                color_discrete_sequence=get_plotly_sequence(palette),
            )
    else:
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            orientation="h",
            title=title,
            template=PLOTLY_TEMPLATE,
            color=x_col,
            color_continuous_scale=get_plotly_colorscale(palette),
        )

    n_bars = len(plot_df)
    height = max(420, min(n_bars * 30, 920))

    style_plotly_figure(
        fig,
        title=title,
        height=height,
        xaxis_title=x_col,
        yaxis_title="",
        showlegend=bool(hue_col),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin={"l": 16, "r": 26, "t": 72, "b": 30},
    )
    fig.update_yaxes(
        categoryorder="total ascending" if sort_desc else "total descending"
    )
    fig.update_traces(
        texttemplate="%{x:,.1f}",
        textposition="outside",
        textfont={"size": 10, "color": COLORS.TEXT_PRIMARY},
        cliponaxis=False,
        marker_line_color="rgba(255,255,255,0.08)",
        marker_line_width=0.8,
    )

    charts_log.info(f"[bar] Rendered | x={x_col} | y={y_col} | bars={n_bars}")
    return fig


def _first_numeric(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    return df.columns[-1] if len(df.columns) else None


def _first_categorical(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return column
    return df.columns[0] if len(df.columns) else None
