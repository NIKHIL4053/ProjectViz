"""charts/line.py — Plotly line chart for trends over time or ordered categories."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts.theme import (
    COLORS,
    PLOTLY_TEMPLATE,
    get_plotly_sequence,
    style_plotly_figure,
)
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col = get_column(df, config.get("x_col")) or df.columns[0]
    y_col = get_column(df, config.get("y_col")) or next(
        (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x_col),
        df.columns[-1],
    )
    hue_col = get_column(df, config.get("hue_col"))
    title = config.get("title", f"{y_col} over {x_col}")

    if hue_col and hue_col not in df.columns:
        hue_col = None

    try:
        df = df.sort_values(by=x_col)
    except Exception:
        pass

    if hue_col:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            title=title,
            template=PLOTLY_TEMPLATE,
            markers=True,
            color_discrete_sequence=get_plotly_sequence(),
        )
    else:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template=PLOTLY_TEMPLATE,
            markers=True,
        )
        fig.update_traces(
            line_color=COLORS.ACCENT_SKY,
            marker_color=COLORS.ACCENT_PRIMARY,
            marker_size=7,
            line_width=3,
        )
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                fill="tozeroy",
                fillcolor="rgba(46,196,182,0.14)",
                line_color="rgba(0,0,0,0)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    style_plotly_figure(
        fig,
        title=title,
        height=460,
        xaxis_title=x_col,
        yaxis_title=y_col,
        x_tickangle=28,
    )
    fig.update_layout(margin={"l": 30, "r": 26, "t": 72, "b": 82})

    charts_log.info(f"[line] x={x_col} | y={y_col} | hue={hue_col} | rows={len(df)}")
    return fig
