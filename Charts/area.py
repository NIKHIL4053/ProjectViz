"""charts/area.py — Plotly stacked area chart."""

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

    try:
        df = df.sort_values(by=x_col)
    except Exception:
        pass

    if hue_col and hue_col in df.columns:
        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            title=title,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=get_plotly_sequence(),
        )
    else:
        fig = px.area(df, x=x_col, y=y_col, title=title, template=PLOTLY_TEMPLATE)
        fig.update_traces(
            line_color=COLORS.ACCENT_SKY,
            fillcolor="rgba(93,169,233,0.32)",
        )

    style_plotly_figure(
        fig,
        title=title,
        height=450,
        xaxis_title=x_col,
        yaxis_title=y_col,
        x_tickangle=28,
    )
    fig.update_layout(margin={"l": 30, "r": 26, "t": 72, "b": 82})

    charts_log.info(f"[area] x={x_col} | y={y_col} | rows={len(df)}")
    return fig
