"""charts/scatter.py — Plotly scatter with optional trend line."""

import numpy as np
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
from config import SCATTER_SAMPLE_ROWS
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col = get_column(df, config.get("x_col"))
    y_col = get_column(df, config.get("y_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title = config.get("title", f"{x_col} vs {y_col}")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x_col or x_col not in df.columns or not pd.api.types.is_numeric_dtype(df[x_col]):
        x_col = num_cols[0] if num_cols else df.columns[0]
    if not y_col or y_col not in df.columns or not pd.api.types.is_numeric_dtype(df[y_col]):
        y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    if hue_col and hue_col not in df.columns:
        hue_col = None

    size_col = next((c for c in num_cols if c not in (x_col, y_col)), None)
    sample = df.sample(min(SCATTER_SAMPLE_ROWS, len(df)), random_state=42)

    common_args = {
        "data_frame": sample,
        "x": x_col,
        "y": y_col,
        "size": size_col,
        "size_max": 30,
        "title": title,
        "template": PLOTLY_TEMPLATE,
        "opacity": 0.78,
    }

    if hue_col:
        if pd.api.types.is_numeric_dtype(sample[hue_col]):
            fig = px.scatter(
                **common_args,
                color=hue_col,
                color_continuous_scale=get_plotly_colorscale("viridis"),
            )
        else:
            fig = px.scatter(
                **common_args,
                color=hue_col,
                color_discrete_sequence=get_plotly_sequence(),
            )
    else:
        fig = px.scatter(**common_args)
        fig.update_traces(marker_color=COLORS.ACCENT_SKY)

    try:
        clean = sample[[x_col, y_col]].dropna()
        if len(clean) > 5:
            m, b = np.polyfit(clean[x_col], clean[y_col], 1)
            xs = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=m * xs + b,
                    mode="lines",
                    line=dict(color=COLORS.TREND_LINE, dash="dash", width=2),
                    name="Trend",
                    showlegend=True,
                )
            )
        corr = clean[x_col].corr(clean[y_col])
        fig.add_annotation(
            text=f"r = {corr:.2f}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font={"color": COLORS.TEXT_MUTED, "size": 11},
        )
    except Exception:
        pass

    style_plotly_figure(
        fig,
        title=title,
        height=500,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    fig.update_layout(margin={"l": 30, "r": 26, "t": 72, "b": 44})

    charts_log.info(f"[scatter] x={x_col} | y={y_col} | sample={len(sample)}/{len(df)}")
    return fig
