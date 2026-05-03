"""charts/kde.py — Plotly violin and distribution chart."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from charts.theme import COLORS, PLOTLY_TEMPLATE, get_plotly_sequence, style_plotly_figure
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col = get_column(df, config.get("x_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title = config.get("title", "Distribution")

    if not x_col or not pd.api.types.is_numeric_dtype(df[x_col]):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                x_col = column
                break

    if not x_col:
        raise ValueError(f"KDE needs a numeric column. Got: {list(df.columns)}")

    if hue_col and hue_col not in df.columns:
        hue_col = None

    plot_df = df[[x_col] + ([hue_col] if hue_col else [])].dropna()
    fig = go.Figure()
    palette = list(get_plotly_sequence())

    if hue_col:
        for i, group in enumerate(plot_df[hue_col].unique()):
            vals = plot_df[plot_df[hue_col] == group][x_col].values
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Violin(
                    x=vals,
                    name=str(group),
                    side="positive",
                    line_color=color,
                    fillcolor="rgba(46,196,182,0.18)",
                    meanline_visible=True,
                    showlegend=True,
                )
            )
    else:
        vals = plot_df[x_col].values
        fig.add_trace(
            go.Violin(
                x=vals,
                name=x_col,
                side="positive",
                line_color=COLORS.ACCENT_PRIMARY,
                fillcolor="rgba(46,196,182,0.26)",
                meanline_visible=True,
                showlegend=False,
            )
        )
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=COLORS.MEAN_LINE,
            annotation_text=f"Mean: {mean_val:,.1f}",
            annotation_font_color=COLORS.MEAN_LINE,
        )
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color=COLORS.TREND_LINE,
            annotation_text=f"Median: {median_val:,.1f}",
            annotation_font_color=COLORS.TREND_LINE,
            annotation_position="bottom right",
        )

    style_plotly_figure(
        fig,
        title=title,
        height=430,
        xaxis_title=x_col,
        yaxis_title="Density",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin={"l": 30, "r": 26, "t": 72, "b": 42})

    charts_log.info(f"[kde] x={x_col} | hue={hue_col} | rows={len(plot_df)}")
    return fig
