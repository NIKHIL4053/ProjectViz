"""charts/heatmap.py — Plotly heatmap for two categorical dimensions vs numeric."""

import pandas as pd
import plotly.graph_objects as go

from charts.theme import PLOTLY_TEMPLATE, get_plotly_colorscale, style_plotly_figure
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col = get_column(df, config.get("x_col"))
    y_col = get_column(df, config.get("y_col"))
    val_col = get_column(df, config.get("hue_col"))
    palette = config.get("palette", "Blues")
    title = config.get("title", "Heatmap")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x_col and len(cat_cols) >= 1:
        x_col = cat_cols[0]
    if not y_col and len(cat_cols) >= 2:
        y_col = cat_cols[1]
    elif not y_col:
        y_col = x_col
    if not val_col and num_cols:
        val_col = next((c for c in num_cols if c not in (x_col, y_col)), num_cols[0])

    if not val_col or not x_col:
        raise ValueError(f"Heatmap needs 2 categorical + 1 numeric. Got: {list(df.columns)}")

    try:
        pivot = df.pivot_table(values=val_col, index=y_col, columns=x_col, aggfunc="sum").fillna(0)
    except Exception as exc:
        raise ValueError(f"Could not pivot: {exc}") from exc

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale=get_plotly_colorscale(palette),
            text=pivot.values.round(0),
            texttemplate="%{text:,.0f}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    n_rows = len(pivot.index)
    height = max(360, min(n_rows * 36 + 120, 820))

    style_plotly_figure(
        fig,
        title=title,
        height=height,
        xaxis_title=x_col,
        yaxis_title=y_col,
        x_tickangle=28,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin={"l": 18, "r": 18, "t": 72, "b": 82})

    charts_log.info(f"[heatmap] x={x_col} | y={y_col} | val={val_col} | pivot={pivot.shape}")
    return fig
