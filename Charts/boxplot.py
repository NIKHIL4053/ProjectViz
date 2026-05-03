"""charts/boxplot.py — Plotly box plot with outlier points."""

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
    x_col = get_column(df, config.get("x_col"))
    y_col = get_column(df, config.get("y_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title = config.get("title", f"{y_col} by {x_col}")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x_col:
        x_col = cat_cols[0] if cat_cols else df.columns[0]
    if not y_col:
        y_col = num_cols[0] if num_cols else df.columns[-1]

    if pd.api.types.is_numeric_dtype(df[x_col]) and not pd.api.types.is_numeric_dtype(
        df.get(y_col, pd.Series())
    ):
        x_col, y_col = y_col, x_col

    if hue_col and hue_col not in df.columns:
        hue_col = None

    if df[x_col].nunique() > 15:
        try:
            top = df.groupby(x_col)[y_col].median().nlargest(15).index
            df = df[df[x_col].isin(top)].copy()
        except Exception:
            pass

    if hue_col:
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            title=title,
            template=PLOTLY_TEMPLATE,
            points="outliers",
            color_discrete_sequence=get_plotly_sequence(),
        )
    else:
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template=PLOTLY_TEMPLATE,
            points="outliers",
            color_discrete_sequence=[COLORS.ACCENT_PRIMARY],
        )

    style_plotly_figure(
        fig,
        title=title,
        height=490,
        xaxis_title=x_col,
        yaxis_title=y_col,
        x_tickangle=28,
    )
    fig.update_layout(margin={"l": 30, "r": 26, "t": 72, "b": 82})

    charts_log.info(f"[boxplot] x={x_col} | y={y_col} | rows={len(df)}")
    return fig
