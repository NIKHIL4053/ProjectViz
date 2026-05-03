"""charts/treemap.py — Plotly treemap for hierarchical portfolio data."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts.theme import PLOTLY_TEMPLATE, get_plotly_colorscale, style_plotly_figure
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    title = config.get("title", "Treemap")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not cat_cols or not num_cols:
        raise ValueError("Treemap needs at least 1 categorical and 1 numeric column")

    path_cols = cat_cols[:2]
    val_col = num_cols[0]
    color_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    fig = px.treemap(
        df,
        path=path_cols,
        values=val_col,
        color=color_col,
        title=title,
        template=PLOTLY_TEMPLATE,
        color_continuous_scale=get_plotly_colorscale("Blues"),
    )

    fig.update_traces(textinfo="label+value+percent entry", textfont_size=12)
    style_plotly_figure(fig, title=title, height=500, showlegend=False)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 72, "b": 12})

    charts_log.info(f"[treemap] path={path_cols} | val={val_col} | rows={len(df)}")
    return fig
