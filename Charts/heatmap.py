"""charts/heatmap.py — Plotly heatmap for two categorical dimensions vs numeric."""
import pandas as pd
import plotly.graph_objects as go
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"

_PALETTE_MAP = {
    "RdYlGn_r": "RdYlGn_r", "RdYlGn": "RdYlGn",
    "Blues": "Blues", "Reds": "Reds",
    "Greens": "Greens", "Oranges": "Oranges",
    "teal": "Teal",
}


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col   = get_column(df, config.get("x_col"))
    y_col   = get_column(df, config.get("y_col"))
    val_col = get_column(df, config.get("hue_col"))
    palette = config.get("palette", "Blues")
    title   = config.get("title", "Heatmap")

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
        pivot = df.pivot_table(
            values=val_col, index=y_col,
            columns=x_col, aggfunc="sum"
        ).fillna(0)
    except Exception as e:
        raise ValueError(f"Could not pivot: {e}")

    colorscale = _PALETTE_MAP.get(palette, "Blues")

    fig = go.Figure(data=go.Heatmap(
        z            = pivot.values,
        x            = [str(c) for c in pivot.columns],
        y            = [str(r) for r in pivot.index],
        colorscale   = colorscale,
        text         = pivot.values.round(0).astype(int),
        texttemplate = "%{text:,}",
        textfont     = {"size": 10},
        hoverongaps  = False,
    ))

    n_rows = len(pivot.index)
    height = max(350, min(n_rows * 35 + 120, 800))

    fig.update_layout(
        title         = title,
        height        = height,
        template      = TEMPLATE,
        paper_bgcolor = "#1e1e2e",
        plot_bgcolor  = "#1e1e2e",
        font          = {"color": "#cdd6f4"},
        title_font    = {"size": 15, "color": "#89b4fa"},
        xaxis         = {"title": x_col, "tickangle": 30},
        yaxis         = {"title": y_col},
        margin        = {"l": 10, "r": 10, "t": 50, "b": 80},
    )
    charts_log.info(f"[heatmap] x={x_col} | y={y_col} | val={val_col} | pivot={pivot.shape}")
    return fig
