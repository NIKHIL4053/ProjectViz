"""charts/treemap.py — Plotly treemap for hierarchical portfolio data."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    title    = config.get("title", "Treemap")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not cat_cols or not num_cols:
        raise ValueError("Treemap needs at least 1 categorical and 1 numeric column")

    path_cols = cat_cols[:2]
    val_col   = num_cols[0]
    color_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    fig = px.treemap(
        df,
        path                   = path_cols,
        values                 = val_col,
        color                  = color_col,
        title                  = title,
        template               = TEMPLATE,
        color_continuous_scale = "Blues",
    )

    fig.update_traces(
        textinfo      = "label+value+percent entry",
        textfont_size = 12,
    )
    fig.update_layout(
        height        = 480,
        paper_bgcolor = "#1e1e2e",
        font          = {"color": "#cdd6f4"},
        title_font    = {"size": 15, "color": "#89b4fa"},
        margin        = {"l": 10, "r": 10, "t": 50, "b": 10},
    )
    charts_log.info(f"[treemap] path={path_cols} | val={val_col} | rows={len(df)}")
    return fig
