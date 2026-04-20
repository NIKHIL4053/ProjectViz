"""charts/boxplot.py — Plotly box plot with outlier points."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col   = get_column(df, config.get("x_col"))
    y_col   = get_column(df, config.get("y_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title   = config.get("title", f"{y_col} by {x_col}")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x_col:
        x_col = cat_cols[0] if cat_cols else df.columns[0]
    if not y_col:
        y_col = num_cols[0] if num_cols else df.columns[-1]

    # * Ensure correct axis orientation
    if pd.api.types.is_numeric_dtype(df[x_col]) and not pd.api.types.is_numeric_dtype(df.get(y_col, pd.Series())):
        x_col, y_col = y_col, x_col

    if hue_col and hue_col not in df.columns:
        hue_col = None

    # * Limit categories to top 15 by median
    if df[x_col].nunique() > 15:
        try:
            top = df.groupby(x_col)[y_col].median().nlargest(15).index
            df  = df[df[x_col].isin(top)].copy()
        except Exception:
            pass

    if hue_col:
        fig = px.box(df, x=x_col, y=y_col, color=hue_col,
                     title=title, template=TEMPLATE, points="outliers")
    else:
        fig = px.box(df, x=x_col, y=y_col,
                     title=title, template=TEMPLATE, points="outliers",
                     color_discrete_sequence=["#89b4fa"])

    fig.update_layout(
        height=480, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font={"color": "#cdd6f4"}, title_font={"size": 15, "color": "#89b4fa"},
        xaxis_title=x_col, yaxis_title=y_col,
        margin={"l": 30, "r": 30, "t": 50, "b": 80},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cdd6f4"}},
    )
    fig.update_xaxes(tickangle=30, gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")
    charts_log.info(f"[boxplot] x={x_col} | y={y_col} | rows={len(df)}")
    return fig
