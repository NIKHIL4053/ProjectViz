"""charts/scatter.py — Plotly scatter with trend line."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config import SCATTER_SAMPLE_ROWS
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col   = get_column(df, config.get("x_col"))
    y_col   = get_column(df, config.get("y_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title   = config.get("title", f"{x_col} vs {y_col}")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not x_col or x_col not in df.columns or not pd.api.types.is_numeric_dtype(df[x_col]):
        x_col = num_cols[0] if num_cols else df.columns[0]
    if not y_col or y_col not in df.columns or not pd.api.types.is_numeric_dtype(df[y_col]):
        y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    if hue_col and hue_col not in df.columns:
        hue_col = None

    size_col = next((c for c in num_cols if c not in (x_col, y_col)), None)
    sample   = df.sample(min(SCATTER_SAMPLE_ROWS, len(df)), random_state=42)

    if hue_col:
        fig = px.scatter(sample, x=x_col, y=y_col, color=hue_col,
                         size=size_col, size_max=30, title=title,
                         template=TEMPLATE, opacity=0.7,
                         color_continuous_scale="Viridis")
    else:
        fig = px.scatter(sample, x=x_col, y=y_col,
                         size=size_col, size_max=30, title=title,
                         template=TEMPLATE, opacity=0.7)
        fig.update_traces(marker_color="#89b4fa")

    # * Trend line
    try:
        clean = sample[[x_col, y_col]].dropna()
        if len(clean) > 5:
            m, b = np.polyfit(clean[x_col], clean[y_col], 1)
            xs   = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
            fig.add_trace(go.Scatter(
                x=xs, y=m*xs+b, mode="lines",
                line=dict(color="#f5c2e7", dash="dash", width=2),
                name="Trend", showlegend=True,
            ))
        corr = clean[x_col].corr(clean[y_col])
        fig.add_annotation(
            text=f"r = {corr:.2f}", xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            font={"color": "#6c7086", "size": 11},
        )
    except Exception:
        pass

    fig.update_layout(
        height=480, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font={"color": "#cdd6f4"}, title_font={"size": 15, "color": "#89b4fa"},
        xaxis_title=x_col, yaxis_title=y_col,
        margin={"l": 30, "r": 30, "t": 50, "b": 40},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cdd6f4"}},
    )
    charts_log.info(f"[scatter] x={x_col} | y={y_col} | sample={len(sample)}/{len(df)}")
    return fig
