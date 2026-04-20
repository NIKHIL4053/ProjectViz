"""charts/line.py — Plotly line chart for trends over time or ordered categories."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col   = get_column(df, config.get("x_col")) or df.columns[0]
    y_col   = get_column(df, config.get("y_col")) or next(
                (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x_col),
                df.columns[-1])
    hue_col = get_column(df, config.get("hue_col"))
    title   = config.get("title", f"{y_col} over {x_col}")

    if hue_col and hue_col not in df.columns:
        hue_col = None

    try:
        df = df.sort_values(by=x_col)
    except Exception:
        pass

    if hue_col:
        fig = px.line(df, x=x_col, y=y_col, color=hue_col,
                      title=title, template=TEMPLATE, markers=True)
    else:
        fig = px.line(df, x=x_col, y=y_col,
                      title=title, template=TEMPLATE, markers=True)
        fig.update_traces(line_color="#89b4fa", marker_color="#89b4fa", marker_size=7)
        # * Subtle fill under line
        fig.add_traces(go.Scatter(
            x=df[x_col], y=df[y_col],
            fill="tozeroy",
            fillcolor="rgba(137,180,250,0.12)",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ))

    fig.update_layout(
        height        = 450,
        paper_bgcolor = "#1e1e2e",
        plot_bgcolor  = "#1e1e2e",
        font          = {"color": "#cdd6f4", "size": 12},
        title_font    = {"size": 15, "color": "#89b4fa"},
        xaxis_title   = x_col,
        yaxis_title   = y_col,
        margin        = {"l": 30, "r": 30, "t": 50, "b": 80},
        legend        = {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cdd6f4"}},
    )
    fig.update_xaxes(tickangle=30, gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    charts_log.info(f"[line] x={x_col} | y={y_col} | hue={hue_col} | rows={len(df)}")
    return fig
