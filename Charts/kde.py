"""charts/kde.py — Plotly violin/distribution chart."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.helpers import get_column
from utils.logger import get_charts_logger

charts_log = get_charts_logger(__name__)
TEMPLATE   = "plotly_dark"
_COLORS    = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8", "#cba6f7"]


def render(df: pd.DataFrame, config: dict) -> go.Figure:
    x_col   = get_column(df, config.get("x_col"))
    hue_col = get_column(df, config.get("hue_col"))
    title   = config.get("title", "Distribution")

    if not x_col or not pd.api.types.is_numeric_dtype(df[x_col]):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                x_col = c
                break

    if not x_col:
        raise ValueError(f"KDE needs a numeric column. Got: {list(df.columns)}")

    if hue_col and hue_col not in df.columns:
        hue_col = None

    plot_df = df[[x_col] + ([hue_col] if hue_col else [])].dropna()
    fig     = go.Figure()

    if hue_col:
        for i, grp in enumerate(plot_df[hue_col].unique()):
            vals  = plot_df[plot_df[hue_col] == grp][x_col].values
            color = _COLORS[i % len(_COLORS)]
            fig.add_trace(go.Violin(
                x=vals, name=str(grp), side="positive",
                line_color=color,
                fillcolor=f"rgba(137,180,250,0.25)",
                meanline_visible=True, showlegend=True,
            ))
    else:
        vals = plot_df[x_col].values
        fig.add_trace(go.Violin(
            x=vals, name=x_col, side="positive",
            line_color="#89b4fa",
            fillcolor="rgba(137,180,250,0.35)",
            meanline_visible=True, showlegend=False,
        ))
        mean_val   = float(np.mean(vals))
        median_val = float(np.median(vals))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#f38ba8",
                      annotation_text=f"Mean: {mean_val:,.1f}",
                      annotation_font_color="#f38ba8")
        fig.add_vline(x=median_val, line_dash="dot", line_color="#cba6f7",
                      annotation_text=f"Median: {median_val:,.1f}",
                      annotation_font_color="#cba6f7", annotation_position="bottom right")

    fig.update_layout(
        title=title, height=420, template=TEMPLATE,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font={"color": "#cdd6f4"}, title_font={"size": 15, "color": "#89b4fa"},
        xaxis_title=x_col, yaxis_title="Density",
        margin={"l": 30, "r": 30, "t": 50, "b": 40},
    )
    charts_log.info(f"[kde] x={x_col} | hue={hue_col} | rows={len(plot_df)}")
    return fig
