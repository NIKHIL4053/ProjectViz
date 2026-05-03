"""
charts/theme.py
---------------
# * Shared visual system for the dashboard.
# * Applies Seaborn defaults, registers the Plotly template,
# * and injects Streamlit CSS for the app shell.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# * Use non-interactive Agg backend — required for Streamlit
matplotlib.use("Agg")

PLOTLY_TEMPLATE = "loan_dashboard_midnight"


class COLORS:
    """# * Shared dashboard colors."""

    BG_APP        = "#08131f"
    BG_MAIN       = BG_APP
    BG_PANEL      = "#0f2233"
    BG_SURFACE    = "#10293c"
    BG_ELEVATED   = "#14344b"

    TEXT_PRIMARY   = "#edf6fb"
    TEXT_SECONDARY = "#9bb6c9"
    TEXT_MUTED     = "#6f8ea3"
    TEXT_ACCENT    = "#2ec4b6"

    BORDER      = "#39586c"
    BORDER_BOLD = "#5d9fd0"
    GRID        = "#294357"

    ACCENT_PRIMARY   = "#2ec4b6"
    ACCENT_SECONDARY = "#f4b860"
    ACCENT_TERTIARY  = "#ff7a59"
    ACCENT_SKY       = "#5da9e9"

    GOOD   = "#38d39f"
    BAD    = "#ff7a59"
    WARN   = "#f4b860"
    INFO   = "#5da9e9"
    PURPLE = "#7b8cff"

    TREND_LINE = "#ffd166"
    MEAN_LINE  = "#ff7a59"
    FILL_ALPHA = 0.18

    KPI_POSITIVE = GOOD
    KPI_NEGATIVE = BAD
    KPI_NEUTRAL  = ACCENT_SKY


class PALETTES:
    """# * Shared palette groups."""

    RISK = [
        COLORS.GOOD,
        COLORS.ACCENT_PRIMARY,
        COLORS.ACCENT_SECONDARY,
        COLORS.ACCENT_TERTIARY,
    ]
    BLUES = [
        "#1d4461",
        COLORS.ACCENT_SKY,
        "#9fd2ff",
        "#d9f0ff",
    ]
    CATEGORICAL = [
        COLORS.ACCENT_PRIMARY,
        COLORS.ACCENT_SKY,
        COLORS.ACCENT_SECONDARY,
        COLORS.ACCENT_TERTIARY,
        COLORS.GOOD,
        "#7b8cff",
        "#84dcc6",
        "#ffc69a",
    ]
    ALERT = [
        "#6b2f24",
        COLORS.ACCENT_TERTIARY,
        COLORS.ACCENT_SECONDARY,
    ]
    DENSITY = [
        "#244e70",
        COLORS.ACCENT_SKY,
        COLORS.ACCENT_PRIMARY,
        COLORS.GOOD,
    ]
    DIVERGING = [
        COLORS.ACCENT_TERTIARY,
        COLORS.ACCENT_SECONDARY,
        COLORS.ACCENT_SKY,
        COLORS.ACCENT_PRIMARY,
    ]
    SEVERITY = [
        "#5b261f",
        COLORS.ACCENT_TERTIARY,
        COLORS.ACCENT_SECONDARY,
    ]
    MULTI = CATEGORICAL

    STATUS = {
        "PAID":     COLORS.GOOD,
        "Paid":     COLORS.GOOD,
        "Tech":     COLORS.BAD,
        "Non Tech": COLORS.WARN,
        "Bounced":  COLORS.BAD,
    }

    BUCKETS = {
        "Current":   COLORS.GOOD,
        "Risk X":    "#bfe0a4",
        "1-29 DPD":  COLORS.ACCENT_SECONDARY,
        "30-59 DPD": "#ff9b54",
        "60-89 DPD": COLORS.ACCENT_TERTIARY,
        "NPA":       COLORS.BAD,
        "Write-off": "#60798b",
    }

    STATUS_MOVEMENT = {
        "Current":   COLORS.GOOD,
        "Norm":      COLORS.ACCENT_PRIMARY,
        "Flow":      COLORS.BAD,
        "Stab":      COLORS.ACCENT_SECONDARY,
        "Roll Back": "#f2a65a",
        "Risk NPA":  COLORS.ACCENT_TERTIARY,
        "NPA":       COLORS.BAD,
        "Write-off": "#60798b",
    }


class FIGSIZE:
    """# * Standard figure sizes by chart type."""

    LINE    = (13, 5)
    AREA    = (13, 5)
    HEATMAP = (13, 7)
    KDE     = (12, 5)
    SCATTER = (12, 6)
    BOXPLOT = (13, 6)
    WIDE    = (15, 6)
    SQUARE  = (10, 8)


_theme_applied = False

_PLOTLY_SCALES = {
    "teal": PALETTES.BLUES,
    "blues": PALETTES.BLUES,
    "reds": PALETTES.SEVERITY,
    "greens": [COLORS.ACCENT_PRIMARY, COLORS.GOOD, "#d6f5ea"],
    "oranges": [COLORS.ACCENT_TERTIARY, COLORS.ACCENT_SECONDARY, "#ffe3bf"],
    "rdylgn_r": PALETTES.RISK,
    "rdylgn": [COLORS.ACCENT_TERTIARY, COLORS.ACCENT_SECONDARY, COLORS.GOOD],
    "viridis": PALETTES.DENSITY,
    "flare": PALETTES.ALERT,
    "coolwarm": PALETTES.DIVERGING,
    "set2": PALETTES.CATEGORICAL,
    "tab10": PALETTES.MULTI,
}


def _register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return

    template = go.layout.Template(
        layout=go.Layout(
            colorway=PALETTES.CATEGORICAL,
            paper_bgcolor=COLORS.BG_APP,
            plot_bgcolor=COLORS.BG_SURFACE,
            font={
                "family": "Trebuchet MS, Segoe UI, sans-serif",
                "color": COLORS.TEXT_PRIMARY,
                "size": 12,
            },
            title={
                "x": 0.02,
                "xanchor": "left",
                "font": {
                    "family": "Georgia, Cambria, Times New Roman, serif",
                    "size": 20,
                    "color": COLORS.TEXT_PRIMARY,
                },
            },
            legend={
                "bgcolor": "rgba(0, 0, 0, 0)",
                "bordercolor": COLORS.BORDER,
                "borderwidth": 0,
                "font": {"color": COLORS.TEXT_PRIMARY},
                "title": {"font": {"color": COLORS.TEXT_SECONDARY}},
            },
            hoverlabel={
                "bgcolor": COLORS.BG_PANEL,
                "bordercolor": COLORS.ACCENT_PRIMARY,
                "font": {
                    "family": "Trebuchet MS, Segoe UI, sans-serif",
                    "color": COLORS.TEXT_PRIMARY,
                    "size": 12,
                },
            },
            margin={"l": 28, "r": 22, "t": 72, "b": 42},
            coloraxis={
                "colorbar": {
                    "outlinewidth": 0,
                    "tickcolor": COLORS.TEXT_SECONDARY,
                    "tickfont": {"color": COLORS.TEXT_SECONDARY},
                    "title": {"font": {"color": COLORS.TEXT_SECONDARY}},
                }
            },
            xaxis={
                "showgrid": True,
                "gridcolor": COLORS.GRID,
                "gridwidth": 1,
                "linecolor": COLORS.BORDER_BOLD,
                "showline": True,
                "tickfont": {"color": COLORS.TEXT_SECONDARY},
                "title": {"font": {"color": COLORS.TEXT_PRIMARY}},
                "zeroline": False,
            },
            yaxis={
                "showgrid": True,
                "gridcolor": COLORS.GRID,
                "gridwidth": 1,
                "linecolor": COLORS.BORDER_BOLD,
                "showline": True,
                "tickfont": {"color": COLORS.TEXT_SECONDARY},
                "title": {"font": {"color": COLORS.TEXT_PRIMARY}},
                "zeroline": False,
            },
        )
    )
    pio.templates[PLOTLY_TEMPLATE] = template


def apply_theme():
    """
    # * Apply the shared Seaborn and Plotly theme.
    # * Idempotent — safe to call multiple times.
    """
    global _theme_applied
    if _theme_applied:
        return

    sns.set_theme(
        style="darkgrid",
        palette=PALETTES.CATEGORICAL,
        font="Trebuchet MS",
        font_scale=1.05,
        rc={
            "axes.facecolor":    COLORS.BG_MAIN,
            "figure.facecolor":  COLORS.BG_MAIN,
            "savefig.facecolor": COLORS.BG_MAIN,
            "axes.labelcolor":   COLORS.TEXT_PRIMARY,
            "axes.titlecolor":   COLORS.TEXT_PRIMARY,
            "xtick.color":       COLORS.TEXT_SECONDARY,
            "ytick.color":       COLORS.TEXT_SECONDARY,
            "text.color":        COLORS.TEXT_PRIMARY,
            "legend.labelcolor": COLORS.TEXT_PRIMARY,
            "grid.color":        COLORS.GRID,
            "grid.alpha":        0.5,
            "axes.edgecolor":    COLORS.BORDER_BOLD,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.dpi":        100,
            "axes.titlesize":    14,
            "axes.labelsize":    11,
            "xtick.labelsize":   10,
            "ytick.labelsize":   10,
            "legend.fontsize":   10,
        },
    )

    _register_plotly_template()
    pio.templates.default = PLOTLY_TEMPLATE
    _theme_applied = True


def inject_app_css():
    """# * Inject Streamlit CSS once per rerun."""
    st.markdown(
        f"""
        <style>
            :root {{
                --ld-bg: {COLORS.BG_APP};
                --ld-panel: {COLORS.BG_PANEL};
                --ld-surface: {COLORS.BG_SURFACE};
                --ld-elevated: {COLORS.BG_ELEVATED};
                --ld-text: {COLORS.TEXT_PRIMARY};
                --ld-text-muted: {COLORS.TEXT_SECONDARY};
                --ld-text-soft: {COLORS.TEXT_MUTED};
                --ld-border: {COLORS.BORDER};
                --ld-border-bold: {COLORS.BORDER_BOLD};
                --ld-grid: {COLORS.GRID};
                --ld-accent: {COLORS.ACCENT_PRIMARY};
                --ld-accent-2: {COLORS.ACCENT_SECONDARY};
                --ld-accent-3: {COLORS.ACCENT_TERTIARY};
                --ld-sky: {COLORS.ACCENT_SKY};
                --ld-good: {COLORS.GOOD};
                --ld-warn: {COLORS.WARN};
                --ld-bad: {COLORS.BAD};
                --ld-shadow: 0 22px 48px rgba(0, 0, 0, 0.22);
            }}

            @keyframes ldRise {{
                from {{
                    opacity: 0;
                    transform: translateY(8px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
                color: var(--ld-text);
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }}

            [data-testid="stAppViewContainer"] {{
                background:
                    radial-gradient(circle at 0% 0%, rgba(46, 196, 182, 0.18), transparent 24%),
                    radial-gradient(circle at 85% 6%, rgba(244, 184, 96, 0.15), transparent 18%),
                    linear-gradient(180deg, #08131f 0%, #0c1c2a 40%, #09131c 100%);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            .block-container {{
                max-width: 1440px;
                padding-top: 1.35rem;
                padding-bottom: 3rem;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(14, 31, 46, 0.97) 0%, rgba(8, 18, 28, 0.99) 100%);
                border-right: 1px solid var(--ld-border);
            }}

            [data-testid="stSidebarContent"] {{
                padding-top: 1rem;
            }}

            h1, h2, h3, h4, h5, h6 {{
                font-family: Georgia, Cambria, "Times New Roman", serif;
                color: var(--ld-text);
                letter-spacing: -0.02em;
            }}

            p, li, label, span, div {{
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }}

            .stMarkdown p,
            .stCaptionContainer,
            small {{
                color: var(--ld-text-muted);
            }}

            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 999px;
                border: 1px solid rgba(93, 169, 233, 0.24);
                background: linear-gradient(135deg, rgba(46, 196, 182, 0.18) 0%, rgba(93, 169, 233, 0.2) 100%);
                color: var(--ld-text);
                min-height: 2.8rem;
                font-weight: 600;
                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
                transition: all 0.18s ease;
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                border-color: rgba(244, 184, 96, 0.4);
                transform: translateY(-1px);
                box-shadow: 0 14px 26px rgba(0, 0, 0, 0.2);
            }}

            .stButton > button[kind="primary"] {{
                background: linear-gradient(135deg, rgba(46, 196, 182, 0.95) 0%, rgba(93, 169, 233, 0.88) 100%);
                color: #06202f;
                border-color: transparent;
            }}

            [data-baseweb="input"] > div,
            [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"] > div,
            .stTextArea textarea {{
                background: rgba(12, 31, 46, 0.72);
                border: 1px solid var(--ld-border);
                border-radius: 16px;
                color: var(--ld-text);
            }}

            [data-baseweb="input"] input {{
                color: var(--ld-text);
            }}

            .stCheckbox label {{
                color: var(--ld-text-muted);
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.45rem;
                background: rgba(10, 26, 39, 0.74);
                padding: 0.35rem;
                border: 1px solid var(--ld-border);
                border-radius: 18px;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 14px;
                color: var(--ld-text-muted);
                font-weight: 700;
                padding: 0.55rem 1rem;
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, rgba(46, 196, 182, 0.2) 0%, rgba(93, 169, 233, 0.22) 100%);
                color: var(--ld-text) !important;
                border: 1px solid rgba(46, 196, 182, 0.24);
            }}

            div[data-testid="stMetric"] {{
                background: linear-gradient(180deg, rgba(14, 37, 55, 0.92) 0%, rgba(11, 27, 39, 0.92) 100%);
                border: 1px solid var(--ld-border);
                border-radius: 22px;
                padding: 0.95rem 1.1rem;
                box-shadow: var(--ld-shadow);
                animation: ldRise 0.28s ease;
            }}

            div[data-testid="stMetricLabel"] {{
                color: var(--ld-text-muted);
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--ld-text);
            }}

            div[data-testid="stMetricDelta"] {{
                color: var(--ld-accent);
            }}

            div[data-testid="stExpander"] {{
                border: 1px solid var(--ld-border);
                border-radius: 20px;
                background: rgba(12, 31, 46, 0.62);
                overflow: hidden;
            }}

            div[data-testid="stExpander"] details > summary {{
                background: linear-gradient(90deg, rgba(46, 196, 182, 0.08) 0%, rgba(93, 169, 233, 0.08) 100%);
            }}

            [data-testid="stAlert"] {{
                background: rgba(11, 28, 42, 0.78);
                border: 1px solid var(--ld-border);
                border-radius: 18px;
                color: var(--ld-text);
            }}

            [data-testid="stStatusWidget"] {{
                border: 1px solid var(--ld-border);
                border-radius: 18px;
                background: rgba(12, 31, 46, 0.62);
            }}

            [data-testid="stDataFrame"],
            .stCodeBlock {{
                border: 1px solid var(--ld-border);
                border-radius: 20px;
                overflow: hidden;
                box-shadow: var(--ld-shadow);
            }}

            .ld-hero,
            .ld-card,
            .ld-sidebar-card,
            .ld-note-card,
            .ld-insight-card,
            .ld-history-item {{
                animation: ldRise 0.3s ease;
            }}

            .ld-hero {{
                position: relative;
                overflow: hidden;
                padding: 1.65rem 1.75rem;
                border-radius: 30px;
                background: linear-gradient(135deg, rgba(14, 38, 57, 0.96) 0%, rgba(19, 54, 77, 0.94) 45%, rgba(14, 26, 39, 0.96) 100%);
                border: 1px solid rgba(93, 169, 233, 0.24);
                box-shadow: 0 28px 60px rgba(0, 0, 0, 0.28);
                margin-bottom: 1rem;
            }}

            .ld-hero::before {{
                content: "";
                position: absolute;
                inset: auto -10% -55% 45%;
                height: 18rem;
                background: radial-gradient(circle, rgba(46, 196, 182, 0.28), transparent 58%);
                pointer-events: none;
            }}

            .ld-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.38rem 0.78rem;
                border-radius: 999px;
                background: rgba(46, 196, 182, 0.12);
                border: 1px solid rgba(46, 196, 182, 0.28);
                color: var(--ld-text);
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.72rem;
                font-weight: 700;
            }}

            .ld-hero h1 {{
                margin: 0.55rem 0 0.4rem 0;
                font-size: clamp(1.95rem, 3vw, 3.1rem);
                line-height: 1.04;
            }}

            .ld-hero p {{
                max-width: 54rem;
                margin: 0;
                font-size: 1rem;
                line-height: 1.72;
                color: var(--ld-text-muted);
            }}

            .ld-chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin-top: 1rem;
            }}

            .ld-chip {{
                min-width: 10rem;
                padding: 0.72rem 0.92rem;
                border-radius: 18px;
                background: rgba(7, 18, 28, 0.38);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}

            .ld-chip small {{
                display: block;
                margin-bottom: 0.2rem;
                color: var(--ld-text-soft);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.7rem;
            }}

            .ld-chip strong {{
                color: var(--ld-text);
                font-size: 0.96rem;
            }}

            .ld-card {{
                height: 100%;
                padding: 1.15rem 1.25rem;
                border-radius: 22px;
                background: linear-gradient(180deg, rgba(13, 34, 49, 0.92) 0%, rgba(11, 27, 39, 0.92) 100%);
                border: 1px solid var(--ld-border);
                box-shadow: var(--ld-shadow);
            }}

            .ld-card h3,
            .ld-note-card strong,
            .ld-sidebar-card strong {{
                display: block;
                margin: 0 0 0.45rem 0;
                color: var(--ld-text);
            }}

            .ld-card p,
            .ld-card li,
            .ld-note-card span,
            .ld-sidebar-card span {{
                color: var(--ld-text-muted);
                line-height: 1.65;
            }}

            .ld-card ul {{
                margin: 0;
                padding-left: 1.1rem;
            }}

            .ld-note-card,
            .ld-sidebar-card,
            .ld-insight-card,
            .ld-history-item {{
                border-radius: 20px;
                padding: 0.95rem 1rem;
                border: 1px solid var(--ld-border);
                background: linear-gradient(180deg, rgba(12, 31, 46, 0.82) 0%, rgba(10, 23, 34, 0.82) 100%);
                box-shadow: var(--ld-shadow);
            }}

            .ld-note-card {{
                border-left: 3px solid rgba(46, 196, 182, 0.72);
                margin-bottom: 0.85rem;
            }}

            .ld-sidebar-card {{
                margin-bottom: 0.85rem;
            }}

            .ld-status-row {{
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                padding: 0.72rem 0;
                border-top: 1px solid rgba(255, 255, 255, 0.05);
            }}

            .ld-status-row:first-child {{
                border-top: none;
                padding-top: 0.1rem;
            }}

            .ld-status-dot {{
                width: 0.72rem;
                height: 0.72rem;
                border-radius: 999px;
                margin-top: 0.25rem;
                flex: 0 0 auto;
            }}

            .ld-status-dot.ok {{
                background: var(--ld-good);
                box-shadow: 0 0 0 6px rgba(56, 211, 159, 0.12);
            }}

            .ld-status-dot.warn {{
                background: var(--ld-warn);
                box-shadow: 0 0 0 6px rgba(244, 184, 96, 0.12);
            }}

            .ld-status-dot.bad {{
                background: var(--ld-bad);
                box-shadow: 0 0 0 6px rgba(255, 122, 89, 0.12);
            }}

            .ld-status-title {{
                color: var(--ld-text);
                font-weight: 700;
            }}

            .ld-status-detail {{
                display: block;
                margin-top: 0.2rem;
                color: var(--ld-text-muted);
                font-size: 0.88rem;
            }}

            .ld-insight-card {{
                border-left: 3px solid rgba(93, 169, 233, 0.76);
                margin-bottom: 0.75rem;
            }}

            .ld-history-item {{
                margin-bottom: 0.8rem;
            }}

            .ld-history-item small {{
                color: var(--ld-text-soft);
            }}

            @media (max-width: 900px) {{
                .block-container {{
                    padding-top: 1rem;
                }}

                .ld-hero {{
                    padding: 1.2rem 1.1rem;
                    border-radius: 24px;
                }}

                .ld-chip {{
                    min-width: 100%;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_theme():
    """# * Reset theme flags — useful in testing."""
    global _theme_applied
    _theme_applied = False


def get_plotly_colorscale(name: Optional[str]) -> Sequence[str]:
    key = (name or "").strip().lower()
    return _PLOTLY_SCALES.get(key, PALETTES.BLUES)


def get_plotly_sequence(name: Optional[str] = None) -> Sequence[str]:
    key = (name or "").strip().lower()
    if key in {"set2", "tab10"}:
        return _PLOTLY_SCALES[key]
    return PALETTES.CATEGORICAL


def style_plotly_figure(
    fig: go.Figure,
    *,
    title: Optional[str] = None,
    height: Optional[int] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    showlegend: Optional[bool] = None,
    x_tickangle: Optional[int] = None,
) -> go.Figure:
    """# * Apply the shared Plotly layout polish to a figure."""
    apply_theme()

    layout = {
        "template": PLOTLY_TEMPLATE,
        "paper_bgcolor": COLORS.BG_APP,
        "plot_bgcolor": COLORS.BG_SURFACE,
        "font": {
            "family": "Trebuchet MS, Segoe UI, sans-serif",
            "color": COLORS.TEXT_PRIMARY,
            "size": 12,
        },
        "title_font": {
            "family": "Georgia, Cambria, Times New Roman, serif",
            "size": 20,
            "color": COLORS.TEXT_PRIMARY,
        },
        "legend": {
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"color": COLORS.TEXT_PRIMARY},
            "title": {"font": {"color": COLORS.TEXT_SECONDARY}},
        },
        "margin": {"l": 28, "r": 22, "t": 72, "b": 42},
    }

    if title is not None:
        layout["title"] = title
    if height is not None:
        layout["height"] = height
    if xaxis_title is not None:
        layout["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        layout["yaxis_title"] = yaxis_title
    if showlegend is not None:
        layout["showlegend"] = showlegend

    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS.GRID,
        linecolor=COLORS.BORDER_BOLD,
        showline=True,
        tickfont={"color": COLORS.TEXT_SECONDARY},
        title_font={"color": COLORS.TEXT_PRIMARY},
        zeroline=False,
        tickangle=x_tickangle if x_tickangle is not None else None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS.GRID,
        linecolor=COLORS.BORDER_BOLD,
        showline=True,
        tickfont={"color": COLORS.TEXT_SECONDARY},
        title_font={"color": COLORS.TEXT_PRIMARY},
        zeroline=False,
    )
    return fig


def style_figure(fig: plt.Figure, title: str = "", subtitle: str = "") -> plt.Figure:
    """# * Style Matplotlib figures for any legacy uses."""
    if title:
        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
            color=COLORS.TEXT_PRIMARY,
            y=1.02,
        )

    if subtitle:
        fig.text(
            0.5,
            -0.02,
            subtitle,
            ha="center",
            fontsize=9,
            color=COLORS.TEXT_SECONDARY,
            style="italic",
        )

    plt.tight_layout()
    return fig
