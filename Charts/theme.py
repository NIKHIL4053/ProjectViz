"""
charts/theme.py
---------------
# * Global dark theme for all Seaborn and Matplotlib charts.
# * Called ONCE at app startup — all charts automatically inherit it.
# * Never import matplotlib or seaborn directly in chart files —
# * always call apply_theme() first to ensure consistent styling.

# ? Design decisions:
# ? Dark background (#1e1e2e) matches the Streamlit dark UI
# ? so charts feel native to the dashboard, not pasted in.
# ? Catppuccin Mocha palette — professional, readable, not garish.

Exports:
    - apply_theme()          : Apply global theme — call at app startup
    - COLORS                 : Named color constants used across all charts
    - PALETTES               : Named palette lists for different chart needs
    - get_fig_size()         : Standard figure sizes by chart type
"""

import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# * Use non-interactive Agg backend — required for Streamlit
# ! Must be set before any plt calls
matplotlib.use("Agg")


# ──────────────────────────────────────────────────────────────────────────────
# * COLOR CONSTANTS — Catppuccin Mocha palette
# ──────────────────────────────────────────────────────────────────────────────

class COLORS:
    """# * Named color constants — use these instead of raw hex strings in charts."""

    # * Backgrounds
    BG_MAIN    = "#1e1e2e"    # * Main chart background
    BG_PANEL   = "#181825"    # * Panel / sidebar background
    BG_SURFACE = "#313244"    # * Surface elements

    # * Text
    TEXT_PRIMARY   = "#cdd6f4"   # * Main text, axis labels
    TEXT_SECONDARY = "#6c7086"   # * Secondary text, captions
    TEXT_ACCENT    = "#89b4fa"   # * Highlighted text, titles

    # * Grid and borders
    GRID   = "#313244"
    BORDER = "#45475a"

    # * Semantic colors — use these for business metrics
    GOOD   = "#a6e3a1"   # * Green — paid, resolved, good
    BAD    = "#f38ba8"   # * Red — bounced, NPA, bad
    WARN   = "#fab387"   # * Orange — pending, warning
    INFO   = "#89b4fa"   # * Blue — informational, neutral
    PURPLE = "#cba6f7"   # * Purple — trend lines, highlights

    # * Chart element colors
    TREND_LINE = "#f5c2e7"   # * Pink — regression/trend lines
    MEAN_LINE  = "#f38ba8"   # * Red — mean reference lines
    FILL_ALPHA = 0.12        # * Alpha for area fills

    # * KPI card colors
    KPI_POSITIVE = "#a6e3a1"
    KPI_NEGATIVE = "#f38ba8"
    KPI_NEUTRAL  = "#89b4fa"


# ──────────────────────────────────────────────────────────────────────────────
# * PALETTE CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class PALETTES:
    """
    # * Named palette sets for different chart contexts.
    # * Match these to the color_palette field in ChartConfig from chart_decider.py.
    """

    # * For bounce/NPA/loss data — red=bad, green=good
    RISK        = "RdYlGn_r"

    # * For neutral volume/count data
    BLUES       = "Blues"

    # * For categorical groupings (buckets, products, teams)
    CATEGORICAL = "Set2"

    # * For risk/alert metrics — warm gradient
    ALERT       = "flare"

    # * For density/distribution data
    DENSITY     = "viridis"

    # * For correlation/scatter — diverging
    DIVERGING   = "coolwarm"

    # * For overdue/loss amounts — severity scale
    SEVERITY    = "Reds"

    # * For many categories (FE names, TL names)
    MULTI       = "tab10"

    # * Custom semantic palette for EMI/bounce status
    STATUS = {
        "PAID":     COLORS.GOOD,
        "Paid":     COLORS.GOOD,
        "Tech":     COLORS.BAD,
        "Non Tech": COLORS.WARN,
        "Bounced":  COLORS.BAD,
    }

    # * Custom palette for bucket colors
    BUCKETS = {
        "Current":   COLORS.GOOD,
        "Risk X":    "#f9e2af",   # * Yellow
        "1-29 DPD":  COLORS.WARN,
        "30-59 DPD": "#fe640b",   # * Orange
        "60-89 DPD": "#e64553",   # * Dark red
        "NPA":       COLORS.BAD,
        "Write-off": "#585b70",   # * Grey
    }

    # * Custom palette for cust wise status
    STATUS_MOVEMENT = {
        "Current":   COLORS.GOOD,
        "Norm":      "#89dceb",   # * Teal
        "Flow":      COLORS.BAD,
        "Stab":      "#f9e2af",   # * Yellow
        "Roll Back": "#fab387",   # * Orange
        "Risk NPA":  "#fe640b",   # * Dark orange
        "NPA":       COLORS.BAD,
        "Write-off": "#585b70",   # * Grey
    }


# ──────────────────────────────────────────────────────────────────────────────
# * FIGURE SIZES
# ──────────────────────────────────────────────────────────────────────────────

class FIGSIZE:
    """# * Standard figure sizes by chart type."""
    LINE      = (13, 5)
    AREA      = (13, 5)
    HEATMAP   = (13, 7)    # * overridden dynamically based on pivot dimensions
    KDE       = (12, 5)
    SCATTER   = (12, 6)
    BOXPLOT   = (13, 6)
    WIDE      = (15, 6)
    SQUARE    = (10, 8)


# ──────────────────────────────────────────────────────────────────────────────
# * THEME APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

_theme_applied = False


def apply_theme():
    """
    # * Apply the global dark theme to all Seaborn and Matplotlib charts.
    # * Idempotent — safe to call multiple times, only applies once.
    # * Call this at app startup in app.py before any charts are rendered.
    """
    global _theme_applied
    if _theme_applied:
        return

    sns.set_theme(
        style     = "darkgrid",
        palette   = PALETTES.CATEGORICAL,
        font      = "DejaVu Sans",
        font_scale= 1.1,
        rc        = {
            # * Backgrounds
            "axes.facecolor":       COLORS.BG_MAIN,
            "figure.facecolor":     COLORS.BG_MAIN,
            "savefig.facecolor":    COLORS.BG_MAIN,

            # * Text
            "axes.labelcolor":      COLORS.TEXT_PRIMARY,
            "axes.titlecolor":      COLORS.TEXT_PRIMARY,
            "xtick.color":          COLORS.TEXT_PRIMARY,
            "ytick.color":          COLORS.TEXT_PRIMARY,
            "text.color":           COLORS.TEXT_PRIMARY,
            "legend.labelcolor":    COLORS.TEXT_PRIMARY,

            # * Grid and borders
            "grid.color":           COLORS.GRID,
            "grid.alpha":           0.4,
            "axes.edgecolor":       COLORS.BORDER,
            "axes.spines.top":      False,
            "axes.spines.right":    False,

            # * Figure
            "figure.dpi":           100,
            "figure.autolayout":    False,

            # * Font sizes
            "axes.titlesize":       14,
            "axes.labelsize":       11,
            "xtick.labelsize":      10,
            "ytick.labelsize":      10,
            "legend.fontsize":      10,
            "legend.title_fontsize":10,
        }
    )

    _theme_applied = True


def reset_theme():
    """
    # * Reset theme flag — forces re-application on next apply_theme() call.
    # ? Useful in testing or if theme needs to be changed mid-session.
    """
    global _theme_applied
    _theme_applied = False


def style_figure(fig: plt.Figure, title: str = "", subtitle: str = "") -> plt.Figure:
    """
    # * Apply consistent title styling to a figure.
    # * Called at the end of every chart function.

    Args:
        fig      : matplotlib Figure to style
        title    : Main chart title
        subtitle : Optional subtitle (shown smaller below title)

    Returns:
        Styled Figure.
    """
    if title:
        fig.suptitle(
            title,
            fontsize   = 14,
            fontweight = "bold",
            color      = COLORS.TEXT_ACCENT,
            y          = 1.02,
        )

    if subtitle:
        fig.text(
            0.5, -0.02,
            subtitle,
            ha        = "center",
            fontsize  = 9,
            color     = COLORS.TEXT_SECONDARY,
            style     = "italic",
        )

    plt.tight_layout()
    return fig