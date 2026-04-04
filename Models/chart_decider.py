"""
models/chart_decider.py
-----------------------
# * Pipeline Step 4 — Chart Type Decision.
# * Qwen 7B looks at the DataFrame shape and user question
# * and decides the best chart type, axes, palette and title.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from models.analyzer import IntentResult
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import parse_json_safely, truncate_string

log       = get_logger(__name__)
model_log = get_model_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * CHART CONFIG DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartConfig:
    """
    # * Structured chart configuration returned to the renderer.

    Attributes:
        chart_type : line | area | heatmap | kde | scatter | boxplot
        x_col      : Column name for x-axis
        y_col      : Column name for y-axis
        hue_col    : Optional grouping column
        palette    : Seaborn color palette name
        title      : Chart title string
    """
    chart_type: str          = "heatmap"
    x_col:      str          = ""
    y_col:      str          = ""
    hue_col:    Optional[str]= None
    palette:    str          = "Set2"
    title:      str          = ""

    def to_dict(self) -> dict:
        return {
            "chart_type":    self.chart_type,
            "x_col":         self.x_col,
            "y_col":         self.y_col,
            "hue_col":       self.hue_col,
            "palette":       self.palette,
            "title":         self.title,
        }


# ──────────────────────────────────────────────────────────────────────────────
# * SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a data visualization expert for a loan collection portfolio dashboard.
Given the user question, available DataFrame columns, and sample data,
decide the best chart type and axis configuration.

CHART RULES — NEVER use bar chart:
- Date/time column + numeric → "line"
- Two categorical columns + numeric → "heatmap"  
- One numeric column, show distribution → "kde"
- Two numeric columns, show relationship → "scatter"
- One categorical + one numeric, show spread → "boxplot"
- Time + multiple categories stacked → "area"

COLOR PALETTE GUIDE:
- "RdYlGn_r"  → bounce, NPA, loss data (red=bad, green=good)
- "Blues"     → count/volume data (neutral)
- "Set2"      → categorical groupings
- "flare"     → risk/alert metrics
- "coolwarm"  → scatter/correlation
- "Reds"      → overdue/loss amounts

Return ONLY valid JSON — no explanation, no markdown:
{
  "chart_type": "heatmap",
  "x_col": "COLUMN_NAME",
  "y_col": "COLUMN_NAME",
  "hue_col": "COLUMN_NAME or null",
  "palette": "Set2",
  "title": "Chart Title"
}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# * CHART DECIDER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ChartDecider:
    """
    # * Step 4 of the pipeline — decides chart type from DataFrame + intent.
    # * Calls Qwen 7B (fast model) for lightweight classification.
    # * Falls back to rule-based decision if model fails.
    """

    def __init__(self):
        self._client = get_client()
        log.info("[chart_decider] Initialised")

    def decide(
        self,
        df:             pd.DataFrame,
        intent:         IntentResult,
        filter_summary: str = "",
    ) -> ChartConfig:
        """
        # * Decide the best chart configuration for the given data and intent.

        Args:
            df             : DataFrame from database client
            intent         : IntentResult from analyzer
            filter_summary : Human-readable filter string for title

        Returns:
            ChartConfig with all fields populated.
        """
        if df is None or df.empty:
            return self._fallback(df, intent, filter_summary)

        with benchmark("chart_decision", query=intent.raw_question):

            # * Build user message
            cols   = list(df.columns)
            sample = df.head(3).to_dict(orient="records")

            user_msg = f"""
User question: "{intent.raw_question}"
Metric: {intent.metric}
DataFrame columns: {cols}
Sample rows: {sample}
Filters applied: {filter_summary or 'None'}

Return the chart config JSON now.
""".strip()

            response: ModelResponse = self._client.call_fast(
                system      = _SYSTEM_PROMPT,
                user        = user_msg,
                step        = "chart_decision",
                expect_json = True,
            )

            if response.failed or not response.parsed:
                log.warning("[chart_decider] Model failed — using rule-based fallback")
                return self._fallback(df, intent, filter_summary)

            return self._parse_config(response.parsed, df, intent, filter_summary)

    def _parse_config(
        self,
        parsed:         dict,
        df:             pd.DataFrame,
        intent:         IntentResult,
        filter_summary: str,
    ) -> ChartConfig:
        """
        # * Parse and validate Qwen's JSON response into a ChartConfig.
        # * Validates all column names exist in df — falls back if not found.
        """
        chart_type = str(parsed.get("chart_type", "heatmap")).lower().strip()
        x_col      = parsed.get("x_col", "")
        y_col      = parsed.get("y_col", "")
        hue_col    = parsed.get("hue_col")
        palette    = str(parsed.get("palette", "Set2"))
        title      = str(parsed.get("title", intent.metric))

        # * Validate columns exist in DataFrame
        valid_cols = list(df.columns)

        if x_col not in valid_cols:
            x_col = self._first_categorical(df) or valid_cols[0]

        if y_col not in valid_cols:
            y_col = self._first_numeric(df, exclude=x_col) or valid_cols[-1]

        if hue_col and hue_col not in valid_cols:
            hue_col = None
        if hue_col and str(hue_col).lower() in ("null", "none", ""):
            hue_col = None

        # * Add filter summary to title
        if filter_summary and filter_summary != "All Data":
            title = f"{title} — {truncate_string(filter_summary, 40)}"

        model_log.info(
            f"[chart_decision] Config | "
            f"type={chart_type} | x={x_col} | y={y_col} | hue={hue_col}"
        )

        return ChartConfig(
            chart_type = chart_type,
            x_col      = x_col,
            y_col      = y_col,
            hue_col    = hue_col,
            palette    = palette,
            title      = title,
        )

    def _fallback(
        self,
        df:             pd.DataFrame,
        intent:         IntentResult,
        filter_summary: str,
    ) -> ChartConfig:
        """
        # * Rule-based fallback when model fails or DataFrame is empty.
        # * Inspects column dtypes to pick the best chart automatically.
        """
        if df is None or df.empty:
            return ChartConfig(
                chart_type = "heatmap",
                x_col      = "",
                y_col      = "",
                title      = intent.metric,
            )

        cols         = list(df.columns)
        cat_cols     = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        num_cols     = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        metric_lower = intent.metric_key.lower()

        # * Decide chart type from data shape and metric
        if len(cat_cols) >= 2 and num_cols:
            chart_type = "heatmap"
            x_col  = cat_cols[1]
            y_col  = cat_cols[0]
            palette= "RdYlGn_r" if any(k in metric_lower for k in ("bounce","npa","flow")) else "Blues"
        elif len(cat_cols) >= 1 and len(num_cols) >= 2:
            chart_type = "scatter"
            x_col  = num_cols[0]
            y_col  = num_cols[1]
            palette= "coolwarm"
        elif len(cat_cols) >= 1 and num_cols:
            chart_type = "boxplot" if len(df) > 20 else "line"
            x_col  = cat_cols[0]
            y_col  = num_cols[0]
            palette= "Set2"
        elif len(num_cols) == 1:
            chart_type = "kde"
            x_col  = num_cols[0]
            y_col  = num_cols[0]
            palette= "Set2"
        elif num_cols:
            chart_type = "line"
            x_col  = cols[0]
            y_col  = num_cols[0]
            palette= "Set2"
        else:
            chart_type = "heatmap"
            x_col  = cols[0]
            y_col  = cols[1] if len(cols) > 1 else cols[0]
            palette= "Blues"

        title = intent.metric
        if filter_summary and filter_summary != "All Data":
            title = f"{title} — {truncate_string(filter_summary, 40)}"

        return ChartConfig(
            chart_type = chart_type,
            x_col      = x_col,
            y_col      = y_col,
            hue_col    = None,
            palette    = palette,
            title      = title,
        )

    # ── Column helpers ────────────────────────────────────────────────────────

    def _first_categorical(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None

    def _first_numeric(self, df: pd.DataFrame, exclude: str = "") -> Optional[str]:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != exclude:
                return col
        return None


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_decider_instance: Optional[ChartDecider] = None


def get_chart_decider() -> ChartDecider:
    """
    # * Returns the shared ChartDecider instance.

    Usage:
        from models.chart_decider import get_chart_decider
        decider = get_chart_decider()
        config  = decider.decide(df, intent_result, filter_summary)
    """
    global _decider_instance
    if _decider_instance is None:
        _decider_instance = ChartDecider()
    return _decider_instance