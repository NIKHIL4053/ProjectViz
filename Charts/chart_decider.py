"""
models/chart_decider.py
-----------------------
# * Pipeline Step 4 — Smart Chart Type Decision.
# * Analyses the ACTUAL DataFrame shape, column types, cardinality, and metric
# * to pick the most appropriate Plotly chart — not a random guess.

# ? Decision hierarchy:
# ?   1. Metric-based rules   — some metrics always map to specific charts
# ?   2. Data shape rules     — column types and counts drive chart type
# ?   3. Cardinality rules    — too many categories → different chart
# ?   4. Qwen 7B confirmation — model validates and may override
# ?   5. Rule-based fallback  — always returns something sensible

# ? Why rule-first instead of model-first?
# ? Qwen 7B was picking scatter for categorical data (images 1-2).
# ? Rules based on actual DataFrame analysis are more reliable than the model
# ? for chart type decisions. Model confirms or adjusts, not decides cold.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from models.analyzer import IntentResult
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string, parse_json_safely

log       = get_logger(__name__)
model_log = get_model_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * CHART CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartConfig:
    """
    # * Full chart specification passed to the Plotly renderer.

    Attributes:
        chart_type : horizontal_bar | line | area | heatmap | kde |
                     scatter | boxplot | treemap
        x_col      : X axis column name
        y_col      : Y axis column name
        hue_col    : Optional colour grouping column
        palette    : Plotly color scale name
        title      : Chart title
        top_n      : Limit to top N rows by y_col (0 = no limit)
        sort_desc  : Sort by y_col descending
    """
    chart_type: str           = "horizontal_bar"
    x_col:      str           = ""
    y_col:      str           = ""
    hue_col:    Optional[str] = None
    palette:    str           = "teal"
    title:      str           = ""
    top_n:      int           = 0
    sort_desc:  bool          = True

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "x_col":      self.x_col,
            "y_col":      self.y_col,
            "hue_col":    self.hue_col,
            "palette":    self.palette,
            "title":      self.title,
            "top_n":      self.top_n,
            "sort_desc":  self.sort_desc,
        }


# ──────────────────────────────────────────────────────────────────────────────
# * METRIC → CHART RULES
# * These take priority over everything else.
# ──────────────────────────────────────────────────────────────────────────────

_METRIC_CHART_RULES = {
    # * Percentage metrics with a group_by → horizontal bar (easy to compare %)
    "Bounce_Percent":       ("horizontal_bar", "RdYlGn_r",  "% metrics"),
    "Resolution_Percent":   ("horizontal_bar", "RdYlGn",    "% metrics"),
    "Coverage_Percent":     ("horizontal_bar", "Blues",     "% metrics"),

    # * Count metrics with group_by → horizontal bar (ranked list)
    "Bounce_Count":         ("horizontal_bar", "Reds",      "count metrics"),
    "Resolution":           ("horizontal_bar", "Greens",    "count metrics"),
    "Coverage":             ("horizontal_bar", "Blues",     "count metrics"),
    "NPA_Count":            ("horizontal_bar", "Reds",      "count metrics"),
    "Add_NPA_Count":        ("horizontal_bar", "Oranges",   "count metrics"),

    # * Matrix metrics → heatmap (two categorical dimensions)
    "Bucket_Movement":      ("heatmap",        "RdYlGn_r",  "matrix metrics"),
    "Bucket_Distribution":  ("heatmap",        "Blues",     "matrix metrics"),

    # * Distribution metrics → kde/histogram
    "DPD_Distribution":     ("kde",            "teal",      "distribution"),

    # * Financial metrics → horizontal bar (easy to read large numbers)
    "Portfolio_Outstanding":("horizontal_bar", "Blues",     "financial"),
    "Total_Overdue":        ("horizontal_bar", "Reds",      "financial"),

    # * FE scorecard → scatter (coverage % vs resolution %)
    "FE_Scorecard":         ("scatter",        "Viridis",   "scorecard"),

    # * Intensity → boxplot (spread across buckets)
    "Intensity":            ("boxplot",        "teal",      "spread"),
}


# ──────────────────────────────────────────────────────────────────────────────
# * DATA SHAPE → CHART RULES
# ──────────────────────────────────────────────────────────────────────────────

def _decide_from_shape(df: pd.DataFrame, intent: IntentResult) -> ChartConfig:
    """
    # * Decide chart type purely from DataFrame column types and cardinality.
    # * This is the most reliable signal — always called as primary logic.
    """
    cols     = list(df.columns)
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    n_rows   = len(df)

    # * Check metric rule first
    metric_key = intent.metric_key
    if metric_key in _METRIC_CHART_RULES:
        chart_type, palette, _ = _METRIC_CHART_RULES[metric_key]
        return _build_config_for_type(chart_type, palette, df, intent, cat_cols, num_cols)

    # * Shape-based rules
    n_cat = len(cat_cols)
    n_num = len(num_cols)

    # * 2 categorical + 1 numeric → heatmap (matrix view)
    if n_cat >= 2 and n_num >= 1:
        return _build_config_for_type("heatmap", "Blues", df, intent, cat_cols, num_cols)

    # * 1 categorical + 1+ numeric:
    if n_cat >= 1 and n_num >= 1:
        cat_col = cat_cols[0]
        num_col = num_cols[0]
        n_unique = df[cat_col].nunique()

        # * Many categories → horizontal bar (readable, sorted)
        if n_unique > 10:
            return ChartConfig(
                chart_type = "horizontal_bar",
                x_col      = num_col,
                y_col      = cat_col,
                palette    = "teal",
                title      = f"{num_col} by {cat_col}",
                top_n      = 15,
                sort_desc  = True,
            )
        # * Few categories + percentage → horizontal bar
        if "%" in num_col or "percent" in num_col.lower() or "rate" in num_col.lower():
            return ChartConfig(
                chart_type = "horizontal_bar",
                x_col      = num_col,
                y_col      = cat_col,
                palette    = "RdYlGn_r" if "bounce" in num_col.lower() else "teal",
                title      = f"{num_col} by {cat_col}",
                top_n      = 20,
                sort_desc  = True,
            )
        # * Few categories → horizontal bar
        return ChartConfig(
            chart_type = "horizontal_bar",
            x_col      = num_col,
            y_col      = cat_col,
            palette    = "teal",
            title      = f"{num_col} by {cat_col}",
            top_n      = 20,
            sort_desc  = True,
        )

    # * 2+ numeric columns → scatter (show relationship)
    if n_num >= 2:
        # * But only if the two columns are not obviously the same metric
        col1, col2 = num_cols[0], num_cols[1]
        if col1.lower() != col2.lower():
            return ChartConfig(
                chart_type = "scatter",
                x_col      = col1,
                y_col      = col2,
                palette    = "Viridis",
                title      = f"{col1} vs {col2}",
            )

    # * 1 numeric only → kde (distribution)
    if n_num == 1:
        return ChartConfig(
            chart_type = "kde",
            x_col      = num_cols[0],
            y_col      = num_cols[0],
            palette    = "teal",
            title      = f"Distribution of {num_cols[0]}",
        )

    # * Final fallback
    return ChartConfig(
        chart_type = "horizontal_bar",
        x_col      = cols[-1] if num_cols else cols[0],
        y_col      = cols[0],
        palette    = "teal",
        title      = intent.metric,
    )


def _build_config_for_type(
    chart_type: str,
    palette:    str,
    df:         pd.DataFrame,
    intent:     IntentResult,
    cat_cols:   list,
    num_cols:   list,
) -> ChartConfig:
    """# * Build a ChartConfig for a given chart type from available columns."""
    cat0 = cat_cols[0] if cat_cols else (df.columns[0])
    cat1 = cat_cols[1] if len(cat_cols) > 1 else cat0
    num0 = num_cols[0] if num_cols else (df.columns[-1])
    num1 = num_cols[1] if len(num_cols) > 1 else num0

    if chart_type == "horizontal_bar":
        return ChartConfig(
            chart_type = "horizontal_bar",
            x_col      = num0,
            y_col      = cat0,
            palette    = palette,
            title      = f"{num0} by {cat0}",
            top_n      = 20,
            sort_desc  = True,
        )
    elif chart_type == "heatmap":
        return ChartConfig(
            chart_type = "heatmap",
            x_col      = cat1,
            y_col      = cat0,
            hue_col    = num0,
            palette    = palette,
            title      = f"{num0} — {cat0} × {cat1}",
        )
    elif chart_type == "scatter":
        return ChartConfig(
            chart_type = "scatter",
            x_col      = num0,
            y_col      = num1,
            hue_col    = cat0 if cat0 else None,
            palette    = palette,
            title      = f"{num0} vs {num1}",
        )
    elif chart_type == "kde":
        return ChartConfig(
            chart_type = "kde",
            x_col      = num0,
            y_col      = num0,
            hue_col    = cat0 if cat0 else None,
            palette    = palette,
            title      = f"Distribution of {num0}",
        )
    elif chart_type == "boxplot":
        return ChartConfig(
            chart_type = "boxplot",
            x_col      = cat0,
            y_col      = num0,
            palette    = palette,
            title      = f"{num0} spread by {cat0}",
        )
    else:
        return ChartConfig(
            chart_type = chart_type,
            x_col      = cat0 or num0,
            y_col      = num0,
            palette    = palette,
            title      = intent.metric,
        )


# ──────────────────────────────────────────────────────────────────────────────
# * CHART DECIDER CLASS
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a data visualization expert for a loan collection dashboard.
Given a proposed chart config and DataFrame info, validate or improve the chart choice.

STRICT RULES — these override everything:
1. NEVER suggest scatter when x and y are not both truly independent numeric columns
2. NEVER suggest line for categorical x-axis data (use horizontal_bar instead)
3. For branch/region/TL comparisons → always horizontal_bar
4. For percentage metrics → always horizontal_bar sorted descending  
5. For two categorical dimensions + one numeric → heatmap
6. For a single numeric distribution → kde
7. For time-series → line or area

Available chart types:
horizontal_bar, line, area, heatmap, kde, scatter, boxplot, treemap

Return ONLY valid JSON — no explanation:
{
  "chart_type": "horizontal_bar",
  "x_col": "COLUMN_NAME",
  "y_col": "COLUMN_NAME",
  "hue_col": "COLUMN_NAME or null",
  "palette": "teal",
  "title": "Descriptive chart title",
  "top_n": 15,
  "sort_desc": true
}
""".strip()


class ChartDecider:
    """
    # * Step 4 of pipeline — decides chart type from DataFrame + intent.
    # * Rule-based logic runs first, Qwen 7B validates/refines.
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
        # * Decide the best chart type for the given data and intent.

        Args:
            df             : DataFrame from database client
            intent         : IntentResult from analyzer
            filter_summary : Active filters string for title

        Returns:
            ChartConfig — always returns something valid.
        """
        if df is None or df.empty:
            return ChartConfig(chart_type="horizontal_bar", title=intent.metric)

        with benchmark("chart_decision", query=intent.raw_question):

            # * Step 1 — Rule-based decision (primary)
            rule_config = _decide_from_shape(df, intent)

            # * Step 2 — Qwen 7B validation (optional refinement)
            qwen_config = self._ask_qwen(df, intent, rule_config)

            # * Step 3 — Use Qwen result if it makes sense, else keep rule
            final = self._pick_best(rule_config, qwen_config, df, intent)

            # * Step 4 — Add filter summary to title
            if filter_summary and filter_summary != "All Data":
                final.title = f"{final.title} — {truncate_string(filter_summary, 35)}"

            model_log.info(
                f"[chart_decision] Final | type={final.chart_type} | "
                f"x={final.x_col} | y={final.y_col} | top_n={final.top_n}"
            )
            return final

    def _ask_qwen(
        self,
        df:          pd.DataFrame,
        intent:      IntentResult,
        rule_config: ChartConfig,
    ) -> Optional[ChartConfig]:
        """
        # * Ask Qwen 7B to validate / refine the rule-based config.
        # * Returns None if model fails or produces nonsense.
        """
        cols      = list(df.columns)
        num_cols  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols  = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        sample    = df.head(3).to_dict(orient="records")

        user_msg = f"""
User question: "{intent.raw_question}"
Metric: {intent.metric} ({intent.metric_key})

DataFrame info:
- Rows: {len(df)}
- Categorical columns: {cat_cols}
- Numeric columns: {num_cols}
- Sample rows: {sample}

Proposed chart (from rules):
- chart_type: {rule_config.chart_type}
- x_col: {rule_config.x_col}
- y_col: {rule_config.y_col}

Validate this choice or suggest a better one. Return JSON config.
""".strip()

        try:
            response: ModelResponse = self._client.call_fast(
                system      = _SYSTEM_PROMPT,
                user        = user_msg,
                step        = "chart_decision",
                expect_json = True,
            )

            if response.failed or not response.parsed:
                return None

            p = response.parsed
            if not isinstance(p, dict):
                return None

            chart_type = str(p.get("chart_type", "")).lower()
            x_col      = str(p.get("x_col", ""))
            y_col      = str(p.get("y_col", ""))

            # * Validate columns exist
            if x_col not in df.columns:
                x_col = rule_config.x_col
            if y_col not in df.columns:
                y_col = rule_config.y_col

            hue = p.get("hue_col")
            if hue and str(hue).lower() in ("null", "none", "") :
                hue = None
            if hue and hue not in df.columns:
                hue = None

            return ChartConfig(
                chart_type = chart_type or rule_config.chart_type,
                x_col      = x_col,
                y_col      = y_col,
                hue_col    = hue,
                palette    = str(p.get("palette", rule_config.palette)),
                title      = str(p.get("title",   rule_config.title)),
                top_n      = int(p.get("top_n",   rule_config.top_n)),
                sort_desc  = bool(p.get("sort_desc", rule_config.sort_desc)),
            )

        except Exception as e:
            log.warning(f"[chart_decider] Qwen validation failed | error={e}")
            return None

    def _pick_best(
        self,
        rule:   ChartConfig,
        qwen:   Optional[ChartConfig],
        df:     pd.DataFrame,
        intent: IntentResult,
    ) -> ChartConfig:
        """
        # * Choose between rule-based and Qwen config.
        # * Keeps Qwen's suggestion only if it doesn't violate hard rules.
        """
        if qwen is None:
            return rule

        cols     = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

        # * Hard block: scatter only if both x and y are numeric
        if qwen.chart_type == "scatter":
            if qwen.x_col not in num_cols or qwen.y_col not in num_cols:
                log.warning("[chart_decider] Blocking invalid scatter — reverting to rule")
                return rule

        # * Hard block: line only if x is date/time-like or ordered
        if qwen.chart_type == "line":
            if qwen.x_col in cat_cols and df[qwen.x_col].nunique() > 15:
                log.warning("[chart_decider] Blocking line for high-cardinality categorical x")
                return rule

        # * Qwen's title is often better — keep it if chart type matches
        if qwen.chart_type == rule.chart_type:
            rule.title = qwen.title or rule.title
            rule.palette = qwen.palette or rule.palette
            return rule

        return qwen


# ── Singleton ─────────────────────────────────────────────────────────────────

_decider_instance: Optional[ChartDecider] = None


def get_chart_decider() -> ChartDecider:
    global _decider_instance
    if _decider_instance is None:
        _decider_instance = ChartDecider()
    return _decider_instance
