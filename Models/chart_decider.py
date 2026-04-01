"""
models/chart_decider.py
-----------------------
# * Pipeline Step 4 — Chart Type Decision.
# * Takes the SQL result DataFrame + IntentResult and decides:
# *   - Chart type (line, area, heatmap, kde, scatter, boxplot)
# *   - x_col, y_col, hue_col, palette, title
# * Returns a ChartConfig dataclass.

# ? Which model: Qwen 7B (FAST_MODEL)
# ? Why: Faster (~3s vs ~15s for 14B). Chart type is classification,
# ?      not complex SQL reasoning — 7B handles it well.
# ? If Qwen 7B fails, rule-based fallback covers all cases instantly.

Exports:
    - ChartConfig        : Full chart rendering configuration
    - ChartDecider       : Main class
    - get_chart_decider(): Singleton accessor
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import FAST_MODEL
from core.dictionary import get_dictionary
from models.analyzer import IntentResult
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string, estimate_tokens

log       = get_logger(__name__)
model_log = get_model_logger(__name__)

VALID_CHART_TYPES = {"line", "area", "heatmap", "kde", "scatter", "boxplot"}
VALID_PALETTES    = {
    "RdYlGn_r", "Blues", "Set2", "flare",
    "viridis", "coolwarm", "Reds", "tab10",
}


# ──────────────────────────────────────────────────────────────────────────────
# CHART CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartConfig:
    """
    Full configuration for rendering one chart.
    app.py calls to_dict() and translates keys for renderer.py.
    """
    chart_type:     str
    x_col:          str            = ""
    y_col:          str            = ""
    hue_col:        Optional[str]  = None
    size_col:       Optional[str]  = None
    value_col:      Optional[str]  = None
    palette:        str            = "Set2"
    title:          str            = ""
    x_label:        str            = ""
    y_label:        str            = ""
    rotate_x:       int            = 0
    annotate:       bool           = True
    metric:         str            = ""
    filter_summary: str            = ""
    success:        bool           = True
    error:          Optional[str]  = None

    @property
    def failed(self) -> bool:
        return not self.success

    def to_dict(self) -> dict:
        return {
            "chart_type":     self.chart_type,
            "x_col":          self.x_col,
            "y_col":          self.y_col,
            "hue_col":        self.hue_col,
            "size_col":       self.size_col,
            "value_col":      self.value_col,
            "palette":        self.palette,
            "title":          self.title,
            "x_label":        self.x_label,
            "y_label":        self.y_label,
            "rotate_x":       self.rotate_x,
            "annotate":       self.annotate,
            "metric":         self.metric,
            "filter_summary": self.filter_summary,
        }


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a data visualization expert for a loan collection portfolio dashboard.
Given DataFrame column names and a sample, return the best chart config as JSON.

CHART TYPE RULES (first match wins — NEVER use bar chart):
1. Date/time column + one numeric          → "line"
2. Date/time column + multiple numerics    → "area"
3. Two categorical + one numeric           → "heatmap"
4. One numeric + distribution question     → "kde"
5. Two numeric columns + correlation       → "scatter"
6. One categorical + one numeric + spread  → "boxplot"
7. One categorical + one numeric (default) → "heatmap"

PALETTE GUIDE:
- "RdYlGn_r" : bounce, NPA, flow (red=bad, green=good)
- "Blues"    : neutral counts
- "Set2"     : categorical groups (default)
- "flare"    : risk/alert metrics
- "viridis"  : density/distribution
- "coolwarm" : correlation/scatter
- "Reds"     : overdue/loss amounts
- "tab10"    : many categories (8+)

COLUMNS:  {columns}
DTYPES:   {dtypes}
ROW COUNT:{row_count}
SAMPLE DATA:
{sample}

METRIC:   {metric}
QUESTION: {question}

Return ONLY valid JSON — no markdown, no explanation, no backticks:
{{
  "chart_type": "heatmap",
  "x_col":      "exact column name from the DataFrame",
  "y_col":      "exact column name from the DataFrame",
  "hue_col":    null,
  "palette":    "Set2",
  "title":      "descriptive chart title"
}}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# CHART DECIDER
# ──────────────────────────────────────────────────────────────────────────────

class ChartDecider:
    """
    Step 4 — decides chart type and axes from the result DataFrame.
    Priority order:
      1. Quick unambiguous rules (fastest — no model call)
      2. Qwen 7B model decision
      3. Full rule-based fallback (if model fails)
    """

    def __init__(self):
        self._client     = get_client()
        self._dictionary = get_dictionary()
        log.info("[chart_decider] Initialised")

    def decide(
        self,
        df:             pd.DataFrame,
        intent:         IntentResult,
        filter_summary: str = "All Data",
    ) -> ChartConfig:
        """
        Main entry point. Always returns a valid ChartConfig.

        Args:
            df             : Result DataFrame from PostgreSQL / mock
            intent         : IntentResult from analyzer.py
            filter_summary : Filter string for chart title

        Returns:
            ChartConfig — never raises, always returns something renderable.
        """
        if df is None or df.empty:
            return ChartConfig(chart_type="heatmap", success=False,
                               error="Empty DataFrame")

        log.info(f"[chart_decider] Deciding | metric='{intent.metric}' | shape={df.shape}")

        with benchmark("chart_decision", query=intent.raw_question):

            # 1. Try quick unambiguous rule first
            quick = self._quick_rule(df, intent, filter_summary)
            if quick:
                log.info(f"[chart_decider] Quick rule → {quick.chart_type}")
                return quick

            # 2. Try Qwen 7B
            if self._client.is_running():
                try:
                    return self._model_decide(df, intent, filter_summary)
                except Exception as e:
                    log.warning(f"[chart_decider] Model failed → rules | {e}")

            # 3. Full rule-based fallback
            return self._rule_based(df, intent, filter_summary)

    # ── Quick rules ───────────────────────────────────────────────────────────

    def _quick_rule(self, df, intent, filter_summary) -> Optional[ChartConfig]:
        """Returns a ChartConfig for completely unambiguous cases, else None."""
        cols   = df.columns.tolist()
        metric = intent.metric_key
        title  = (f"{intent.metric} — {filter_summary}"
                  if filter_summary != "All Data" else intent.metric)

        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

        # Bucket movement matrix
        if metric == "Bucket_Movement" and len(cat_cols) >= 2 and num_cols:
            return ChartConfig(
                chart_type="heatmap", x_col=cat_cols[0], y_col=cat_cols[1],
                palette="Blues", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # DPD distribution
        if metric == "DPD_Distribution" and num_cols:
            x = next((c for c in num_cols if "dpd" in c.lower()), num_cols[0])
            h = next((c for c in cat_cols if "bucket" in c.lower()), None)
            return ChartConfig(
                chart_type="kde", x_col=x, y_col="", hue_col=h,
                palette="Set2", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # FE Scorecard — scatter
        if metric == "FE_Scorecard" and len(num_cols) >= 2:
            return ChartConfig(
                chart_type="scatter",
                x_col=num_cols[0], y_col=num_cols[1],
                hue_col=cat_cols[0] if cat_cols else None,
                palette="viridis", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        return None  # not unambiguous — let model decide

    # ── Model-based decision ──────────────────────────────────────────────────

    def _model_decide(self, df, intent, filter_summary) -> ChartConfig:
        """Call Qwen 7B and parse JSON into ChartConfig."""
        cols = list(df.columns)
        try:
            sample = df.head(3).to_string(index=False, max_colwidth=25)
        except Exception:
            sample = str(df.head(3).to_dict())

        system = _SYSTEM_PROMPT.format(
            columns   = ", ".join(f'"{c}"' for c in cols),
            dtypes    = {c: str(df[c].dtype) for c in cols},
            row_count = len(df),
            sample    = sample,
            metric    = intent.metric,
            question  = intent.raw_question,
        )

        response: ModelResponse = self._client.call_fast(
            system      = system,
            user        = "Return ONLY the JSON config.",
            step        = "chart_decision",
            expect_json = True,
        )

        if response.failed or response.parsed is None:
            return self._rule_based(df, intent, filter_summary)

        return self._parse_config(response.parsed, df, intent, filter_summary)

    # ── Parse model JSON ──────────────────────────────────────────────────────

    def _parse_config(self, parsed, df, intent, filter_summary) -> ChartConfig:
        """Validate model JSON fields and build ChartConfig."""
        if not isinstance(parsed, dict):
            return self._rule_based(df, intent, filter_summary)

        df_cols = set(df.columns.tolist())

        def resolve(val) -> Optional[str]:
            if not val or str(val).lower() in ("null", "none", ""):
                return None
            s = str(val).strip().strip('"')
            if s in df_cols:
                return s
            for c in df_cols:
                if c.lower() == s.lower():
                    return c
            log.warning(f"[chart_decider] Column '{s}' not in DataFrame — skipping")
            return None

        chart_type = str(parsed.get("chart_type", "heatmap")).lower().strip()
        if chart_type not in VALID_CHART_TYPES:
            chart_type = "heatmap"

        palette = str(parsed.get("palette", "Set2")).strip()
        if palette not in VALID_PALETTES:
            palette = "Set2"

        cols_list = list(df.columns)
        x_col = resolve(parsed.get("x_col")) or (cols_list[0] if cols_list else "")
        y_col = resolve(parsed.get("y_col")) or (cols_list[1] if len(cols_list) > 1 else x_col)
        hue   = resolve(parsed.get("hue_col"))

        title = str(parsed.get("title", intent.metric)).strip()
        if filter_summary and filter_summary != "All Data" and filter_summary not in title:
            title = f"{title} — {filter_summary}"

        model_log.info(
            f"[chart_decision] Model OK | type={chart_type} | "
            f"x={x_col} y={y_col} hue={hue} palette={palette}"
        )

        return ChartConfig(
            chart_type=chart_type, x_col=x_col, y_col=y_col, hue_col=hue,
            palette=palette, title=title,
            metric=intent.metric, filter_summary=filter_summary, success=True,
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based(self, df, intent, filter_summary) -> ChartConfig:
        """Full rule-based fallback — always produces a renderable chart."""
        log.info(f"[chart_decider] Rule-based fallback | metric={intent.metric_key}")

        cols   = df.columns.tolist()
        metric = intent.metric_key
        title  = (f"{intent.metric} — {filter_summary}"
                  if filter_summary != "All Data" else intent.metric)

        date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])
                     or "date" in c.lower()]
        num_cols  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])
                     and c not in date_cols]
        cat_cols  = [c for c in cols if c not in num_cols and c not in date_cols]

        # Time series
        if date_cols and num_cols:
            return ChartConfig(
                chart_type="line", x_col=date_cols[0], y_col=num_cols[0],
                hue_col=cat_cols[0] if cat_cols else None,
                palette="Set2", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # Scatter for performance metrics
        if metric in ("FE_Scorecard", "Coverage_Percent") and len(num_cols) >= 2:
            return ChartConfig(
                chart_type="scatter", x_col=num_cols[0], y_col=num_cols[1],
                hue_col=cat_cols[0] if cat_cols else None,
                palette="viridis", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # Two categoricals + numeric → heatmap
        if len(cat_cols) >= 2 and num_cols:
            palette = ("RdYlGn_r" if metric in
                       ("Bounce_Percent", "Bounce_Count", "NPA_Count")
                       else "Blues")
            return ChartConfig(
                chart_type="heatmap", x_col=cat_cols[0], y_col=cat_cols[1],
                palette=palette, title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # One categorical + numeric → heatmap
        if cat_cols and num_cols:
            palette = ("RdYlGn_r" if metric in
                       ("Bounce_Percent", "NPA_Count", "Resolution_Percent")
                       else "Blues")
            return ChartConfig(
                chart_type="heatmap", x_col=cat_cols[0], y_col=num_cols[0],
                palette=palette, title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # Pure numeric → KDE
        if num_cols:
            return ChartConfig(
                chart_type="kde", x_col=num_cols[0], y_col="",
                palette="Set2", title=title,
                metric=intent.metric, filter_summary=filter_summary
            )

        # Absolute last resort
        x = cols[0] if cols else ""
        y = cols[1] if len(cols) > 1 else x
        return ChartConfig(
            chart_type="heatmap", x_col=x, y_col=y,
            palette="Blues", title=title,
            metric=intent.metric, filter_summary=filter_summary
        )


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

_instance: Optional[ChartDecider] = None


def get_chart_decider() -> ChartDecider:
    """
    Returns the shared ChartDecider instance.

    Usage:
        from models.chart_decider import get_chart_decider
        config = get_chart_decider().decide(df, intent, "Branch: PUNE")
        # config.to_dict() → app.py translates → renderer.py renders
    """
    global _instance
    if _instance is None:
        _instance = ChartDecider()
    return _instance