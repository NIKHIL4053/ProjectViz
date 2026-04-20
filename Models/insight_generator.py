"""
models/insight_generator.py
---------------------------
# * Generates 3-5 concise business insights from the query result DataFrame.
# * Uses Qwen 7B (fast model) — lightweight task, no SQL needed.
# * Insights are plain English observations a collections manager would care about.

Exports:
    - InsightResult        : Dataclass
    - InsightGenerator     : Main class
    - get_insight_generator: Singleton accessor
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string

log       = get_logger(__name__)
model_log = get_model_logger(__name__)


@dataclass
class InsightResult:
    success:  bool
    insights: list[str]        = field(default_factory=list)
    error:    Optional[str]    = None


_SYSTEM_PROMPT = """
You are a senior loan collections analyst for an NBFC (Non-Banking Financial Company).
Given a data summary from a collections dashboard query, write 3 to 5 concise business insights.

RULES:
- Each insight must be ONE sentence
- Mention specific numbers, branch names, or percentages from the data
- Focus on actionable observations a collections manager would care about
- Highlight the highest, lowest, or most concerning values
- Do NOT use generic statements like "the data shows variation"
- Do NOT explain what bounce rate means — assume the reader knows
- Format: return a JSON array of strings only, no markdown, no extra text

Example output:
["INDORE branch has the highest bounce count at 998, nearly 50% more than the next branch.",
 "BENGALURU TUMKUR ROAD and TUMKUR SIT MAIN ROAD together account for 1,298 bounces.",
 "NELLORE and UJJAIN show similar bounce counts around 390-392, suggesting comparable collection challenges.",
 "The top 3 branches contribute over 35% of all bounced loans in the portfolio.",
 "Branches in Karnataka region dominate the high-bounce list, warranting regional intervention."]
""".strip()


class InsightGenerator:
    """
    # * Generates plain-English insights from a query result DataFrame.
    # * Called after chart is rendered — insights appear below the chart.
    """

    def __init__(self):
        self._client = get_client()
        log.info("[insight_generator] Initialised")

    def generate(
        self,
        df:             pd.DataFrame,
        metric:         str,
        question:       str,
        filter_summary: str = "All Data",
    ) -> InsightResult:
        """
        # * Generate 3-5 business insights from the DataFrame.

        Args:
            df             : Query result DataFrame
            metric         : Metric name (e.g. "Bounce Rate")
            question       : Original user question
            filter_summary : Active filters string

        Returns:
            InsightResult with list of insight strings.
        """
        if df is None or df.empty:
            return InsightResult(success=False, error="Empty DataFrame")

        with benchmark("insight_generation", query=question):

            # * Build data summary — top/bottom rows + stats
            data_summary = self._summarise(df, metric)

            user_msg = f"""
Question asked: "{question}"
Metric: {metric}
Filters: {filter_summary}

Data Summary:
{data_summary}

Write 3-5 insights as a JSON array of strings. Return ONLY the JSON array.
""".strip()

            response: ModelResponse = self._client.call_fast(
                system      = _SYSTEM_PROMPT,
                user        = user_msg,
                step        = "insight_generation",
                expect_json = True,
            )

            if response.failed or not response.parsed:
                log.warning("[insight_generator] Model failed — using rule-based fallback")
                return self._rule_based(df, metric)

            # * parsed should be a list of strings
            parsed = response.parsed
            if isinstance(parsed, list):
                insights = [str(i).strip() for i in parsed if str(i).strip()]
                if insights:
                    model_log.info(
                        f"[insight_generation] OK | count={len(insights)}"
                    )
                    return InsightResult(success=True, insights=insights[:5])

            return self._rule_based(df, metric)

    def _summarise(self, df: pd.DataFrame, metric: str) -> str:
        """
        # * Build a concise text summary of the DataFrame for the model.
        # * Includes top 5, bottom 3, totals, and column stats.
        """
        lines = []
        lines.append(f"Total rows: {len(df)}")
        lines.append(f"Columns: {list(df.columns)}")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        # * Stats for numeric columns
        for col in num_cols[:3]:
            mn  = df[col].min()
            mx  = df[col].max()
            avg = df[col].mean()
            tot = df[col].sum()
            lines.append(
                f"{col}: min={mn:.1f}, max={mx:.1f}, avg={avg:.1f}, total={tot:.1f}"
            )

        # * Top 5 rows
        if num_cols:
            top_col = num_cols[0]
            try:
                top5 = df.nlargest(5, top_col)
                lines.append(f"\nTop 5 by {top_col}:")
                lines.append(top5.to_string(index=False))
            except Exception:
                lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")
        else:
            lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")

        # * Bottom 3
        if num_cols and len(df) > 5:
            bot_col = num_cols[0]
            try:
                bot3 = df.nsmallest(3, bot_col)
                lines.append(f"\nBottom 3 by {bot_col}:")
                lines.append(bot3.to_string(index=False))
            except Exception:
                pass

        return "\n".join(lines)

    def _rule_based(self, df: pd.DataFrame, metric: str) -> InsightResult:
        """
        # * Fallback: generate insights using simple pandas operations.
        # * No model needed — always produces something meaningful.
        """
        insights = []
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        if not num_cols:
            return InsightResult(
                success  = True,
                insights = [f"Query returned {len(df)} rows across {len(df.columns)} columns."]
            )

        primary_num = num_cols[0]
        primary_cat = cat_cols[0] if cat_cols else None

        try:
            top_val  = df[primary_num].max()
            bot_val  = df[primary_num].min()
            avg_val  = df[primary_num].mean()
            total    = df[primary_num].sum()

            if primary_cat:
                top_row  = df.loc[df[primary_num].idxmax()]
                bot_row  = df.loc[df[primary_num].idxmin()]
                top_name = top_row[primary_cat]
                bot_name = bot_row[primary_cat]

                insights.append(
                    f"{top_name} has the highest {primary_num} at {top_val:,.1f}, "
                    f"which is {((top_val/avg_val)-1)*100:.0f}% above the average."
                )
                insights.append(
                    f"{bot_name} has the lowest {primary_num} at {bot_val:,.1f}."
                )

                # * Top 3 share
                top3_sum  = df.nlargest(3, primary_num)[primary_num].sum()
                top3_pct  = (top3_sum / total * 100) if total > 0 else 0
                top3_names= df.nlargest(3, primary_num)[primary_cat].tolist()
                insights.append(
                    f"Top 3 ({', '.join(str(n) for n in top3_names)}) account for "
                    f"{top3_pct:.0f}% of total {primary_num} ({top3_sum:,.0f})."
                )

            else:
                insights.append(
                    f"The highest {primary_num} is {top_val:,.1f} and the lowest is {bot_val:,.1f}."
                )

            insights.append(
                f"The average {primary_num} across all {len(df)} entries is {avg_val:,.1f}."
            )

            # * Spread insight
            if top_val > 0 and bot_val >= 0:
                spread = top_val / max(bot_val, 1)
                if spread > 3:
                    insights.append(
                        f"There is a {spread:.0f}x spread between the highest and lowest "
                        f"{primary_num}, indicating significant variation requiring attention."
                    )

        except Exception as e:
            log.error(f"[insight_generator] Rule-based fallback failed | error={e}")
            insights.append(f"Query returned {len(df)} rows of {metric} data.")

        return InsightResult(success=True, insights=insights[:5])


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[InsightGenerator] = None


def get_insight_generator() -> InsightGenerator:
    global _instance
    if _instance is None:
        _instance = InsightGenerator()
    return _instance
