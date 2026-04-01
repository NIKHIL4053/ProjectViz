"""
models/sql_generator.py
-----------------------
# * Pipeline Step 3 — SQL Generation.
# * Takes IntentResult + slicer answers from clarifier.py and generates
# * a valid PostgreSQL SELECT query against the loan_dashboard table.

# ? Which model: Qwen Coder 14B
# ? Why: SQL generation requires strong code generation + domain awareness.
# ?      Qwen Coder outperforms base Qwen on structured code output.
# ?      14B gives enough capacity for complex aggregations with multiple filters.

# ? What Qwen receives:
# ?   - The metric to calculate (from IntentResult.metric_key)
# ?   - The columns needed (from IntentResult.columns_needed)
# ?   - The GROUP BY column (from IntentResult.group_by)
# ?   - The slicer answers (from clarifier.collect_answers())
# ?   - Relevant SQL patterns from ChromaDB (from dictionary.get_coder_context())
# ?   - Hard SQL rules (table name, quoting, NULLIF, COUNT DISTINCT)

# ? What Qwen returns:
# ?   - A clean PostgreSQL SELECT query string
# ?   - No EVALUATE, no DAX, no markdown — just SQL

Exports:
    - SQLResult          : Dataclass for generated SQL output
    - SQLGenerator       : Main class
    - get_sql_generator(): Singleton accessor
"""

import re
from dataclasses import dataclass
from typing import Optional

from config import PG_TABLE_NAME, PG_SCHEMA
from core.dictionary import get_dictionary
from models.analyzer import IntentResult
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string, estimate_tokens

# * Module level loggers
log       = get_logger(__name__)
model_log = get_model_logger(__name__)

# * Forbidden SQL keywords — never allowed in generated SQL
_FORBIDDEN = {
    "INSERT", "UPDATE", "DELETE", "DROP",
    "TRUNCATE", "ALTER", "CREATE", "GRANT",
    "REVOKE", "EXECUTE",
}


# ──────────────────────────────────────────────────────────────────────────────
# * SQL RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SQLResult:
    """
    # * Structured output from the SQL generator step.

    Attributes:
        success      : True if SQL was generated and validated
        sql          : The generated PostgreSQL SELECT query
        metric_key   : Which metric this SQL calculates
        filters_used : Dict of slicer filters applied in WHERE clause
        error        : Error message if success=False
    """
    success:      bool
    sql:          str                   = ""
    metric_key:   str                   = ""
    filters_used: dict                  = None
    error:        Optional[str]         = None

    def __post_init__(self):
        if self.filters_used is None:
            self.filters_used = {}

    @property
    def failed(self) -> bool:
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# * SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """
You are an expert PostgreSQL query writer for a loan collection portfolio system.
Your job is to write a single valid PostgreSQL SELECT query based on the metric and filters provided.

DATABASE RULES — follow every rule exactly:
1. Table name: {table_name}  Schema: {schema}
2. Always double-quote column names that contain spaces or special chars:
   "Op bucket", "Cust ID", "Bounce status", "Cust wise status", "Next Date",
   "Bounce charges", "Bounce charge collected", "Visit Count", "Loan level bounce",
   "Cust level bounce", "Day wise asset classification", "Digital/cash",
   "MOB Bucket", "NPA_Origination_Date", "NPA MOB", "Portfolio new",
   "Payment mode", "Payment Day bucket", "Allocated or not", "Visit or not",
   "Coll/Sales", "Add NPA", "Type of Arrangement", "Transaction Type",
   "Closing bucket", "Cust wise status", "Due Date", "Installment No"
3. Unquoted columns (no spaces): loanappno, dpd, dpd_casewise, bal_prin, tod
4. ALWAYS use COUNT(DISTINCT "Cust ID") for customer-level metrics
5. Use COUNT(loanappno) for loan-level metrics
6. Use ROUND(..., 2) for all percentages and ratios
7. Use NULLIF(denominator, 0) to prevent division by zero
8. Percentage formula: ROUND(numerator * 100.0 / NULLIF(denominator, 0), 2)
9. Date filter format: "Next Date" = 'YYYY-MM-DD' or "Next Date" >= 'YYYY-MM-DD'
10. For multiple filter values use IN ('val1', 'val2')
11. Always end grouped queries with ORDER BY
12. Return ONLY the SQL query — no explanation, no markdown, no comments

METRIC RULES:
- Bounce: WHERE "Bounce status" IN ('Tech', 'Non Tech') — NEVER include 'PAID'
- Resolution: WHERE "Cust wise status" = 'Norm'
- Coverage: WHERE "Visit or not" = 'Visited' AND "Allocated or not" = 'Allocated'
  Coverage denominator: WHERE "Allocated or not" = 'Allocated'
- Intensity: ROUND(SUM("Visit Count")::NUMERIC / NULLIF(COUNT(DISTINCT "Cust ID"), 0), 2)
- NPA: WHERE "Op bucket" = 'NPA'
- New NPA: WHERE "Add NPA" = 1
- Portfolio outstanding: SUM(bal_prin)
- Total overdue: SUM(tod)

RELEVANT SQL PATTERNS FROM DATA DICTIONARY:
{context}

APPLIED FILTERS:
{filters_block}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# * SQL GENERATOR CLASS
# ──────────────────────────────────────────────────────────────────────────────

class SQLGenerator:
    """
    # * Step 3 of the pipeline — generates PostgreSQL SQL from IntentResult + slicers.
    # * Calls Qwen Coder 14B with SQL patterns + domain rules injected into prompt.
    # * Validates the generated SQL before returning.
    """

    def __init__(self):
        self._client     = get_client()
        self._dictionary = get_dictionary()
        log.info("[sql_generator] Initialised")

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        intent:  IntentResult,
        slicers: dict,
    ) -> SQLResult:
        """
        # * Generate a PostgreSQL SELECT query for the given intent and slicers.

        Args:
            intent  : IntentResult from analyzer.py
            slicers : Dict of {field: value} from clarifier.collect_answers()
                      "All" values are ignored — not included in WHERE clause

        Returns:
            SQLResult with success=True and the SQL string,
            or SQLResult with success=False and error message.

        Example:
            result = generator.generate(
                intent  = intent_result,
                slicers = {"Region": "Pune", "Branch": "All", "Next Date": "Feb 2026"}
            )
            if result.success:
                df = db_client.run_query(result.sql)
        """
        if intent.failed:
            return SQLResult(
                success=False,
                error="Cannot generate SQL — intent analysis failed"
            )

        log.info(
            f"[sql_generator] Generating | "
            f"metric='{intent.metric_key}' | "
            f"group_by='{intent.group_by}' | "
            f"slicers={list(slicers.keys())}"
        )

        with benchmark("sql_generation", query=intent.raw_question):

            # * Step 1 — Format active slicer filters
            active_slicers = self._build_active_slicers(slicers)
            filters_block  = self._format_filters_block(active_slicers)

            # * Step 2 — Fetch SQL context from ChromaDB
            context = self._fetch_context(intent)

            # * Step 3 — Build prompt
            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                table_name    = PG_TABLE_NAME,
                schema        = PG_SCHEMA,
                context       = context,
                filters_block = filters_block,
            )

            user_message = self._build_user_message(intent, active_slicers)

            model_log.debug(
                f"[sql_generation] Prompt built | "
                f"~{estimate_tokens(system_prompt)} tokens"
            )

            # * Step 4 — Call Qwen Coder
            response: ModelResponse = self._client.call_coder(
                system      = system_prompt,
                user        = user_message,
                step        = "sql_generation",
                expect_json = False,   # * SQL is plain text, not JSON
            )

            if response.failed:
                log.warning(
                    f"[sql_generator] Model call failed — using rule-based fallback | "
                    f"error={response.error}"
                )
                return self._rule_based_sql(intent, active_slicers)

            # * Step 5 — Clean and validate
            sql = self._clean_sql(response.content)
            validation_error = self._validate_sql(sql)

            if validation_error:
                model_log.warning(
                    f"[sql_generation] Validation failed — using fallback | "
                    f"error={validation_error} | "
                    f"sql_preview=\"{truncate_string(sql, 100)}\""
                )
                return self._rule_based_sql(intent, active_slicers)

            model_log.info(
                f"[sql_generation] OK | "
                f"metric='{intent.metric_key}' | "
                f"sql_length={len(sql)} | "
                f"preview=\"{truncate_string(sql, 100)}\""
            )

            return SQLResult(
                success      = True,
                sql          = sql,
                metric_key   = intent.metric_key,
                filters_used = active_slicers,
            )

    # ── Filter building ───────────────────────────────────────────────────────

    def _build_active_slicers(self, slicers: dict) -> dict:
        """
        # * Filter out 'All' / empty / None values from slicer dict.
        # * Converts "Feb 2026" date strings to ISO format.
        # * Normalises Add NPA labels ("1 (New NPA)" → 1).

        Args:
            slicers : Raw dict from clarifier.collect_answers()

        Returns:
            Dict with only active, clean filter values.
        """
        active = {}
        for field, value in slicers.items():
            if not value:
                continue
            # * Skip All / empty
            if isinstance(value, str) and value.lower() in ("all", ""):
                continue
            if isinstance(value, list) and len(value) == 0:
                continue

            # * Convert month string to ISO date
            if field == "Next Date" and isinstance(value, str):
                iso = self._month_to_iso(value)
                if iso:
                    active[field] = iso
                continue

            # * Normalise Add NPA label
            if field == "Add NPA":
                if isinstance(value, str):
                    value = 1 if "1" in value else 0
                active[field] = value
                continue

            active[field] = value

        return active

    def _month_to_iso(self, month_str: str) -> Optional[str]:
        """
        # * Convert "Feb 2026" or "February 2026" → "2026-02-01".

        Args:
            month_str : Month string from clarifying question answer

        Returns:
            ISO date string or None if conversion fails.
        """
        import datetime
        month_map = {
            "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
            "may": 5,  "jun": 6,  "jul": 7,  "aug": 8,
            "sep": 9,  "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            parts = month_str.strip().split()
            if len(parts) == 2:
                month_num = month_map.get(parts[0].lower()[:3])
                year      = int(parts[1])
                if month_num and year:
                    return f"{year}-{month_num:02d}-01"
        except Exception:
            pass
        return None

    def _format_filters_block(self, active_slicers: dict) -> str:
        """
        # * Build a human-readable filters block for the system prompt.
        # * This tells Qwen exactly which WHERE conditions to include.

        Args:
            active_slicers : Clean dict of active filter values

        Returns:
            Formatted string like:
            - Region = 'Pune'
            - Bounce status IN ('Tech', 'Non Tech')
            - Next Date = '2026-02-01'
        """
        if not active_slicers:
            return "No filters — query covers all data"

        lines = []
        for field, value in active_slicers.items():
            if isinstance(value, list):
                vals = ", ".join(f"'{v}'" for v in value)
                lines.append(f'- "{field}" IN ({vals})')
            elif isinstance(value, int):
                lines.append(f'- "{field}" = {value}')
            else:
                lines.append(f'- "{field}" = \'{value}\'')

        return "\n".join(lines)

    def _build_user_message(self, intent: IntentResult, active_slicers: dict) -> str:
        """
        # * Build the user message for the Qwen Coder prompt.

        Args:
            intent         : IntentResult from analyzer
            active_slicers : Clean active filter dict

        Returns:
            User message string.
        """
        group_by_str = (
            f'GROUP BY "{intent.group_by}"'
            if intent.group_by else
            "No GROUP BY (single aggregate value)"
        )

        return f"""
Generate a PostgreSQL SELECT query for this request:

User question: "{intent.raw_question}"
Metric to calculate: {intent.metric}
Metric key: {intent.metric_key}
Columns needed: {intent.columns_needed}
{group_by_str}
Aggregation: {intent.aggregation}
Granularity: {intent.granularity} level

Active WHERE filters to include:
{self._format_filters_block(active_slicers)}

SQL pattern hint: {intent.sql_pattern_hint or 'none'}

Write the SQL query now. Return ONLY the SQL — no explanation, no markdown.
""".strip()

    # ── Context fetch ─────────────────────────────────────────────────────────

    def _fetch_context(self, intent: IntentResult) -> str:
        """
        # * Fetch relevant SQL patterns and business logic from ChromaDB.

        Args:
            intent : IntentResult for query context

        Returns:
            Context string for prompt injection.
        """
        if not self._dictionary.is_ready:
            return ""

        query = f"{intent.raw_question} {intent.metric} {intent.metric_key}"

        sql_chunks   = self._dictionary.search(query, top_k=5, source="sql_patterns")
        logic_chunks = self._dictionary.search(query, top_k=3, source="business_logic")

        sections = []
        if sql_chunks:
            sections.append("SQL PATTERNS:\n" + "\n\n".join(sql_chunks))
        if logic_chunks:
            sections.append("BUSINESS LOGIC:\n" + "\n\n".join(logic_chunks))

        return "\n\n".join(sections)

    # ── SQL cleaning ──────────────────────────────────────────────────────────

    def _clean_sql(self, raw: str) -> str:
        """
        # * Strip markdown fences and extra whitespace from model output.
        # * Models sometimes wrap SQL in ```sql ... ``` blocks.

        Args:
            raw : Raw model response string

        Returns:
            Clean SQL string.
        """
        clean = raw.strip()

        # * Strip markdown code fences
        if "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                clean = parts[1]
                if clean.lower().startswith(("sql", "postgresql", "pgsql")):
                    clean = clean[clean.index("\n") + 1:]

        return clean.strip().rstrip(";").strip()

    # ── SQL validation ────────────────────────────────────────────────────────

    def _validate_sql(self, sql: str) -> Optional[str]:
        """
        # * Validate generated SQL before returning to pipeline.
        # * Returns error string if invalid, None if valid.

        Rules:
            1. Must start with SELECT or WITH
            2. No forbidden modification keywords
            3. Not empty
            4. Not excessively long

        Args:
            sql : Cleaned SQL string

        Returns:
            Error string or None.
        """
        if not sql or not sql.strip():
            return "Generated SQL is empty"

        upper = sql.strip().upper()

        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return f"SQL must start with SELECT or WITH — got: {sql[:50]}"

        words = set(re.findall(r'\b[A-Z]+\b', upper))
        forbidden = words & _FORBIDDEN
        if forbidden:
            return f"SQL contains forbidden keywords: {forbidden}"

        if len(sql) > 5000:
            return f"SQL too long ({len(sql)} chars)"

        return None

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_sql(
        self,
        intent:         IntentResult,
        active_slicers: dict,
    ) -> SQLResult:
        """
        # * Generate SQL using hardcoded rules when Qwen fails.
        # * Covers all 8 primary metric types with correct PostgreSQL syntax.
        # * Always produces valid SQL — never returns empty.

        Args:
            intent         : IntentResult from analyzer
            active_slicers : Active filter dict

        Returns:
            SQLResult with fallback SQL.
        """
        log.info(f"[sql_generator] Rule-based fallback | metric={intent.metric_key}")

        table  = PG_TABLE_NAME
        where  = self._build_where_clause(active_slicers)
        group  = f'GROUP BY "{intent.group_by}"' if intent.group_by else ""
        order  = f'ORDER BY "{intent.group_by}"' if intent.group_by else ""
        metric = intent.metric_key

        # ── Metric templates ──────────────────────────────────────────────────

        if metric in ("Bounce_Count", "Bounce_Percent"):
            if intent.group_by:
                sql = f"""
SELECT
  "{intent.group_by}",
  COUNT(DISTINCT CASE WHEN "Bounce status" IN ('Tech', 'Non Tech') THEN "Cust ID" END) AS "Bounce Count",
  COUNT(DISTINCT "Cust ID") AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN "Bounce status" IN ('Tech', 'Non Tech') THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT "Cust ID"), 0),
  2) AS "Bounce %"
FROM {table}
{where}
{group}
{order}
""".strip()
            else:
                sql = f"""
SELECT
  COUNT(DISTINCT CASE WHEN "Bounce status" IN ('Tech', 'Non Tech') THEN "Cust ID" END) AS "Bounce Count",
  COUNT(DISTINCT "Cust ID") AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN "Bounce status" IN ('Tech', 'Non Tech') THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT "Cust ID"), 0),
  2) AS "Bounce %"
FROM {table}
{where}
""".strip()

        elif metric in ("Resolution", "Resolution_Percent"):
            if intent.group_by:
                sql = f"""
SELECT
  "{intent.group_by}",
  COUNT(DISTINCT CASE WHEN "Cust wise status" = 'Norm' THEN "Cust ID" END) AS "Resolved Count",
  COUNT(DISTINCT "Cust ID") AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN "Cust wise status" = 'Norm' THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT "Cust ID"), 0),
  2) AS "Resolution %"
FROM {table}
{where}
{group}
{order}
""".strip()
            else:
                sql = f"""
SELECT
  COUNT(DISTINCT CASE WHEN "Cust wise status" = 'Norm' THEN "Cust ID" END) AS "Resolved Count",
  ROUND(
    COUNT(DISTINCT CASE WHEN "Cust wise status" = 'Norm' THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT "Cust ID"), 0),
  2) AS "Resolution %"
FROM {table}
{where}
""".strip()

        elif metric in ("Coverage", "Coverage_Percent"):
            if intent.group_by:
                sql = f"""
SELECT
  "{intent.group_by}",
  COUNT(DISTINCT CASE WHEN "Visit or not" = 'Visited' AND "Allocated or not" = 'Allocated' THEN "Cust ID" END) AS "Visited Count",
  COUNT(DISTINCT CASE WHEN "Allocated or not" = 'Allocated' THEN "Cust ID" END) AS "Allocated Count",
  ROUND(
    COUNT(DISTINCT CASE WHEN "Visit or not" = 'Visited' AND "Allocated or not" = 'Allocated' THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN "Allocated or not" = 'Allocated' THEN "Cust ID" END), 0),
  2) AS "Coverage %"
FROM {table}
{where}
{group}
{order}
""".strip()
            else:
                sql = f"""
SELECT
  ROUND(
    COUNT(DISTINCT CASE WHEN "Visit or not" = 'Visited' AND "Allocated or not" = 'Allocated' THEN "Cust ID" END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN "Allocated or not" = 'Allocated' THEN "Cust ID" END), 0),
  2) AS "Coverage %"
FROM {table}
{where}
""".strip()

        elif metric == "Intensity":
            if intent.group_by:
                sql = f"""
SELECT
  "{intent.group_by}",
  ROUND(SUM("Visit Count")::NUMERIC / NULLIF(COUNT(DISTINCT "Cust ID"), 0), 2) AS "Avg Visits per Customer",
  SUM("Visit Count") AS "Total Visits",
  COUNT(DISTINCT "Cust ID") AS "Customer Count"
FROM {table}
{where}
{group}
{order}
""".strip()
            else:
                sql = f"""
SELECT
  ROUND(SUM("Visit Count")::NUMERIC / NULLIF(COUNT(DISTINCT "Cust ID"), 0), 2) AS "Intensity"
FROM {table}
{where}
""".strip()

        elif metric in ("Bucket_Distribution", "Bucket_Movement"):
            sql = f"""
SELECT
  "Op bucket",
  COUNT(loanappno) AS "Loan Count",
  COUNT(DISTINCT "Cust ID") AS "Customer Count",
  SUM(bal_prin) AS "Balance Principal",
  SUM(tod) AS "Total Overdue"
FROM {table}
{where}
GROUP BY "Op bucket"
ORDER BY "Loan Count" DESC
""".strip()

        elif metric in ("NPA_Count", "Add_NPA_Count"):
            npa_filter = 'WHERE "Add NPA" = 1' if metric == "Add_NPA_Count" else 'WHERE "Op bucket" = \'NPA\''
            group_col  = intent.group_by or "Region"
            sql = f"""
SELECT
  "{group_col}",
  COUNT(loanappno) AS "NPA Count",
  COUNT(DISTINCT "Cust ID") AS "Customer Count",
  SUM(bal_prin) AS "Outstanding"
FROM {table}
{npa_filter}
GROUP BY "{group_col}"
ORDER BY "NPA Count" DESC
""".strip()

        elif metric in ("Portfolio_Outstanding", "Total_Overdue"):
            agg_col = "bal_prin" if metric == "Portfolio_Outstanding" else "tod"
            agg_label = "Balance Principal" if metric == "Portfolio_Outstanding" else "Total Overdue"
            group_col = intent.group_by or "Op bucket"
            sql = f"""
SELECT
  "{group_col}",
  SUM({agg_col}) AS "{agg_label}",
  COUNT(loanappno) AS "Loan Count"
FROM {table}
{where}
GROUP BY "{group_col}"
ORDER BY "{agg_label}" DESC
""".strip()

        else:
            # * Generic fallback — count by group_by or branch
            group_col = intent.group_by or "Branch"
            sql = f"""
SELECT
  "{group_col}",
  COUNT(DISTINCT "Cust ID") AS "Customer Count",
  COUNT(loanappno) AS "Loan Count",
  SUM(bal_prin) AS "Balance Principal"
FROM {table}
{where}
GROUP BY "{group_col}"
ORDER BY "Customer Count" DESC
""".strip()

        model_log.info(
            f"[sql_generation] FALLBACK SQL | "
            f"metric={metric} | "
            f"preview=\"{truncate_string(sql, 100)}\""
        )

        return SQLResult(
            success      = True,
            sql          = sql,
            metric_key   = intent.metric_key,
            filters_used = active_slicers,
        )

    def _build_where_clause(self, active_slicers: dict) -> str:
        """
        # * Build a WHERE clause string from active slicer dict.

        Args:
            active_slicers : Dict of {field: value} — already cleaned

        Returns:
            "WHERE ..." string or empty string if no filters.
        """
        if not active_slicers:
            return ""

        conditions = []
        for field, value in active_slicers.items():
            if isinstance(value, list):
                vals = ", ".join(f"'{v}'" for v in value)
                conditions.append(f'"{field}" IN ({vals})')
            elif isinstance(value, int):
                conditions.append(f'"{field}" = {value}')
            else:
                conditions.append(f'"{field}" = \'{value}\'')

        return "WHERE " + "\n  AND ".join(conditions)


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_generator_instance: Optional[SQLGenerator] = None


def get_sql_generator() -> SQLGenerator:
    """
    # * Returns the shared SQLGenerator instance.
    # * Creates it on first call — reused on every subsequent call.

    Usage:
        from models.sql_generator import get_sql_generator
        generator = get_sql_generator()
        result    = generator.generate(intent_result, slicer_answers)
        if result.success:
            df = db_client.run_query(result.sql)
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SQLGenerator()
    return _generator_instance