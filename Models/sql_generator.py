"""
models/sql_generator.py
-----------------------
Pipeline Step 3 — SQL Generation.
Takes IntentResult + slicer answers and generates a valid PostgreSQL SELECT query.

Column naming:
  PostgreSQL table uses snake_case:  op_bucket, cust_id, bounce_status etc.
  Clarifier uses display names:      "Op bucket", "Cust ID", "Bounce status" etc.
  This file translates display → snake_case before building SQL.
  database/client.py rename_map translates snake_case → display names in the
  returned DataFrame so the rest of the app works unchanged.

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

log       = get_logger(__name__)
model_log = get_model_logger(__name__)

_FORBIDDEN = {
    "INSERT", "UPDATE", "DELETE", "DROP",
    "TRUNCATE", "ALTER", "CREATE", "GRANT",
    "REVOKE", "EXECUTE",
}


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY NAME → DB COLUMN MAP
# Clarifier produces display names. SQL must use snake_case DB column names.
# ──────────────────────────────────────────────────────────────────────────────

FIELD_TO_DB = {
    # Identifiers
    "loanappno":                     "loanappno",
    "Cust ID":                       "cust_id",
    # Delinquency
    "dpd":                           "dpd",
    "dpd_casewise":                  "dpd_casewise",
    "day_wise_asset_classification": "day_wise_asset_classification",
    # Financial
    "TOD":                           "tod",
    "bal_prin":                      "bal_prin",
    "overdue_amount":                "overdue_amount",
    "Bounce charges":                "bounce_charges",
    "Bounce charge collected":       "bounce_charge_collected",
    # Dates
    "Next Date":                     "next_date",
    "Due Date":                      "due_date",
    "NPA_Origination_Date":          "npa_origination_date",
    # Buckets
    "Op bucket":                     "op_bucket",
    "Closing bucket":                "closing_bucket",
    "Cust wise status":              "cust_wise_status",
    # Payment
    "Bounce status":                 "bounce_status",
    "Emi increase":                  "emi_increase",
    "Payment mode":                  "payment_mode",
    "Payment Day bucket":            "payment_day_bucket",
    "Digital/cash":                  "digital_cash",
    "Loan level bounce":             "loan_level_bounce",
    "Cust level bounce":             "cust_level_bounce",
    # Geography
    "Region":                        "region",
    "Branch":                        "branch",
    # Portfolio
    "Type of Arrangement":           "type_of_arrangement",
    "Portfolio new":                 "portfolio_new",
    "Transaction Type":              "transaction_type",
    "Product":                       "product",
    "Scheme":                        "scheme",
    # Collection team
    "Allocation 1":                  "allocation_1",
    "SH":                            "sh",
    "TL":                            "tl",
    "Allocated or not":              "allocated_or_not",
    "Visit or not":                  "visit_or_not",
    "Visit Count":                   "visit_count",
    "Visited":                       "visited",
    # Risk
    "Risk NPA":                      "risk_npa",
    "Add NPA":                       "add_npa",
    "NPA MOB":                       "npa_mob",
    # Loan
    "MOB Bucket":                    "mob_bucket",
    "Installment No":                "installment_no",
    "Coll/Sales":                    "coll_sales",
}


def _to_db(field: str) -> Optional[str]:
    """Convert display field name to DB snake_case column name."""
    if not field:
        return None
    return FIELD_TO_DB.get(field, field.lower().replace(" ", "_").replace("/", "_"))


# ──────────────────────────────────────────────────────────────────────────────
# SQL RESULT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SQLResult:
    """
    Structured output from the SQL generator.

    Attributes:
        success      : True if SQL was generated and validated
        sql          : PostgreSQL SELECT query string
        metric_key   : Which metric this SQL calculates
        filters_used : Active slicer filters applied (snake_case keys)
        error        : Error message if success=False
    """
    success:      bool
    sql:          str           = ""
    metric_key:   str           = ""
    filters_used: dict          = None
    error:        Optional[str] = None

    def __post_init__(self):
        if self.filters_used is None:
            self.filters_used = {}

    @property
    def failed(self) -> bool:
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """
You are an expert PostgreSQL query writer for a loan collection portfolio system.
Write a single valid PostgreSQL SELECT query based on the metric and filters below.

DATABASE:
- Table : {table_name}
- Schema: {schema}

COLUMN NAMES — all snake_case, no quotes needed for column names:
  Identifiers : loanappno, cust_id
  Delinquency : dpd, dpd_casewise, day_wise_asset_classification
  Financial   : bal_prin, tod, overdue_amount, bounce_charges, bounce_charge_collected
  Dates       : next_date, due_date, npa_origination_date
  Buckets     : op_bucket, closing_bucket, cust_wise_status
  Payment     : bounce_status, emi_increase, payment_mode, payment_day_bucket,
                digital_cash, loan_level_bounce, cust_level_bounce
  Geography   : region, branch
  Portfolio   : type_of_arrangement, portfolio_new, transaction_type, product, scheme
  Collection  : allocation_1, sh, tl, allocated_or_not, visit_or_not,
                visit_count, visited, coll_sales
  Risk        : risk_npa, add_npa, npa_mob
  Loan        : mob_bucket, installment_no

SQL RULES — follow every rule exactly:
1. Use snake_case column names with NO double-quoting
2. Output column ALIASES should be human-readable in double quotes:
   e.g. COUNT(DISTINCT cust_id) AS "Customer Count"
3. ALWAYS use COUNT(DISTINCT cust_id) for customer-level metrics
4. Use COUNT(loanappno) for loan-level metrics
5. Use ROUND(..., 2) for all percentages and ratios
6. Use NULLIF(denominator, 0) to prevent division by zero
7. Percentage: ROUND(numerator * 100.0 / NULLIF(denominator, 0), 2)
8. Date filter: next_date = 'YYYY-MM-DD'
9. Multiple values: op_bucket IN ('NPA', '60-89 DPD')
10. Always end grouped queries with ORDER BY
11. Return ONLY the SQL — no explanation, no markdown, no backticks

METRIC RULES:
- Bounce     : WHERE bounce_status IN ('Tech', 'Non Tech') — NEVER include 'PAID'
- Resolution : WHERE cust_wise_status = 'Norm'
- Coverage   : WHERE visit_or_not = 'Visited' AND allocated_or_not = 'Allocated'
  Denom      : WHERE allocated_or_not = 'Allocated'
- Intensity  : ROUND(SUM(visit_count)::NUMERIC / NULLIF(COUNT(DISTINCT cust_id), 0), 2)
- NPA        : WHERE op_bucket = 'NPA'
- New NPA    : WHERE add_npa = 1
- Outstanding: SUM(bal_prin)
- Overdue    : SUM(tod)

RELEVANT SQL PATTERNS:
{context}

FILTERS TO APPLY IN WHERE CLAUSE:
{filters_block}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# SQL GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class SQLGenerator:
    """
    Step 3 — generates PostgreSQL SQL from IntentResult + slicer answers.
    1. Tries Qwen Coder 14B with SQL patterns injected
    2. Falls back to hardcoded rule-based SQL if model fails
    """

    def __init__(self):
        self._client     = get_client()
        self._dictionary = get_dictionary()
        log.info("[sql_generator] Initialised")

    # ── Main entry ────────────────────────────────────────────────────────────

    def generate(self, intent: IntentResult, slicers: dict) -> SQLResult:
        """
        Generate a PostgreSQL SELECT query.

        Args:
            intent  : IntentResult from analyzer.py
            slicers : {display_field: value} from clarifier — "All" ignored

        Returns:
            SQLResult with .sql, or .failed=True with .error
        """
        if intent.failed:
            return SQLResult(success=False, error="Intent analysis failed")

        # Translate display field names → snake_case DB columns
        active_slicers = self._build_active_slicers(slicers)

        # Translate group_by display name → snake_case
        group_by_db = _to_db(intent.group_by) if intent.group_by else None

        log.info(
            f"[sql_generator] Generating | metric='{intent.metric_key}' | "
            f"group_by='{group_by_db}' | slicers={active_slicers}"
        )

        with benchmark("sql_generation", query=intent.raw_question):

            filters_block = self._format_filters_block(active_slicers)
            context       = self._fetch_context(intent)

            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                table_name    = PG_TABLE_NAME,
                schema        = PG_SCHEMA,
                context       = context,
                filters_block = filters_block,
            )

            user_message = self._build_user_message(intent, group_by_db, active_slicers)
            model_log.debug(f"[sql_generation] ~{estimate_tokens(system_prompt)} tokens")

            response: ModelResponse = self._client.call_coder(
                system      = system_prompt,
                user        = user_message,
                step        = "sql_generation",
                expect_json = False,
            )

            if response.failed:
                log.warning(f"[sql_generator] Model failed → rule-based | {response.error}")
                return self._rule_based_sql(intent, group_by_db, active_slicers)

            sql = self._clean_sql(response.content)
            err = self._validate_sql(sql)

            if err:
                model_log.warning(f"[sql_generation] Validation failed → fallback | {err}")
                return self._rule_based_sql(intent, group_by_db, active_slicers)

            model_log.info(
                f"[sql_generation] OK | metric='{intent.metric_key}' | "
                f"preview=\"{truncate_string(sql, 120)}\""
            )
            return SQLResult(
                success=True, sql=sql,
                metric_key=intent.metric_key, filters_used=active_slicers,
            )

    # ── Slicer processing ─────────────────────────────────────────────────────

    def _build_active_slicers(self, slicers: dict) -> dict:
        """
        Filter All/empty, translate display names → DB columns,
        convert date strings to ISO, normalise Add NPA labels.
        Returns {db_column: value}.
        """
        active = {}
        for display_field, value in slicers.items():
            if not value:
                continue
            if isinstance(value, str) and value.lower() in ("all", ""):
                continue
            if isinstance(value, list) and len(value) == 0:
                continue

            db_col = _to_db(display_field)

            # Convert "Feb 2026" → "2026-02-01"
            if db_col == "next_date" and isinstance(value, str):
                iso = self._month_to_iso(value)
                if iso:
                    active[db_col] = iso
                continue

            # Normalise Add NPA: "1 (New NPA)" → 1
            if db_col == "add_npa":
                active[db_col] = 1 if "1" in str(value) else 0
                continue

            active[db_col] = value

        return active

    def _month_to_iso(self, s: str) -> Optional[str]:
        """Convert "Feb 2026" → "2026-02-01"."""
        mm = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            p = s.strip().split()
            if len(p) == 2:
                m = mm.get(p[0].lower()[:3])
                y = int(p[1])
                if m and y:
                    return f"{y}-{m:02d}-01"
        except Exception:
            pass
        return None

    def _format_filters_block(self, active_slicers: dict) -> str:
        """Human-readable filter block for system prompt."""
        if not active_slicers:
            return "No filters — query covers all data"
        lines = []
        for col, value in active_slicers.items():
            if isinstance(value, list):
                vals = ", ".join(f"'{v}'" for v in value)
                lines.append(f"- {col} IN ({vals})")
            elif isinstance(value, int):
                lines.append(f"- {col} = {value}")
            else:
                lines.append(f"- {col} = '{value}'")
        return "\n".join(lines)

    def _build_user_message(
        self,
        intent:         IntentResult,
        group_by_db:    Optional[str],
        active_slicers: dict,
    ) -> str:
        group_str = (
            f"GROUP BY {group_by_db}" if group_by_db
            else "No GROUP BY — return a single aggregate row"
        )
        return f"""Generate a PostgreSQL SELECT query for:

Question   : "{intent.raw_question}"
Metric     : {intent.metric} ({intent.metric_key})
{group_str}
Aggregation: {intent.aggregation}
Granularity: {intent.granularity} level

WHERE filters to include:
{self._format_filters_block(active_slicers)}

SQL pattern hint: {intent.sql_pattern_hint or 'none'}

Return ONLY the SQL — no explanation, no markdown, no backticks."""

    # ── Context fetch ─────────────────────────────────────────────────────────

    def _fetch_context(self, intent: IntentResult) -> str:
        if not self._dictionary.is_ready:
            return ""
        q            = f"{intent.raw_question} {intent.metric} {intent.metric_key}"
        sql_chunks   = self._dictionary.search(q, top_k=4, source="sql_patterns")
        logic_chunks = self._dictionary.search(q, top_k=2, source="business_logic")
        parts = []
        if sql_chunks:
            parts.append("SQL PATTERNS:\n" + "\n\n".join(sql_chunks))
        if logic_chunks:
            parts.append("BUSINESS LOGIC:\n" + "\n\n".join(logic_chunks))
        return "\n\n".join(parts)

    # ── Clean + validate ──────────────────────────────────────────────────────

    def _clean_sql(self, raw: str) -> str:
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                clean = parts[1]
                if clean.lower().startswith(("sql", "postgresql", "pgsql")):
                    clean = clean[clean.index("\n") + 1:]
        return clean.strip().rstrip(";").strip()

    def _validate_sql(self, sql: str) -> Optional[str]:
        if not sql or not sql.strip():
            return "Empty SQL"
        upper = sql.strip().upper()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return f"Must start with SELECT — got: {sql[:50]}"
        bad = set(re.findall(r'\b[A-Z]+\b', upper)) & _FORBIDDEN
        if bad:
            return f"Forbidden keywords: {bad}"
        if len(sql) > 5000:
            return f"SQL too long ({len(sql)} chars)"
        return None

    # ── WHERE clause builder ──────────────────────────────────────────────────

    def _build_where_clause(self, active_slicers: dict) -> str:
        """Build WHERE clause. Keys are already snake_case DB column names."""
        if not active_slicers:
            return ""
        conditions = []
        for col, value in active_slicers.items():
            if isinstance(value, list):
                vals = ", ".join(f"'{v}'" for v in value)
                conditions.append(f"{col} IN ({vals})")
            elif isinstance(value, int):
                conditions.append(f"{col} = {value}")
            else:
                conditions.append(f"{col} = '{value}'")
        return "WHERE " + "\n  AND ".join(conditions)

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_sql(
        self,
        intent:      IntentResult,
        group_by_db: Optional[str],
        slicers:     dict,
    ) -> SQLResult:
        """
        Hardcoded correct SQL for every metric type.
        All column names are snake_case to match the DB.
        Aliases are human-readable strings for the UI.
        Always returns valid SQL — never fails.
        """
        log.info(f"[sql_generator] Rule-based | metric={intent.metric_key} | group={group_by_db}")

        table  = PG_TABLE_NAME
        where  = self._build_where_clause(slicers)
        metric = intent.metric_key
        grp    = group_by_db

        # ── Bounce ────────────────────────────────────────────────────────────
        if metric in ("Bounce_Count", "Bounce_Percent"):
            if grp:
                sql = f"""SELECT
  {grp},
  COUNT(DISTINCT CASE WHEN bounce_status IN ('Tech', 'Non Tech') THEN cust_id END) AS "Bounce Count",
  COUNT(DISTINCT cust_id) AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN bounce_status IN ('Tech', 'Non Tech') THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT cust_id), 0), 2
  ) AS "Bounce %"
FROM {table}
{where}
GROUP BY {grp}
ORDER BY "Bounce Count" DESC"""
            else:
                sql = f"""SELECT
  COUNT(DISTINCT CASE WHEN bounce_status IN ('Tech', 'Non Tech') THEN cust_id END) AS "Bounce Count",
  COUNT(DISTINCT cust_id) AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN bounce_status IN ('Tech', 'Non Tech') THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT cust_id), 0), 2
  ) AS "Bounce %"
FROM {table}
{where}"""

        # ── Resolution ────────────────────────────────────────────────────────
        elif metric in ("Resolution", "Resolution_Percent"):
            if grp:
                sql = f"""SELECT
  {grp},
  COUNT(DISTINCT CASE WHEN cust_wise_status = 'Norm' THEN cust_id END) AS "Resolved Count",
  COUNT(DISTINCT cust_id) AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN cust_wise_status = 'Norm' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT cust_id), 0), 2
  ) AS "Resolution %"
FROM {table}
{where}
GROUP BY {grp}
ORDER BY "Resolution %" DESC"""
            else:
                sql = f"""SELECT
  COUNT(DISTINCT CASE WHEN cust_wise_status = 'Norm' THEN cust_id END) AS "Resolved Count",
  COUNT(DISTINCT cust_id) AS "Total Customers",
  ROUND(
    COUNT(DISTINCT CASE WHEN cust_wise_status = 'Norm' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT cust_id), 0), 2
  ) AS "Resolution %"
FROM {table}
{where}"""

        # ── Coverage ──────────────────────────────────────────────────────────
        elif metric in ("Coverage", "Coverage_Percent"):
            if grp:
                sql = f"""SELECT
  {grp},
  COUNT(DISTINCT CASE WHEN visit_or_not = 'Visited' AND allocated_or_not = 'Allocated' THEN cust_id END) AS "Visited Count",
  COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END) AS "Allocated Count",
  ROUND(
    COUNT(DISTINCT CASE WHEN visit_or_not = 'Visited' AND allocated_or_not = 'Allocated' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END), 0), 2
  ) AS "Coverage %"
FROM {table}
{where}
GROUP BY {grp}
ORDER BY "Coverage %" DESC"""
            else:
                sql = f"""SELECT
  COUNT(DISTINCT CASE WHEN visit_or_not = 'Visited' AND allocated_or_not = 'Allocated' THEN cust_id END) AS "Visited Count",
  COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END) AS "Allocated Count",
  ROUND(
    COUNT(DISTINCT CASE WHEN visit_or_not = 'Visited' AND allocated_or_not = 'Allocated' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END), 0), 2
  ) AS "Coverage %"
FROM {table}
{where}"""

        # ── Intensity ─────────────────────────────────────────────────────────
        elif metric == "Intensity":
            if grp:
                sql = f"""SELECT
  {grp},
  ROUND(SUM(visit_count)::NUMERIC / NULLIF(COUNT(DISTINCT cust_id), 0), 2) AS "Avg Visits per Customer",
  SUM(visit_count) AS "Total Visits",
  COUNT(DISTINCT cust_id) AS "Customer Count"
FROM {table}
{where}
GROUP BY {grp}
ORDER BY "Avg Visits per Customer" DESC"""
            else:
                sql = f"""SELECT
  ROUND(SUM(visit_count)::NUMERIC / NULLIF(COUNT(DISTINCT cust_id), 0), 2) AS "Intensity"
FROM {table}
{where}"""

        # ── Bucket movement ───────────────────────────────────────────────────
        elif metric == "Bucket_Movement":
            sql = f"""SELECT
  op_bucket,
  closing_bucket,
  COUNT(loanappno) AS "Loan Count",
  COUNT(DISTINCT cust_id) AS "Customer Count"
FROM {table}
{where}
GROUP BY op_bucket, closing_bucket
ORDER BY op_bucket, "Loan Count" DESC"""

        # ── Bucket distribution ───────────────────────────────────────────────
        elif metric == "Bucket_Distribution":
            sql = f"""SELECT
  op_bucket,
  COUNT(loanappno) AS "Loan Count",
  COUNT(DISTINCT cust_id) AS "Customer Count",
  SUM(bal_prin) AS "Balance Principal",
  SUM(tod) AS "Total Overdue"
FROM {table}
{where}
GROUP BY op_bucket
ORDER BY "Loan Count" DESC"""

        # ── NPA / Add NPA ─────────────────────────────────────────────────────
        elif metric in ("NPA_Count", "Add_NPA_Count"):
            npa_cond  = "add_npa = 1" if metric == "Add_NPA_Count" else "op_bucket = 'NPA'"
            group_col = grp or "region"
            combined  = (where + f"\n  AND {npa_cond}") if where else f"WHERE {npa_cond}"
            sql = f"""SELECT
  {group_col},
  COUNT(loanappno) AS "NPA Count",
  COUNT(DISTINCT cust_id) AS "Customer Count",
  SUM(bal_prin) AS "Outstanding"
FROM {table}
{combined}
GROUP BY {group_col}
ORDER BY "NPA Count" DESC"""

        # ── Portfolio outstanding / overdue ───────────────────────────────────
        elif metric in ("Portfolio_Outstanding", "Total_Overdue"):
            agg_col   = "bal_prin" if metric == "Portfolio_Outstanding" else "tod"
            agg_label = "Balance Principal" if metric == "Portfolio_Outstanding" else "Total Overdue"
            group_col = grp or "op_bucket"
            sql = f"""SELECT
  {group_col},
  SUM({agg_col}) AS "{agg_label}",
  COUNT(loanappno) AS "Loan Count"
FROM {table}
{where}
GROUP BY {group_col}
ORDER BY "{agg_label}" DESC"""

        # ── DPD distribution ──────────────────────────────────────────────────
        elif metric == "DPD_Distribution":
            combined = (where + "\n  AND dpd_casewise IS NOT NULL") if where else "WHERE dpd_casewise IS NOT NULL"
            sql = f"""SELECT
  dpd_casewise,
  op_bucket,
  branch,
  region
FROM {table}
{combined}
ORDER BY dpd_casewise
LIMIT 5000"""

        # ── FE Scorecard ──────────────────────────────────────────────────────
        elif metric == "FE_Scorecard":
            fe_where = (where + "\n  AND allocation_1 <> 'NA'") if where else "WHERE allocation_1 <> 'NA'"
            sql = f"""SELECT
  allocation_1,
  tl,
  ROUND(
    COUNT(DISTINCT CASE WHEN cust_wise_status = 'Norm' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT cust_id), 0), 2
  ) AS "Resolution %",
  ROUND(
    COUNT(DISTINCT CASE WHEN visit_or_not = 'Visited' AND allocated_or_not = 'Allocated' THEN cust_id END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END), 0), 2
  ) AS "Coverage %",
  ROUND(SUM(visit_count)::NUMERIC / NULLIF(COUNT(DISTINCT cust_id), 0), 2) AS "Intensity",
  COUNT(DISTINCT CASE WHEN allocated_or_not = 'Allocated' THEN cust_id END) AS "Allocated Count"
FROM {table}
{fe_where}
GROUP BY allocation_1, tl
ORDER BY "Resolution %" DESC"""

        # ── Payment analysis ──────────────────────────────────────────────────
        elif metric == "Payment_Analysis":
            group_col = grp or "payment_mode"
            sql = f"""SELECT
  {group_col},
  COUNT(DISTINCT cust_id) AS "Customer Count",
  COUNT(loanappno) AS "Loan Count"
FROM {table}
{where}
GROUP BY {group_col}
ORDER BY "Customer Count" DESC"""

        # ── Generic fallback ──────────────────────────────────────────────────
        else:
            group_col = grp or "branch"
            sql = f"""SELECT
  {group_col},
  COUNT(DISTINCT cust_id) AS "Customer Count",
  COUNT(loanappno) AS "Loan Count",
  SUM(bal_prin) AS "Balance Principal"
FROM {table}
{where}
GROUP BY {group_col}
ORDER BY "Customer Count" DESC"""

        sql = sql.strip()
        model_log.info(
            f"[sql_generation] RULE-BASED | metric={metric} | "
            f"preview=\"{truncate_string(sql, 100)}\""
        )
        return SQLResult(
            success=True, sql=sql,
            metric_key=intent.metric_key, filters_used=slicers,
        )


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

_generator_instance: Optional[SQLGenerator] = None


def get_sql_generator() -> SQLGenerator:
    """
    Returns the shared SQLGenerator instance.

    Usage:
        from models.sql_generator import get_sql_generator
        result = get_sql_generator().generate(intent, slicer_answers)
        if result.success:
            df = db_client.run_query(result.sql)
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SQLGenerator()
    return _generator_instance