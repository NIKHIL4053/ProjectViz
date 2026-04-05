"""
database/client.py
------------------
# * High-level database client for the Loan Dashboard project.
# * Sits between sql_generator.py and connection.py.
# * Adds query validation, column cleaning, mock routing, and error formatting
# * on top of the raw connection pool in connection.py.

# ? Why two files (connection.py + client.py) instead of one?
# ? connection.py owns the pool — it knows nothing about the business domain.
# ? client.py owns the query lifecycle — validates SQL, cleans column names,
# ? routes to mock, formats errors for the UI.
# ? Separation makes each file testable independently.

# ? Flow for every query:
# ?   sql_generator.py builds SQL string
# ?        ↓
# ?   client.py validates → routes to real DB or mock
# ?        ↓
# ?   connection.py executes against PostgreSQL pool
# ?        ↓
# ?   client.py cleans column names → returns DataFrame

Exports:
    - QueryResult       : Dataclass wrapping every query response
    - DatabaseClient    : Main class
    - get_client()      : Singleton accessor
"""

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    USE_MOCK_DATA,
    PG_TABLE_NAME,
    PG_SCHEMA,
    PG_MAX_ROWS,
)
from database.connection import get_connection
from utils.logger import get_logger, get_db_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string

# * Module level loggers
log    = get_logger(__name__)
db_log = get_db_logger(__name__)

# * Forbidden SQL keywords — model should never generate these
# ! If any appear in the SQL, query is rejected before hitting the database
_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
    "ALTER", "CREATE", "GRANT", "REVOKE", "EXECUTE",
}


# ──────────────────────────────────────────────────────────────────────────────
# * QUERY RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    # * Wraps every database query response in a consistent structure.
    # * Always check .success before using .dataframe.

    Attributes:
        success      : True if query ran and returned data
        dataframe    : pandas DataFrame with results (empty if no rows)
        row_count    : Number of rows returned
        col_count    : Number of columns returned
        columns      : List of column names
        sql_executed : The actual SQL that was run (cleaned)
        source       : "database" or "mock" — which source served this result
        error        : Error message if success=False, else None
    """
    success:      bool
    dataframe:    pd.DataFrame          = None
    row_count:    int                   = 0
    col_count:    int                   = 0
    columns:      list[str]             = None
    sql_executed: str                   = ""
    source:       str                   = "database"
    error:        Optional[str]         = None

    def __post_init__(self):
        if self.dataframe is None:
            self.dataframe = pd.DataFrame()
        if self.columns is None:
            self.columns = []

    @property
    def failed(self) -> bool:
        """# * True if the query failed."""
        return not self.success

    @property
    def is_empty(self) -> bool:
        """# * True if query succeeded but returned zero rows."""
        return self.success and self.row_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# * DATABASE CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseClient:
    """
    # * High-level client for executing SQL queries against loan_dashboard table.
    # * All model files and UI files call this — never connection.py directly.
    """

    def __init__(self):
        self._conn = get_connection()
        log.info(
            f"[db_client] Initialised | "
            f"mock_mode={USE_MOCK_DATA} | "
            f"table={PG_SCHEMA}.{PG_TABLE_NAME}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_query(self, sql: str) -> QueryResult:
        """
        # * Execute a SQL query and return a structured QueryResult.
        # * Routes to mock data if USE_MOCK_DATA=true.
        # * Validates, executes, cleans, and returns result.

        Args:
            sql : SQL SELECT string from sql_generator.py

        Returns:
            QueryResult — always check .success before using .dataframe.

        Example:
            client = get_client()
            result = client.run_query(
                'SELECT "Branch", COUNT(DISTINCT "Cust ID") FROM loan_dashboard
                 WHERE "Bounce status" IN (\'Tech\',\'Non Tech\') GROUP BY "Branch"'
            )
            if result.success:
                df = result.dataframe
        """
        if not sql or not sql.strip():
            return QueryResult(
                success=False,
                error="Empty SQL string received"
            )

        # * Step 1 — Validate SQL
        validation_error = self._validate_sql(sql)
        if validation_error:
            db_log.error(f"[db_client] SQL validation failed | {validation_error}")
            return QueryResult(
                success      = False,
                sql_executed = sql,
                error        = validation_error
            )

        # * Step 2 — Route to mock or real DB
        if USE_MOCK_DATA:
            return self._run_mock(sql)
        else:
            return self._run_real(sql)

    # ── Real database execution ───────────────────────────────────────────────

    def _run_real(self, sql: str) -> QueryResult:
        """
        # * Execute against the real PostgreSQL database via connection pool.

        Args:
            sql : Validated SQL string

        Returns:
            QueryResult with DataFrame from PostgreSQL.
        """
        with benchmark("db_fetch"):
            try:
                df = self._conn.execute(sql)
                df = self._clean_columns(df)

                db_log.info(
                    f"[db_client] Real query OK | "
                    f"rows={len(df)} | cols={len(df.columns)}"
                )

                return QueryResult(
                    success      = True,
                    dataframe    = df,
                    row_count    = len(df),
                    col_count    = len(df.columns),
                    columns      = list(df.columns),
                    sql_executed = sql,
                    source       = "database",
                )

            except RuntimeError as e:
                db_log.error(f"[db_client] Real query FAILED | error={e}")
                return QueryResult(
                    success      = False,
                    sql_executed = sql,
                    source       = "database",
                    error        = str(e),
                )

            except Exception as e:
                db_log.error(f"[db_client] Unexpected error | error={e}")
                return QueryResult(
                    success      = False,
                    sql_executed = sql,
                    source       = "database",
                    error        = f"Unexpected error: {str(e)[:200]}",
                )

    # ── Mock routing ──────────────────────────────────────────────────────────

    def _run_mock(self, sql: str) -> QueryResult:
        """
        # * Route to mock.py when USE_MOCK_DATA=true.
        # * Detects query intent from SQL keywords and returns realistic fake data.

        Args:
            sql : SQL string (used to detect intent for mock routing)

        Returns:
            QueryResult with mock DataFrame.
        """
        with benchmark("db_fetch"):
            try:
                from database.mock import get_mock_data
                df = get_mock_data(sql)
                df = self._clean_columns(df)

                db_log.info(
                    f"[db_client] Mock query OK | "
                    f"rows={len(df)} | cols={len(df.columns)}"
                )

                return QueryResult(
                    success      = True,
                    dataframe    = df,
                    row_count    = len(df),
                    col_count    = len(df.columns),
                    columns      = list(df.columns),
                    sql_executed = sql,
                    source       = "mock",
                )

            except Exception as e:
                db_log.error(f"[db_client] Mock routing failed | error={e}")
                return QueryResult(
                    success      = False,
                    sql_executed = sql,
                    source       = "mock",
                    error        = f"Mock data error: {str(e)[:200]}",
                )

    # ── SQL validation ────────────────────────────────────────────────────────

    def _validate_sql(self, sql: str) -> Optional[str]:
        """
        # * Validate SQL before sending to database.
        # * Returns error string if invalid, None if valid.

        # ! Hard rules:
        # !   1. Must start with SELECT or WITH (CTEs allowed)
        # !   2. Must not contain forbidden modification keywords
        # !   3. Must reference loan_dashboard table (sanity check)
        # !   4. Must not be suspiciously long (possible prompt injection)

        Args:
            sql : SQL string to validate

        Returns:
            Error message string if invalid, None if valid.
        """
        clean = sql.strip()
        upper = clean.upper()

        # * Rule 1 — must be SELECT or WITH
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return f"SQL must start with SELECT or WITH — got: {clean[:40]}"

        # * Rule 2 — check for forbidden keywords
        # ? Use word boundary regex to avoid false positives
        # ? e.g. "TRUNCATE" in a column value should not trigger this
        words = set(re.findall(r'\b[A-Z]+\b', upper))
        forbidden_found = words & _FORBIDDEN_KEYWORDS
        if forbidden_found:
            return f"SQL contains forbidden keywords: {forbidden_found}"

        # * Rule 3 — must reference the loan_dashboard table
        # ? Allows both quoted and unquoted table references
        if "loan_dashboard" not in clean.lower():
            db_log.warning(
                f"[db_client] SQL does not reference loan_dashboard | "
                f"sql_preview=\"{truncate_string(clean, 80)}\""
            )
            # ! Warning only — not a hard block, CTE queries may alias the table

        # * Rule 4 — length sanity check (> 5000 chars is suspicious)
        if len(clean) > 5000:
            return f"SQL is suspiciously long ({len(clean)} chars) — possible injection"

        return None  # * All checks passed

    # ── Column name cleaning ──────────────────────────────────────────────────

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if df is None or df.empty:
            return df
    
        # Map PostgreSQL snake_case → display names the rest of the app expects
        rename_map = {
            "loanappno":                    "loanappno",
            "dpd":                          "dpd",
            "dpd_casewise":                 "dpd_casewise",
            "day_wise_asset_classification":"day_wise_asset_classification",
            "tod":                          "TOD",
            "bal_prin":                     "bal_prin",
            "next_date":                    "Next Date",
            "bounce_status":                "Bounce status",
            "emi_increase":                 "Emi increase",
            "cust_id":                      "Cust ID",
            "risk_npa":                     "Risk NPA",
            "op_bucket":                    "Op bucket",
            "closing_bucket":               "Closing bucket",
            "cust_wise_status":             "Cust wise status",
            "due_date":                     "Due Date",
            "installment_no":               "Installment No",
            "branch":                       "Branch",
            "type_of_arrangement":          "Type of Arrangement",
            "scheme":                       "Scheme",
            "payment_mode":                 "Payment mode",
            "payment_day_bucket":           "Payment Day bucket",
            "digital_cash":                 "Digital/cash",
            "loan_level_bounce":            "Loan level bounce",
            "cust_level_bounce":            "Cust level bounce",
            "region":                       "Region",
            "mob_bucket":                   "MOB Bucket",
            "overdue_amount":               "overdue_amount",
            "bounce_charges":               "Bounce charges",
            "bounce_charge_collected":      "Bounce charge collected",
            "visited":                      "Visited",
            "allocation_1":                 "Allocation 1",
            "sh":                           "SH",
            "tl":                           "TL",
            "allocated_or_not":             "Allocated or not",
            "visit_or_not":                 "Visit or not",
            "visit_count":                  "Visit Count",
            "add_npa":                      "Add NPA",
            "coll_sales":                   "Coll/Sales",
            "npa_origination_date":         "NPA_Origination_Date",
            "npa_mob":                      "NPA MOB",
            "portfolio_new":                "Portfolio new",
            "transaction_type":             "Transaction Type",
            "product":                      "Product",
        }
    
        # Apply rename only for columns that exist in the DataFrame
        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        if existing_renames:
            df = df.rename(columns=existing_renames)
    
        return df
    
    # ── Health check ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """
        # * True if the database is reachable and ready to accept queries.
        # * Used by app.py startup check and sidebar status.
        """
        if USE_MOCK_DATA:
            return True
        return self._conn.ping()

    def get_status(self) -> dict:
        """
        # * Returns full status dict for sidebar display.
        # * Includes connection health, table existence, and mock mode flag.
        """
        return self._conn.status_summary()

    # ── Convenience query helpers ─────────────────────────────────────────────

    def get_distinct_values(
        self,
        column:     str,
        table:      str     = PG_TABLE_NAME,
        max_values: int     = 100,
    ) -> list[str]:
        """
        # * Fetch distinct values for a column — used to populate filter dropdowns.
        # * Returns empty list on failure.

        Args:
            column     : Column name (will be double-quoted automatically)
            table      : Table name (default: loan_dashboard)
            max_values : Cap on distinct values returned

        Returns:
            Sorted list of distinct string values.

        Example:
            branches = client.get_distinct_values("Branch")
            # → ["AHMEDABAD", "BANGALORE", "CHENNAI", ...]
        """
        sql = (
            f'SELECT DISTINCT "{column}" '
            f'FROM {table} '
            f'WHERE "{column}" IS NOT NULL '
            f'ORDER BY "{column}" '
            f'LIMIT {max_values}'
        )
        result = self.run_query(sql)
        if result.success and not result.is_empty:
            return result.dataframe.iloc[:, 0].astype(str).tolist()
        return []

    def get_row_count(self, table: str = PG_TABLE_NAME) -> int:
        """
        # * Returns the total row count of the table.
        # * Used in KPI cards and startup validation.

        Args:
            table : Table name (default: loan_dashboard)

        Returns:
            Integer row count, or 0 on failure.
        """
        result = self.run_query(f"SELECT COUNT(*) AS count FROM {table}")
        if result.success and not result.is_empty:
            return int(result.dataframe.iloc[0, 0])
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_client_instance: Optional[DatabaseClient] = None


def get_client() -> DatabaseClient:
    """
    # * Returns the shared DatabaseClient instance.
    # * Creates it on first call — reused on every subsequent call.

    Usage in any file:
        from database.client import get_client
        db     = get_client()
        result = db.run_query(sql_string)
        if result.success:
            df = result.dataframe
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = DatabaseClient()
    return _client_instance