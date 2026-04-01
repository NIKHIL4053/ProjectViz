"""
database/connection.py
----------------------
# * PostgreSQL connection pool for the Loan Dashboard project.
# * Uses SQLAlchemy for connection pooling and psycopg2 as the driver.
# * Single entry point for all database connections — no other file
# * creates connections directly.

# ? Why SQLAlchemy + psycopg2 instead of just psycopg2?
# ? SQLAlchemy gives us connection pooling out of the box.
# ? With 89K rows and multiple chart queries per session,
# ? re-connecting on every query would be slow.
# ? The pool keeps connections warm and reuses them.

# ? Why not use SQLAlchemy ORM?
# ? We only need raw SQL execution — no ORM needed.
# ? sql_generator.py builds SQL strings, client.py executes them.
# ? Keeping it lightweight.

Exports:
    - DatabaseConnection    : Main class — engine, pool, ping, execute
    - get_connection()      : Singleton accessor
"""

from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from config import (
    PG_HOST,
    PG_PORT,
    PG_DATABASE,
    PG_USER,
    PG_PASSWORD,
    PG_SCHEMA,
    PG_TABLE_NAME,
    PG_DSN,
    PG_MAX_ROWS,
    USE_MOCK_DATA,
)
from utils.logger import get_logger, get_db_logger
from utils.benchmark import benchmark

# * Module level loggers
log    = get_logger(__name__)
db_log = get_db_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * DATABASE CONNECTION CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseConnection:
    """
    # * Manages the PostgreSQL connection pool for the entire app session.
    # * One instance shared across all database calls via get_connection().

    # ? Connection pool settings:
    # ?   pool_size     = 3  — 3 persistent connections kept warm
    # ?   max_overflow  = 2  — 2 extra connections allowed under heavy load
    # ?   pool_timeout  = 30 — wait max 30s for a free connection
    # ?   pool_recycle  = 1800 — recycle connections every 30 mins
    # ?                          prevents stale connection errors in Docker
    """

    def __init__(self):
        self._engine = None
        self._ready  = False

    # ── Initialise ────────────────────────────────────────────────────────────

    def initialise(self) -> bool:
        """
        # * Create the SQLAlchemy engine and connection pool.
        # * Safe to call multiple times — skips if already initialised.
        # * Called once at app startup from app.py.

        Returns:
            True if engine created successfully, False if connection failed.
        """
        if self._ready:
            log.debug("[db] Already initialised — skipping")
            return True

        if USE_MOCK_DATA:
            log.info("[db] USE_MOCK_DATA=true — skipping real DB connection")
            self._ready = True
            return True

        db_log.info(
            f"[db] Connecting | "
            f"host={PG_HOST} | port={PG_PORT} | "
            f"db={PG_DATABASE} | user={PG_USER}"
        )

        with benchmark("db_initialise"):
            try:
                self._engine = create_engine(
                    PG_DSN,
                    poolclass    = QueuePool,
                    pool_size    = 3,
                    max_overflow = 2,
                    pool_timeout = 30,
                    pool_recycle = 1800,   # * recycle every 30 mins — prevents Docker stale conn
                    pool_pre_ping= True,   # * ping before using — detects dropped connections
                    echo         = False,  # * set True to log all SQL to console during debugging
                    connect_args = {
                        "connect_timeout": 10,
                        "options":         f"-csearch_path={PG_SCHEMA}",
                    }
                )

                # * Test connection immediately
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                self._ready = True
                db_log.info(
                    f"[db] Connected successfully | "
                    f"pool_size=3 | schema={PG_SCHEMA}"
                )
                return True

            except OperationalError as e:
                db_log.error(
                    f"[db] Connection failed — is Docker running? | "
                    f"host={PG_HOST}:{PG_PORT} | error={e}"
                )
                self._ready = False
                return False

            except Exception as e:
                db_log.error(f"[db] Unexpected error during init | error={e}")
                self._ready = False
                return False

    # ── Ping ──────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        # * Check if the database is reachable right now.
        # * Used by sidebar status indicator and app startup health check.

        Returns:
            True if DB responds, False otherwise.
        """
        if USE_MOCK_DATA:
            return True

        if self._engine is None:
            return False

        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ── Execute query → DataFrame ─────────────────────────────────────────────

    def execute(self, sql: str, params: dict = {}) -> pd.DataFrame:
        """
        # * Execute a SQL query and return results as a pandas DataFrame.
        # * All queries go through this single method — never use engine directly.

        # ? Why return DataFrame and not raw rows?
        # ? Everything downstream (charts, filters, session) expects DataFrames.
        # ? Converting here once is cleaner than converting in every caller.

        Args:
            sql    : SQL SELECT string generated by sql_generator.py
            params : Optional dict of bind parameters for safe parameterisation
                     e.g. {"branch": "PUNE"} with sql containing ":branch"

        Returns:
            pandas DataFrame with query results.
            Empty DataFrame if query returns no rows (not an error).

        Raises:
            RuntimeError if query fails — caller should catch this.

        Example:
            df = db.execute(
                'SELECT "Branch", COUNT(*) FROM loan_dashboard GROUP BY "Branch"'
            )
        """
        if not self._ready:
            raise RuntimeError(
                "Database not initialised — call initialise() first or check Docker"
            )

        if not sql or not sql.strip():
            raise ValueError("execute() received empty SQL string")

        # ! Safety check — only allow SELECT statements
        # ! Prevents accidental data modification from model-generated SQL
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            raise ValueError(
                f"Only SELECT queries are allowed | received: {sql[:80]}"
            )

        with benchmark("db_execute"):
            db_log.info(
                f"[db] Executing query | "
                f"preview=\"{sql.strip()[:120].replace(chr(10), ' ')}\""
            )
            start_chars = len(sql)

            try:
                with self._engine.connect() as conn:
                    result = pd.read_sql(
                        sql    = text(sql),
                        con    = conn,
                        params = params if params else None,
                    )

                # * Safety cap — warn if result is unexpectedly large
                if len(result) > PG_MAX_ROWS:
                    db_log.warning(
                        f"[db] Result exceeds PG_MAX_ROWS | "
                        f"rows={len(result)} | cap={PG_MAX_ROWS} | "
                        f"truncating to {PG_MAX_ROWS}"
                    )
                    result = result.head(PG_MAX_ROWS)

                db_log.info(
                    f"[db] Query OK | "
                    f"rows={len(result)} | cols={len(result.columns)} | "
                    f"columns={list(result.columns)}"
                )
                return result

            except SQLAlchemyError as e:
                db_log.error(
                    f"[db] Query FAILED | "
                    f"error={type(e).__name__}: {str(e)[:200]} | "
                    f"sql_preview=\"{sql[:120]}\""
                )
                raise RuntimeError(f"Database query failed: {str(e)[:200]}")

            except Exception as e:
                db_log.error(f"[db] Unexpected query error | error={e}")
                raise RuntimeError(f"Unexpected database error: {str(e)[:200]}")

    # ── Schema inspection ─────────────────────────────────────────────────────

    def get_table_columns(self, table_name: str = PG_TABLE_NAME) -> list[dict]:
        """
        # * Fetch column names and data types from the PostgreSQL information schema.
        # * Used at startup to verify the table exists and columns match expectations.

        Args:
            table_name : Table to inspect (default: PG_TABLE_NAME from config)

        Returns:
            List of {"column_name": str, "data_type": str}
            Empty list if table not found or connection fails.
        """
        if not self._ready or USE_MOCK_DATA:
            return []

        sql = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name   = :table
            ORDER BY ordinal_position
        """
        try:
            with self._engine.connect() as conn:
                result = pd.read_sql(
                    sql    = text(sql),
                    con    = conn,
                    params = {"schema": PG_SCHEMA, "table": table_name},
                )
            cols = result.to_dict(orient="records")
            db_log.info(
                f"[db] Table inspection | "
                f"table={PG_SCHEMA}.{table_name} | "
                f"columns={len(cols)}"
            )
            return cols
        except Exception as e:
            db_log.error(f"[db] get_table_columns failed | error={e}")
            return []

    def table_exists(self, table_name: str = PG_TABLE_NAME) -> bool:
        """
        # * Check if the loan_dashboard table exists in the database.
        # * Called at startup — surfaces a clear error if table is missing.

        Args:
            table_name : Table to check (default: PG_TABLE_NAME from config)

        Returns:
            True if table exists, False otherwise.
        """
        if not self._ready or USE_MOCK_DATA:
            return True  # * In mock mode, always pretend table exists

        sql = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_name   = :table
            )
        """
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(sql),
                    {"schema": PG_SCHEMA, "table": table_name}
                )
                exists = result.scalar()
            db_log.info(f"[db] table_exists | {PG_SCHEMA}.{table_name} = {exists}")
            return bool(exists)
        except Exception as e:
            db_log.error(f"[db] table_exists check failed | error={e}")
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    def status_summary(self) -> dict:
        """
        # * Returns a status dict for the sidebar health indicator.
        # * Checks connection, table existence, and pool stats.

        Returns:
            {
                "connected"    : bool,
                "table_exists" : bool,
                "host"         : str,
                "database"     : str,
                "table"        : str,
                "mock_mode"    : bool,
            }
        """
        connected = self.ping()
        return {
            "connected":    connected,
            "table_exists": self.table_exists() if connected else False,
            "host":         f"{PG_HOST}:{PG_PORT}",
            "database":     PG_DATABASE,
            "schema":       PG_SCHEMA,
            "table":        PG_TABLE_NAME,
            "mock_mode":    USE_MOCK_DATA,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """# * True if pool is initialised and ready to accept queries."""
        return self._ready

    def dispose(self):
        """
        # * Close all connections in the pool.
        # * Called on app shutdown — not normally needed in Streamlit
        # ? but useful for clean teardown in testing.
        """
        if self._engine:
            self._engine.dispose()
            db_log.info("[db] Connection pool disposed")
        self._ready = False


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_db_instance: Optional[DatabaseConnection] = None


def get_connection() -> DatabaseConnection:
    """
    # * Returns the shared DatabaseConnection instance.
    # * Creates and initialises it on first call.
    # * All subsequent calls return the same pooled connection.

    Usage in any file:
        from database.connection import get_connection
        db = get_connection()
        df = db.execute('SELECT * FROM loan_dashboard WHERE "Op bucket" = \'NPA\'')
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
        _db_instance.initialise()
    return _db_instance