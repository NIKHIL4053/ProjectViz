"""
core/session.py
---------------
# * Manages Streamlit session state for the entire Loan Dashboard.
# * Single source of truth for all stateful data during a user session.
# * Handles temp data lifecycle — creates, tracks, and auto-deletes on session end.

# ? Why centralise session state here?
# ? Streamlit reruns the entire script on every interaction.
# ? st.session_state is the only way to persist data across reruns.
# ? Without a central manager, session keys get scattered across files
# ? and it becomes impossible to know what state exists at any point.

# ? Temp data lifecycle:
# ?   1. PostgreSQL returns data → saved to Data/uploads/{session_id}_{query_hash}.parquet
# ?   2. Session stays active → data is reused (no re-fetch)
# ?   3. Session ends OR TTL expires → cleanup() deletes the files
# !   Only files in Data/uploads/ are deleted — never logs, context JSONs, or code.

Exports:
    - SessionManager     : Main class — init, get/set state, cleanup
    - get_session()      : Returns the shared SessionManager for this session
"""

import uuid
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from config import PARQUET_COMPRESSION


from config import (
    DATA_TEMP_DIR,
    TEMP_DATA_TTL_MINUTES,
    MAX_CHAT_HISTORY,
)
from utils.logger import get_logger
from utils.benchmark import QueryBenchmark

# * Module level logger
log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * SESSION KEYS
# * All st.session_state keys defined in one place.
# * Import these constants instead of typing key strings manually.
# ! Never use raw strings like st.session_state["my_key"] elsewhere.
# ! Always use these constants to avoid typos causing silent state bugs.
# ──────────────────────────────────────────────────────────────────────────────

class Keys:
    """
    # * Namespace for all Streamlit session state key constants.
    # * Import and use Keys.CHAT_HISTORY instead of the raw string "chat_history".
    """
    # * Session identity
    SESSION_ID          = "session_id"
    SESSION_START_TIME  = "session_start_time"
    SESSION_INITIALISED = "session_initialised"

    # * Current query pipeline state
    CURRENT_QUESTION    = "current_question"       # raw user question
    CURRENT_INTENT      = "current_intent"         # dict from analyzer.py
    CURRENT_SLICERS     = "current_slicers"        # dict from clarifier.py
    CURRENT_SQL         = "current_sql"            # SQL string from sql_generator.py
    CURRENT_CHART_CFG   = "current_chart_config"   # dict from chart_decider.py

    # * Data state
    CURRENT_DF          = "current_df"             # DataFrame from Power BI result
    TEMP_FILES          = "temp_files"             # list of temp file paths (for cleanup)

    # * Filter state
    FILTER_DEFINITIONS  = "filter_definitions"     # list[FilterDefinition]
    FILTER_SELECTIONS   = "filter_selections"      # dict of active filter values

    # * Chat history
    CHAT_HISTORY        = "chat_history"           # list of chat message dicts

    # * Benchmark history — stored per query for debug panel
    BENCHMARK_HISTORY   = "benchmark_history"      # list of QueryBenchmark.to_dict()

    # * UI state
    SHOW_DEBUG_PANEL    = "show_debug_panel"       # bool — show benchmark panel
    LAST_CHART_FIG      = "last_chart_fig"         # last rendered matplotlib figure
    DASHBOARD_CHARTS    = "dashboard_charts"       # list of chart configs for tab view

    # * Dictionary / ChromaDB state
    DICTIONARY_READY    = "dictionary_ready"       # bool — ChromaDB initialised


# ──────────────────────────────────────────────────────────────────────────────
# * SESSION MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    # * Manages all session state for one Streamlit session.
    # * Wraps st.session_state with typed getters/setters and defaults.
    # * Handles temp file creation, tracking, and cleanup.

    # ? One SessionManager per browser session — enforced by get_session() singleton.
    # ? Each browser tab gets its own Streamlit session and its own SessionManager.
    """

    def __init__(self):
        self._initialise_defaults()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_defaults(self):
        """
        # * Set default values for all session keys on first run.
        # * Safe to call on every rerun — only sets values that don't exist yet.
        # ? Streamlit reruns the script on every interaction.
        # ? This method is idempotent — calling it 100 times has same effect as once.
        """
        defaults = {
            Keys.SESSION_ID:          str(uuid.uuid4())[:8],
            Keys.SESSION_START_TIME:  time.time(),
            Keys.SESSION_INITIALISED: True,
            Keys.CURRENT_QUESTION:    "",
            Keys.CURRENT_INTENT:      {},
            Keys.CURRENT_SLICERS:     {},
            Keys.CURRENT_SQL:         "",
            Keys.CURRENT_CHART_CFG:   {},
            Keys.CURRENT_DF:          None,
            Keys.TEMP_FILES:          [],
            Keys.FILTER_DEFINITIONS:  [],
            Keys.FILTER_SELECTIONS:   {},
            Keys.CHAT_HISTORY:        [],
            Keys.BENCHMARK_HISTORY:   [],
            Keys.SHOW_DEBUG_PANEL:    False,
            Keys.LAST_CHART_FIG:      None,
            Keys.DASHBOARD_CHARTS:    [],
            Keys.DICTIONARY_READY:    False,
        }

        for key, default_val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_val

        log.debug(
            f"[session] Initialised | "
            f"session_id={st.session_state[Keys.SESSION_ID]}"
        )

    # ── Generic get / set ─────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        # * Get a value from session state with an optional default.

        Args:
            key     : Session state key (use Keys.* constants)
            default : Returned if key doesn't exist

        Returns:
            Current value or default.
        """
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any):
        """
        # * Set a value in session state.

        Args:
            key   : Session state key (use Keys.* constants)
            value : Value to store
        """
        st.session_state[key] = value

    def clear_key(self, key: str):
        """# * Remove a key from session state if it exists."""
        if key in st.session_state:
            del st.session_state[key]

    # ── Session identity ──────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """# * Unique 8-character ID for this browser session."""
        return st.session_state.get(Keys.SESSION_ID, "unknown")

    @property
    def session_age_minutes(self) -> float:
        """# * How many minutes this session has been active."""
        start = st.session_state.get(Keys.SESSION_START_TIME, time.time())
        return (time.time() - start) / 60

    # ── Query pipeline state ──────────────────────────────────────────────────

    def set_query(self, question: str):
        """
        # * Store a new user question and clear stale pipeline state.
        # * Called at the start of each new user query.
        # ? Clearing stale state prevents old SQL/charts from showing for new queries.

        Args:
            question : Raw user question string
        """
        log.info(f"[session] New query | session={self.session_id} | question='{question[:80]}'")
        self.set(Keys.CURRENT_QUESTION,  question)
        self.set(Keys.CURRENT_INTENT,    {})
        self.set(Keys.CURRENT_SLICERS,   {})
        self.set(Keys.CURRENT_SQL,       "")
        self.set(Keys.CURRENT_CHART_CFG, {})
        self.set(Keys.CURRENT_DF,        None)

    def set_intent(self, intent: dict):
        """# * Store the intent analysis result from analyzer.py."""
        self.set(Keys.CURRENT_INTENT, intent)
        log.debug(f"[session] Intent stored | metric={intent.get('metric', 'unknown')}")

    def set_slicers(self, slicers: dict):
        """# * Store the clarifying question answers / slicer selections."""
        self.set(Keys.CURRENT_SLICERS, slicers)
        log.debug(f"[session] Slicers stored | {list(slicers.keys())}")

    def set_sql(self, sql: str):
        """# * Store the generated SQL query string."""
        self.set(Keys.CURRENT_SQL, sql)
        log.debug(f"[session] SQL stored | length={len(sql)} chars")

    def set_chart_config(self, config: dict):
        """# * Store the chart configuration from chart_decider.py."""
        self.set(Keys.CURRENT_CHART_CFG, config)
        log.debug(f"[session] Chart config stored | type={config.get('chart_type', 'unknown')}")

    def get_current_question(self) -> str:
        return self.get(Keys.CURRENT_QUESTION, "")

    def get_current_intent(self) -> dict:
        return self.get(Keys.CURRENT_INTENT, {})

    def get_CURRENT_SQL(self) -> str:
        return self.get(Keys.CURRENT_SQL, "")

    def get_current_chart_config(self) -> dict:
        return self.get(Keys.CURRENT_CHART_CFG, {})

    # ── DataFrame / temp data ─────────────────────────────────────────────────

    def store_dataframe(self, df: pd.DataFrame, query_hash: str = "") -> Optional[Path]:
        """
        # * Save a DataFrame to a temp parquet file and register it for cleanup.
        # * Also stores the DataFrame in session state for immediate access.

        # ? Why save to disk AND keep in memory?
        # ? Streamlit session state has no size limit but large DataFrames
        # ? slow down reruns. Parquet on disk means we can reload if needed
        # ? without hitting Power BI again.

        # ! Only saves to Data/uploads/ — never anywhere else.
        # ! File is registered in TEMP_FILES list for cleanup on session end.

        Args:
            df         : DataFrame to store
            query_hash : Short hash to make filename unique per query

        Returns:
            Path to saved parquet file, or None if save failed.
        """
        if df is None or df.empty:
            log.warning("[session] store_dataframe called with empty DataFrame")
            return None

        # * Keep in memory for immediate use
        self.set(Keys.CURRENT_DF, df)

        # * Save to disk as parquet (fast, compressed, preserves dtypes)
        try:
            q_hash    = query_hash or hashlib.md5(
                self.get_current_question().encode()
            ).hexdigest()[:8]
            filename  = f"{self.session_id}_{q_hash}_{int(time.time())}.parquet"
            file_path = DATA_TEMP_DIR / filename

            df.to_parquet(file_path, index=False, compression=PARQUET_COMPRESSION)


            # * Register for cleanup
            temp_files = self.get(Keys.TEMP_FILES, [])
            temp_files.append(str(file_path))
            self.set(Keys.TEMP_FILES, temp_files)

            log.info(
                f"[session] DataFrame saved | "
                f"file={filename} | "
                f"rows={len(df)} | "
                f"cols={len(df.columns)}"
            )
            return file_path

        except Exception as e:
            log.error(f"[session] Failed to save DataFrame to disk | error={e}")
            return None

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        # * Retrieve the current DataFrame from session state.
        # * Returns None if no data has been loaded yet.
        """
        return self.get(Keys.CURRENT_DF)

    def load_dataframe_from_disk(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        # * Load a previously saved DataFrame from a parquet file.
        # ? Used to reload data without hitting Power BI again if session state was cleared.

        Args:
            file_path : Path to the parquet file

        Returns:
            DataFrame or None if file not found.
        """
        try:
            if not file_path.exists():
                log.warning(f"[session] Parquet file not found: {file_path}")
                return None
            df = pd.read_parquet(file_path)
            self.set(Keys.CURRENT_DF, df)
            log.info(f"[session] DataFrame reloaded from disk | rows={len(df)}")
            return df
        except Exception as e:
            log.error(f"[session] Failed to reload DataFrame | error={e}")
            return None

    # ── Chat history ──────────────────────────────────────────────────────────

    def add_chat_message(self, role: str, content: str, extras: dict = {}):
        """
        # * Append a message to the chat history.
        # * Automatically trims to MAX_CHAT_HISTORY to prevent memory bloat.

        Args:
            role    : "user" or "assistant"
            content : Text content of the message
            extras  : Optional extra data (e.g. chart config, SQL, benchmark timing)
        """
        history = self.get(Keys.CHAT_HISTORY, [])
        history.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            **extras,
        })

        # * Trim to max history length
        if len(history) > MAX_CHAT_HISTORY:
            history = history[-MAX_CHAT_HISTORY:]
            log.debug(f"[session] Chat history trimmed to {MAX_CHAT_HISTORY} messages")

        self.set(Keys.CHAT_HISTORY, history)

    def get_chat_history(self) -> list[dict]:
        """# * Returns the full chat history list."""
        return self.get(Keys.CHAT_HISTORY, [])

    def clear_chat_history(self):
        """# * Wipe chat history — called when user clicks Clear Chat."""
        self.set(Keys.CHAT_HISTORY, [])
        log.info(f"[session] Chat history cleared | session={self.session_id}")

    # ── Benchmark history ─────────────────────────────────────────────────────

    def record_benchmark(self, qb: QueryBenchmark):
        """
        # * Store a completed QueryBenchmark result for the debug panel.
        # * Keeps last 10 benchmarks in session state.

        Args:
            qb : Completed QueryBenchmark instance (after qb.report() called)
        """
        history = self.get(Keys.BENCHMARK_HISTORY, [])
        history.append(qb.to_dict())

        # * Keep last 10 benchmark results
        if len(history) > 10:
            history = history[-10:]

        self.set(Keys.BENCHMARK_HISTORY, history)
        log.debug(f"[session] Benchmark recorded | total_ms={qb.to_dict().get('total_ms')}ms")

    def get_benchmark_history(self) -> list[dict]:
        """# * Returns list of past benchmark result dicts."""
        return self.get(Keys.BENCHMARK_HISTORY, [])

    def get_last_benchmark(self) -> Optional[dict]:
        """# * Returns the most recent benchmark result dict."""
        history = self.get_benchmark_history()
        return history[-1] if history else None

    # ── Dashboard charts ──────────────────────────────────────────────────────

    def set_dashboard_charts(self, chart_configs: list[dict]):
        """
        # * Store the list of chart configs for the dashboard tab view.
        # * Each config is a dict produced by chart_decider.py.

        Args:
            chart_configs : List of chart config dicts
        """
        self.set(Keys.DASHBOARD_CHARTS, chart_configs)
        log.debug(f"[session] Dashboard charts stored | count={len(chart_configs)}")

    def get_dashboard_charts(self) -> list[dict]:
        """# * Returns the current dashboard chart configs."""
        return self.get(Keys.DASHBOARD_CHARTS, [])

    # ── Debug panel ───────────────────────────────────────────────────────────

    def toggle_debug_panel(self):
        """# * Toggle the benchmark debug panel visibility."""
        current = self.get(Keys.SHOW_DEBUG_PANEL, False)
        self.set(Keys.SHOW_DEBUG_PANEL, not current)

    def is_debug_panel_visible(self) -> bool:
        """# * Returns True if the debug panel should be shown."""
        return self.get(Keys.SHOW_DEBUG_PANEL, False)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_temp_files(self, force: bool = False):
        """
        # * Delete temp parquet files created during this session.
        # * Called on session end OR when TTL has expired.

        # ! ONLY deletes files from Data/uploads/ — never logs, code, or context JSONs.
        # ! Uses an allowlist check on each path before deleting.

        Args:
            force : If True, delete all temp files regardless of TTL.
                    If False, only delete files older than TEMP_DATA_TTL_MINUTES.
        """
        temp_files = self.get(Keys.TEMP_FILES, [])

        if not temp_files:
            log.debug("[session] No temp files to clean up")
            return

        deleted  = 0
        skipped  = 0
        failed   = 0
        now      = time.time()
        ttl_secs = TEMP_DATA_TTL_MINUTES * 60

        for file_str in temp_files:
            file_path = Path(file_str)

            # ! Safety check — only delete files inside DATA_TEMP_DIR
            # ! This prevents accidental deletion of anything outside uploads/
            try:
                file_path.resolve().relative_to(DATA_TEMP_DIR.resolve())
            except ValueError:
                log.error(
                    f"[session] SAFETY BLOCK: Refused to delete file outside "
                    f"Data/uploads/: {file_path}"
                )
                skipped += 1
                continue

            if not file_path.exists():
                skipped += 1
                continue

            # * Check TTL if not forcing
            if not force:
                file_age_secs = now - file_path.stat().st_mtime
                if file_age_secs < ttl_secs:
                    log.debug(
                        f"[session] Skipping (TTL not expired) | "
                        f"file={file_path.name} | "
                        f"age={file_age_secs:.0f}s / {ttl_secs}s"
                    )
                    skipped += 1
                    continue

            try:
                file_path.unlink()
                deleted += 1
                log.debug(f"[session] Deleted temp file | {file_path.name}")
            except Exception as e:
                failed += 1
                log.error(f"[session] Failed to delete temp file | {file_path} | error={e}")

        # * Update the temp files list — remove deleted ones
        remaining = [
            f for f in temp_files
            if Path(f).exists()
        ]
        self.set(Keys.TEMP_FILES, remaining)

        log.info(
            f"[session] Cleanup complete | "
            f"deleted={deleted} | skipped={skipped} | failed={failed} | "
            f"remaining={len(remaining)}"
        )

    def cleanup_all_stale_files(self):
        """
        # * Scan the entire Data/uploads/ folder and delete any parquet files
        # * older than TEMP_DATA_TTL_MINUTES — even from other sessions.
        # * Called at app startup to clear files from crashed sessions.

        # ! Only targets .parquet files in Data/uploads/ — nothing else.
        """
        log.info("[session] Scanning for stale temp files across all sessions...")

        if not DATA_TEMP_DIR.exists():
            return

        ttl_secs = TEMP_DATA_TTL_MINUTES * 60
        now      = time.time()
        deleted  = 0

        for file_path in DATA_TEMP_DIR.glob("*.parquet"):
            try:
                file_age = now - file_path.stat().st_mtime
                if file_age > ttl_secs:
                    file_path.unlink()
                    deleted += 1
                    log.debug(f"[session] Stale file deleted | {file_path.name} | age={file_age:.0f}s")
            except Exception as e:
                log.error(f"[session] Failed to delete stale file | {file_path} | error={e}")

        log.info(f"[session] Stale file cleanup complete | deleted={deleted} files")

    # ── Full reset ────────────────────────────────────────────────────────────

    def reset_query_state(self):
        """
        # * Reset only the current query pipeline state.
        # * Preserves chat history, benchmark history, and session identity.
        # ? Use this when user starts a new question.
        """
        self.set(Keys.CURRENT_QUESTION,  "")
        self.set(Keys.CURRENT_INTENT,    {})
        self.set(Keys.CURRENT_SLICERS,   {})
        self.set(Keys.CURRENT_SQL,       "")
        self.set(Keys.CURRENT_CHART_CFG, {})
        self.set(Keys.CURRENT_DF,        None)
        self.set(Keys.FILTER_DEFINITIONS,[])
        self.set(Keys.FILTER_SELECTIONS, {})
        self.set(Keys.DASHBOARD_CHARTS,  [])
        log.debug(f"[session] Query state reset | session={self.session_id}")

    def full_reset(self):
        """
        # * Full session reset — clears everything including chat history.
        # * Deletes all temp files.
        # ? Use this for a "Start Over" button in the UI.
        """
        log.info(f"[session] Full reset | session={self.session_id}")
        self.cleanup_temp_files(force=True)

        keys_to_clear = [
            Keys.CURRENT_QUESTION,
            Keys.CURRENT_INTENT,
            Keys.CURRENT_SLICERS,
            Keys.CURRENT_SQL,
            Keys.CURRENT_CHART_CFG,
            Keys.CURRENT_DF,
            Keys.FILTER_DEFINITIONS,
            Keys.FILTER_SELECTIONS,
            Keys.CHAT_HISTORY,
            Keys.BENCHMARK_HISTORY,
            Keys.DASHBOARD_CHARTS,
            Keys.LAST_CHART_FIG,
        ]
        for key in keys_to_clear:
            self.clear_key(key)

        # * Re-initialise defaults after clearing
        self._initialise_defaults()


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

def get_session() -> SessionManager:
    """
    # * Returns the SessionManager for the current Streamlit session.
    # * Creates a new one on first call per session.
    # * Safe to call anywhere — returns same instance on every rerun.

    # ? Why store SessionManager in st.session_state?
    # ? Streamlit reruns the script from top to bottom on every interaction.
    # ? Storing the manager instance in session_state ensures the same
    # ? object is returned on every rerun within the same browser session.

    Usage in any file:
        from core.session import get_session
        session = get_session()
        session.set_query(user_question)
        df = session.get_dataframe()
    """
    _MANAGER_KEY = "_session_manager_instance"

    if _MANAGER_KEY not in st.session_state:
        st.session_state[_MANAGER_KEY] = SessionManager()
        log.info(
            f"[session] New SessionManager created | "
            f"session_id={st.session_state[_MANAGER_KEY].session_id}"
        )

    return st.session_state[_MANAGER_KEY]