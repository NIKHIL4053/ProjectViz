"""
utils/helpers.py
----------------
# * Shared utility functions used across the entire Loan Dashboard project.
# * No business logic lives here — pure helper functions only.
# * No dependencies on other project files except logger.

# ? Why centralize helpers here?
# ? Prevents copy-paste of the same small functions across files.
# ? Single place to fix a bug if a utility function breaks.
# ? Every file in the project can import from here safely.

Exports:
    - parse_json_safely()       : Extract JSON from model output (handles markdown fences)
    - get_column()              : Case-insensitive column lookup in a DataFrame
    - format_number()           : Format large numbers for KPI display (1.2M, 45.3K)
    - format_currency()         : Format INR amounts with commas and symbol
    - format_percent()          : Format floats as percentage strings
    - truncate_string()         : Truncate long strings with ellipsis
    - is_numeric_column()       : Check if a DataFrame column is numeric
    - is_date_column()          : Check if a DataFrame column is date-like
    - get_numeric_columns()     : Return all numeric column names from a DataFrame
    - get_categorical_columns() : Return all categorical column names
    - get_date_columns()        : Return all date column names
    - safe_divide()             : Division that returns 0 instead of ZeroDivisionError
    - flatten_dict()            : Flatten nested dict to single level
    - chunk_list()              : Split a list into chunks of size n
    - clean_column_name()       : Normalize column names (strip, title case)
    - build_filter_summary()    : Build human-readable string of applied filters
    - estimate_tokens()         : Rough token count estimate for prompt sizing
"""

import json
import re
from typing import Any, Optional

import pandas as pd

from utils.logger import get_logger

# * Module level logger
log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * JSON PARSING
# * Models sometimes wrap JSON in markdown fences — these handle that safely.
# ──────────────────────────────────────────────────────────────────────────────

def parse_json_safely(raw: str) -> Optional[dict | list]:
    """
    # * Safely parse JSON from model output.
    # * Handles cases where Qwen wraps JSON in markdown code fences.

    # ? Why needed?
    # ? Qwen sometimes returns ```json { ... } ``` instead of bare JSON.
    # ? json.loads() fails on fenced output — this strips the fences first.

    Args:
        raw : Raw string output from Qwen model

    Returns:
        Parsed dict or list, or None if parsing fails.

    Example:
        raw = "```json\n{\"chart_type\": \"line\"}\n```"
        result = parse_json_safely(raw)
        # → {"chart_type": "line"}
    """
    if not raw or not raw.strip():
        log.warning("parse_json_safely received empty string")
        return None

    clean = raw.strip()

    # * Strip markdown code fences if present
    if "```" in clean:
        parts = clean.split("```")
        # ! parts[1] is the content between first pair of fences
        if len(parts) >= 2:
            clean = parts[1]
            # * Remove language tag (json, python, etc.) from first line
            if clean.startswith(("json", "JSON", "python")):
                clean = clean[clean.index("\n") + 1:] if "\n" in clean else clean[4:]

    clean = clean.strip().rstrip("`").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        log.error(f"parse_json_safely failed | error={e} | raw_preview={raw[:120]}")
        return None


def extract_json_from_text(text: str) -> Optional[dict | list]:
    """
    # * More aggressive JSON extraction — finds first {...} or [...] block in text.
    # * Use this when parse_json_safely fails (model added extra explanation text).

    Args:
        text : Any string that may contain a JSON block somewhere inside it

    Returns:
        First valid JSON object or array found, or None.
    """
    # * Try to find JSON object or array pattern
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # nested object
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # nested array
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    log.warning(f"extract_json_from_text found no valid JSON | preview={text[:100]}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# * DATAFRAME COLUMN UTILITIES
# * Safe, case-insensitive helpers for working with dynamic column names.
# ──────────────────────────────────────────────────────────────────────────────

def get_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """
    # * Case-insensitive column lookup in a DataFrame.
    # * Returns the actual column name as it exists in the DataFrame.

    # ? Why needed?
    # ? Qwen may return column names in different casing than the actual data.
    # ? e.g. Qwen returns "branch" but column is "Branch" — this handles it.

    Args:
        df   : The DataFrame to search
        name : Column name to look up (any casing)

    Returns:
        Actual column name from df.columns, or None if not found.

    Example:
        get_column(df, "bounce status")  # → "Bounce status"
        get_column(df, "REGION")         # → "Region"
        get_column(df, "nonexistent")    # → None
    """
    if not name or str(name).lower() in ["null", "none", ""]:
        return None

    name_lower = str(name).strip().lower()

    for col in df.columns:
        if str(col).strip().lower() == name_lower:
            return col

    log.debug(f"get_column: '{name}' not found in {list(df.columns)}")
    return None


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    # * Return all numeric column names from a DataFrame.
    # * Excludes columns that look like IDs (all unique integer values).

    Args:
        df : Source DataFrame

    Returns:
        List of column names with numeric dtype, excluding likely ID columns.
    """
    numeric = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # ! Skip columns that are likely IDs (all unique, integer type)
            if df[col].nunique() == len(df) and df[col].dtype in ["int64", "int32"]:
                continue
            numeric.append(col)
    return numeric


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """
    # * Return all categorical column names from a DataFrame.
    # * A column is categorical if it's object dtype OR numeric with few unique values.

    Args:
        df         : Source DataFrame
        max_unique : Max unique values to still be considered categorical (default 50)

    Returns:
        List of column names suitable for use as categories/slicers.
    """
    categorical = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= max_unique:
            categorical.append(col)
    return categorical


def get_date_columns(df: pd.DataFrame) -> list[str]:
    """
    # * Return all date/datetime column names from a DataFrame.
    # * Also attempts to detect string columns that parse as dates.

    Args:
        df : Source DataFrame

    Returns:
        List of column names that are date-like.
    """
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        # * Try to detect date strings (e.g. "01-02-2026", "Feb 2026")
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().head(5)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                date_cols.append(col)
            except Exception:
                pass
    return date_cols


def is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """# * Returns True if the column exists and is numeric dtype."""
    if col not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[col])


def is_date_column(df: pd.DataFrame, col: str) -> bool:
    """# * Returns True if the column exists and is datetime dtype."""
    if col not in df.columns:
        return False
    return pd.api.types.is_datetime64_any_dtype(df[col])


def clean_column_name(name: str) -> str:
    """
    # * Normalize a column name — strip whitespace, remove special chars.
    # * Used when loading Excel files with inconsistently named columns.

    Args:
        name : Raw column name from Excel

    Returns:
        Cleaned column name.

    Example:
        clean_column_name("  Bounce Status  ")  # → "Bounce Status"
        clean_column_name("Op_bucket")          # → "Op bucket"
    """
    cleaned = str(name).strip()
    cleaned = re.sub(r'_+', ' ', cleaned)       # * Replace underscores with spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)      # * Collapse multiple spaces
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# * NUMBER FORMATTING
# * For KPI cards and chart annotations — readable display of large numbers.
# ──────────────────────────────────────────────────────────────────────────────

def format_number(value: float | int, decimals: int = 1) -> str:
    """
    # * Format large numbers into human-readable shorthand.

    Args:
        value    : Numeric value to format
        decimals : Decimal places in output (default 1)

    Returns:
        Formatted string.

    Example:
        format_number(1_200_000)  # → "1.2M"
        format_number(45_300)     # → "45.3K"
        format_number(850)        # → "850"
    """
    try:
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.{decimals}f}B"
        elif abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.{decimals}f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.{decimals}f}K"
        else:
            return f"{value:.0f}"
    except (TypeError, ValueError):
        return str(value)


def format_currency(value: float | int, symbol: str = "₹") -> str:
    """
    # * Format a number as Indian Rupee currency with commas.

    Args:
        value  : Numeric value (in rupees)
        symbol : Currency symbol (default ₹)

    Returns:
        Formatted currency string.

    Example:
        format_currency(1_234_567)  # → "₹12,34,567"
        format_currency(89_255.50)  # → "₹89,255.50"
    """
    try:
        value = float(value)
        # * Use Indian number format (lakhs/crores)
        if abs(value) >= 10_000_000:
            return f"{symbol}{value / 10_000_000:.2f} Cr"
        elif abs(value) >= 100_000:
            return f"{symbol}{value / 100_000:.2f} L"
        else:
            return f"{symbol}{value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: float, decimals: int = 1) -> str:
    """
    # * Format a float as a percentage string.

    Args:
        value    : Float value (e.g. 0.253 or 25.3 — both handled)
        decimals : Decimal places (default 1)

    Returns:
        Percentage string.

    Example:
        format_percent(25.3)   # → "25.3%"
        format_percent(0.253)  # → "25.3%"  (auto-detected as fraction)
    """
    try:
        v = float(value)
        # * Auto-detect if value is already a percentage or a fraction
        if abs(v) <= 1.0:
            v = v * 100
        return f"{v:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


# ──────────────────────────────────────────────────────────────────────────────
# * STRING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def truncate_string(text: str, max_len: int = 80, suffix: str = "...") -> str:
    """
    # * Truncate a string to max_len characters with a suffix.

    Args:
        text    : Input string
        max_len : Maximum length including suffix
        suffix  : String to append when truncated (default "...")

    Returns:
        Truncated string if longer than max_len, original string otherwise.
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def build_filter_summary(filters: dict) -> str:
    """
    # * Build a human-readable one-line summary of applied filters.
    # * Used in chart titles and export headers.

    Args:
        filters : Dict of {field_name: selected_value}
                  Values of "All" or None are skipped.

    Returns:
        Formatted filter string for display.

    Example:
        filters = {"Region": "Pune", "Op bucket": "All", "Portfolio new": "VASTU"}
        build_filter_summary(filters)
        # → "Region: Pune | Portfolio new: VASTU"
    """
    active = {
        k: v for k, v in filters.items()
        if v and str(v).lower() not in ["all", "none", ""]
    }
    if not active:
        return "All Data"
    return "  |  ".join(f"{k}: {v}" for k, v in active.items())


def estimate_tokens(text: str) -> int:
    """
    # * Rough token count estimate for prompt sizing.
    # * Rule of thumb: 1 token ≈ 4 characters for English text.
    # ! This is an approximation — not the same as tiktoken or model tokenizer.
    # ? Used to warn if a prompt is getting too large for the model context window.

    Args:
        text : Any string (system prompt, user message, context JSON)

    Returns:
        Estimated token count as integer.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


# ──────────────────────────────────────────────────────────────────────────────
# * MATH UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    # * Safe division that returns a default value instead of raising ZeroDivisionError.
    # * Use for any percentage or ratio calculation in the pipeline.

    Args:
        numerator   : Top number
        denominator : Bottom number
        default     : Value to return if denominator is 0 (default 0.0)

    Returns:
        numerator / denominator, or default if denominator is 0.

    Example:
        safe_divide(500, 2000)   # → 0.25
        safe_divide(100, 0)      # → 0.0  (no error)
        safe_divide(100, 0, -1)  # → -1   (custom default)
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# * COLLECTION UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def chunk_list(items: list, size: int) -> list[list]:
    """
    # * Split a list into chunks of a given size.
    # * Used when batching ChromaDB inserts for large data dictionaries.

    Args:
        items : Any list
        size  : Max items per chunk

    Returns:
        List of sublists.

    Example:
        chunk_list([1,2,3,4,5], 2)  # → [[1,2], [3,4], [5]]
    """
    if size <= 0:
        # ! Prevent infinite loop if size is 0 or negative
        log.error(f"chunk_list called with invalid size={size}")
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    # * Flatten a nested dictionary to a single level.
    # * Used when converting context JSONs to a flat format for prompt injection.

    Args:
        d          : Dictionary to flatten (can be nested)
        parent_key : Prefix for keys (used in recursion)
        sep        : Separator between key levels (default ".")

    Returns:
        Flat dictionary with dotted keys.

    Example:
        flatten_dict({"a": {"b": 1, "c": 2}})
        # → {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)