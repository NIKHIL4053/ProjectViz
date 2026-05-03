"""
ui/kpis.py
----------
# * Smart KPI card detection and rendering.
"""

import pandas as pd
import streamlit as st

from utils.helpers import format_currency, format_number, format_percent
from utils.logger import get_logger

log = get_logger(__name__)

_KPI_PATTERNS = [
    ("bounce %", "Bounce Rate", "pct"),
    ("resolution %", "Resolution", "pct"),
    ("coverage %", "Coverage", "pct"),
    ("intensity", "Intensity", "num"),
    ("bounce count", "Bounces", "num"),
    ("resolved count", "Resolved", "num"),
    ("visited count", "Visited", "num"),
    ("allocated count", "Allocated", "num"),
    ("customer count", "Customers", "num"),
    ("total customers", "Customers", "num"),
    ("loan count", "Loans", "num"),
    ("npa count", "NPA", "num"),
    ("npa customer", "NPA Customers", "num"),
    ("new npa count", "New NPA", "num"),
    ("balance principal", "Portfolio", "cur"),
    ("total overdue", "Overdue", "cur"),
    ("outstanding", "Outstanding", "cur"),
    ("avg visits", "Avg Visits", "num"),
    ("count", "Count", "num"),
]


def render_kpis(df: pd.DataFrame, max_kpis: int = 4):
    """
    # * Detect and render KPI metric cards from a DataFrame.
    """
    if df is None or df.empty:
        return

    kpis = []
    cols_lower = {c.lower(): c for c in df.columns}
    used_cols = set()

    for pattern, label, fmt in _KPI_PATTERNS:
        if len(kpis) >= max_kpis:
            break
        for col_lower, col_actual in cols_lower.items():
            if col_actual in used_cols:
                continue
            if pattern in col_lower:
                try:
                    value = df[col_actual].iloc[0] if len(df) == 1 else df[col_actual].sum()
                    if pd.isna(value):
                        continue
                    value = float(value)
                    display = (
                        format_percent(value)
                        if fmt == "pct"
                        else format_currency(value)
                        if fmt == "cur"
                        else format_number(value)
                    )
                    kpis.append((label, display))
                    used_cols.add(col_actual)
                    break
                except Exception:
                    continue

    if not kpis:
        log.debug(f"[kpis] No KPI patterns matched | columns={list(df.columns)}")
        return

    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        col.metric(label=label, value=value)

    log.debug(f"[kpis] Rendered {len(kpis)} KPI cards | {[k[0] for k in kpis]}")
