"""
ui/sidebar.py
-------------
# * Post-result slicer bar shown at the top of the dashboard.
# * Dynamically builds filter dropdowns from the current DataFrame.
# * Applies selected filters and returns the filtered DataFrame.
"""

import pandas as pd
import streamlit as st

from core.filters import FilterManager, FilterSelection
from utils.logger import get_logger

log = get_logger(__name__)


def render_slicer_bar(
    df:          pd.DataFrame,
    max_slicers: int = 4,
    key_prefix:  str = "slicer_bar",
) -> tuple[pd.DataFrame, str]:
    """
    # * Render a horizontal slicer bar at the top of the dashboard.
    # * Auto-detects relevant filter columns from the DataFrame.
    # * Returns the filtered DataFrame and a filter summary string.

    Args:
        df          : Current result DataFrame
        max_slicers : Max number of slicer dropdowns to show (default 4)
        key_prefix  : Streamlit widget key prefix (avoid duplicate keys)

    Returns:
        (filtered_df, filter_summary_str)
    """
    if df is None or df.empty:
        return df, "All Data"

    fm = FilterManager(max_slicers=max_slicers)
    definitions = fm.detect_filters(df)

    if not definitions:
        return df, "All Data"

    st.markdown("##### 🔧 Refine Results")

    # * Render slicers in a horizontal row
    n_cols = min(len(definitions), max_slicers)
    cols   = st.columns(n_cols)

    active_sels = []

    for i, defn in enumerate(definitions[:n_cols]):
        with cols[i]:
            widget_key = f"{key_prefix}_{defn.field.lower().replace(' ', '_').replace('/', '_')}"
            options    = ["All"] + [str(v) for v in defn.options]

            if defn.widget_type == "multiselect":
                selected = st.multiselect(
                    label   = defn.label,
                    options = [str(v) for v in defn.options],
                    default = [],
                    key     = widget_key,
                    help    = f"Leave empty for all {defn.field} values",
                )
                val = selected if selected else "All"
            else:
                val = st.selectbox(
                    label   = defn.label,
                    options = options,
                    index   = 0,
                    key     = widget_key,
                )

            if val and str(val).lower() not in ("all", ""):
                sel_val = val if isinstance(val, list) else val
                active_sels.append(
                    FilterSelection(
                        field       = defn.field,
                        value       = sel_val,
                        widget_type = defn.widget_type,
                    )
                )

    # * Apply selections
    if active_sels:
        filtered_df    = fm.apply_filters(df, active_sels)
        filter_summary = fm.get_summary(active_sels)
        rows_removed   = len(df) - len(filtered_df)
        if rows_removed > 0:
            st.caption(
                f"Showing {len(filtered_df):,} of {len(df):,} rows "
                f"({rows_removed:,} filtered out)"
            )
        return filtered_df, filter_summary
    else:
        return df, "All Data"