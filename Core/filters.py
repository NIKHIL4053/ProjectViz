"""
core/filters.py
---------------
# * Manages all filter/slicer logic for the Loan Dashboard.
# * Builds sidebar filter widgets dynamically from Power BI result data.
# * Applies user slicer selections to DataFrames returned from Power BI.
# * Provides filter summary strings for chart titles and export headers.

# ? Why dynamic filters instead of hardcoded ones?
# ? Power BI returns different columns depending on the DAX query.
# ? Hardcoding filters would break when the query changes.
# ? Dynamic filters read the actual columns + unique values from returned data.

# ? Filter priority — which slicers to show first:
# ?   1. Geography    → Region, Branch
# ?   2. Bucket       → Op bucket, Cust wise status
# ?   3. Portfolio    → Portfolio new, Product
# ?   4. Collection   → TL, SH, Coll/Sales
# ?   5. Payment      → Bounce status, Digital/cash, Payment mode
# ?   6. Risk         → Add NPA, NPA MOB

Exports:
    - FilterManager         : Main class — build, apply, summarise filters
    - FilterDefinition      : Dataclass for one filter's config
    - FilterSelection       : Dataclass for one filter's current selection
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import streamlit as st

from config import MAX_SIDEBAR_SLICERS
from utils.logger import get_logger
from utils.benchmark import benchmark
from utils.helpers import build_filter_summary, get_categorical_columns, get_date_columns

# * Module level logger
log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * CONSTANTS
# * Slicer columns in priority order — shown first if present in data.
# * All other slicer columns follow after these.
# ──────────────────────────────────────────────────────────────────────────────

# * Priority slicer order — matches business hierarchy from data dictionary
PRIORITY_SLICERS = [
    "Region",               # Geography
    "Branch",               # Geography
    "Op bucket",            # Bucket / Status
    "Cust wise status",     # Bucket / Status
    "Portfolio new",        # Portfolio
    "Product",              # Portfolio
    "TL",                   # Collection Team
    "SH",                   # Collection Team
    "Coll/Sales",           # Collection Team
    "Bounce status",        # Payment
    "Digital/cash",         # Payment
    "Payment mode",         # Payment
    "Add NPA",              # Risk
    "MOB Bucket",           # Loan
    "Visited",              # Collection
    "Allocated or not",     # Collection
    "Visit or not",         # Collection
    "Type of Arrangement",  # Portfolio
    "Transaction Type",     # Product
    "Scheme",               # Product
    "Payment Day bucket",   # Payment
    "Loan level bounce",    # Payment
    "Cust level bounce",    # Payment
]

# * Fields that should NEVER be shown as slicers even if categorical
# ! These are identifiers or derived fields — not useful as filters
EXCLUDED_FROM_SLICERS = {
    "loanappno",
    "Cust ID",
    "Allocation 1",         # too many unique values (FE names)
    "NPA_Origination_Date", # date — handled separately if needed
    "Due Date",             # date — too granular for slicer
    "Next Date",            # date — too granular for slicer
    "Closing bucket",       # derived — use Op bucket instead
    "Installment No",       # numeric — not useful as dropdown
}

# * Fields with known fixed value sets — used to build options without scanning data
# ? These override dynamic value detection for cleaner UI
KNOWN_VALUES: dict[str, list] = {
    "Op bucket":        ["Current", "Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD", "NPA", "Write-off"],
    "Cust wise status": ["Current", "Norm", "Flow", "Stab", "Roll Back", "Risk NPA", "NPA", "Write-off"],
    "Bounce status":    ["Tech", "Non Tech", "PAID"],
    "Digital/cash":     ["Digital", "Cash"],
    "Allocated or not": ["Allocated", "Non Allocated"],
    "Visit or not":     ["Visited", "Not Visited"],
    "Coll/Sales":       ["Collection", "Sales"],
    "Add NPA":          [0, 1],
    "Loan level bounce":[0, 1],
    "Cust level bounce":[0, 1],
    "MOB Bucket":       ["1-6 MOB", "7-12 MOB", "13-18 MOB", "19-24 MOB", "24+ MOB"],
}


# ──────────────────────────────────────────────────────────────────────────────
# * DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterDefinition:
    """
    # * Defines the configuration for one filter/slicer widget.
    # * Built from the DataFrame columns before rendering in sidebar.

    Attributes:
        field       : Column name in the DataFrame (e.g. "Bounce status")
        label       : Human-readable label for the UI (e.g. "Bounce Status")
        widget_type : "dropdown" | "multiselect" | "slider" | "date_range"
        options     : Available values for dropdown/multiselect
        default     : Default selected value
        category    : Filter category for grouping in sidebar
    """
    field:       str
    label:       str
    widget_type: str                  = "dropdown"
    options:     list                 = field(default_factory=list)
    default:     Any                  = "All"
    category:    str                  = "Other"


@dataclass
class FilterSelection:
    """
    # * Stores the user's current selection for one filter.
    # * Created after user interacts with sidebar widgets.

    Attributes:
        field      : Column name in the DataFrame
        value      : Selected value ("All" means no filter applied)
        widget_type: Matches FilterDefinition.widget_type
    """
    field:       str
    value:       Any
    widget_type: str = "dropdown"

    @property
    def is_active(self) -> bool:
        """# * True if this filter has a real selection (not All / None / empty)."""
        if self.value is None:
            return False
        if isinstance(self.value, str) and self.value.lower() in ["all", ""]:
            return False
        if isinstance(self.value, list) and len(self.value) == 0:
            return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# * FILTER MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class FilterManager:
    """
    # * Manages the full lifecycle of dashboard filters.
    # * Step 1 — detect_filters()   : Scan DataFrame, decide which filters to show
    # * Step 2 — render_sidebar()   : Render Streamlit widgets in sidebar
    # * Step 3 — apply_filters()    : Apply user selections to DataFrame
    # * Step 4 — get_summary()      : Build filter summary string for titles

    # ? Why keep filter state in this class and not Streamlit session state?
    # ? FilterManager is instantiated fresh per query — it reads current
    # ? widget values from Streamlit which persist in session state naturally.
    # ? This keeps filter logic separate from Streamlit's state management.
    """

    def __init__(self, max_slicers: int = MAX_SIDEBAR_SLICERS):
        self.max_slicers:    int                      = max_slicers
        self._definitions:   list[FilterDefinition]   = []
        self._selections:    list[FilterSelection]    = []

    # ── Step 1: Detect filters ────────────────────────────────────────────────

    def detect_filters(
        self,
        df:              pd.DataFrame,
        forced_columns:  Optional[list[str]] = None,
    ) -> list[FilterDefinition]:
        """
        # * Scan the DataFrame and decide which filter widgets to show.
        # * Respects priority order, exclusion list, and max_slicers limit.

        Args:
            df              : DataFrame returned from Power BI DAX query
            forced_columns  : Optional list of columns to always include as filters
                              (e.g. from Qwen's clarifying question output)

        Returns:
            List of FilterDefinition objects in priority order.
        """
        with benchmark("filter_detection"):
            if df is None or df.empty:
                log.warning("[filters] detect_filters called with empty DataFrame")
                return []

            available_cols = set(df.columns.tolist())
            definitions    = []

            # * Start with forced columns (from Qwen's clarifying output)
            forced = forced_columns or []

            # * Build ordered list: forced first, then priority order, then remaining
            ordered = []
            for col in forced:
                if col not in ordered:
                    ordered.append(col)
            for col in PRIORITY_SLICERS:
                if col not in ordered:
                    ordered.append(col)

            # * Add any remaining categorical columns not already in ordered list
            remaining_cats = get_categorical_columns(df, max_unique=50)
            for col in remaining_cats:
                if col not in ordered:
                    ordered.append(col)

            # * Build FilterDefinition for each valid column
            for col in ordered:
                if len(definitions) >= self.max_slicers:
                    break

                # * Skip if not in this DataFrame
                if col not in available_cols:
                    continue

                # * Skip excluded fields
                if col in EXCLUDED_FROM_SLICERS:
                    continue

                # * Skip columns with too many unique values (not useful as filters)
                n_unique = df[col].nunique()
                if n_unique > 60:
                    log.debug(f"[filters] Skipping '{col}' — {n_unique} unique values (too many)")
                    continue

                # * Skip columns with only one unique value (no filtering value)
                if n_unique <= 1:
                    log.debug(f"[filters] Skipping '{col}' — only {n_unique} unique value")
                    continue

                defn = self._build_definition(df, col)
                if defn:
                    definitions.append(defn)

            self._definitions = definitions
            log.info(f"[filters] Detected {len(definitions)} filters from {len(available_cols)} columns")
            return definitions

    def _build_definition(
        self,
        df:  pd.DataFrame,
        col: str,
    ) -> Optional[FilterDefinition]:
        """
        # * Build a FilterDefinition for a single column.
        # * Uses KNOWN_VALUES if available, otherwise scans DataFrame.

        Args:
            df  : Source DataFrame
            col : Column name

        Returns:
            FilterDefinition or None if column is not suitable.
        """
        try:
            # * Determine widget type
            widget_type = self._get_widget_type(df, col)

            # * Get options
            if col in KNOWN_VALUES:
                # * Use predefined values — cleaner order than scanning data
                options = KNOWN_VALUES[col]
            else:
                # * Scan data for unique values — sort for cleaner UX
                raw_vals = df[col].dropna().unique().tolist()
                options  = sorted([str(v) for v in raw_vals])

            # * Human readable label — title case, handle slash fields
            label = col.replace("/", " / ").title()

            # * Category for grouping
            category = self._get_category(col)

            return FilterDefinition(
                field=col,
                label=label,
                widget_type=widget_type,
                options=options,
                default="All",
                category=category,
            )

        except Exception as e:
            log.error(f"[filters] _build_definition failed for '{col}' | error={e}")
            return None

    def _get_widget_type(self, df: pd.DataFrame, col: str) -> str:
        """
        # * Decide which Streamlit widget type to use for a column.

        Rules:
            - Binary (0/1) columns             → dropdown (All / 0 / 1)
            - Columns with ≤ 7 unique values    → multiselect (allows multiple selection)
            - Columns with 8-60 unique values   → dropdown (single select)
        """
        if col in KNOWN_VALUES and len(KNOWN_VALUES[col]) <= 2:
            return "dropdown"

        n_unique = df[col].nunique()
        if n_unique <= 7:
            return "multiselect"
        return "dropdown"

    def _get_category(self, col: str) -> str:
        """# * Map column name to a sidebar category group label."""
        geography  = {"Region", "Branch"}
        bucket     = {"Op bucket", "Cust wise status", "Closing bucket"}
        portfolio  = {"Portfolio new", "Type of Arrangement", "Transaction Type", "Scheme", "Product"}
        team       = {"TL", "SH", "Allocation 1", "Allocated or not", "Coll/Sales"}
        payment    = {"Bounce status", "Digital/cash", "Payment mode",
                      "Payment Day bucket", "Loan level bounce", "Cust level bounce"}
        risk       = {"Add NPA", "NPA MOB", "Risk NPA"}
        collection = {"Visited", "Visit or not", "Visit Count"}
        loan       = {"MOB Bucket", "Installment No", "Emi increase"}

        if col in geography:  return "📍 Geography"
        if col in bucket:     return "📊 Bucket / Status"
        if col in portfolio:  return "🗂️ Portfolio"
        if col in team:       return "👥 Collection Team"
        if col in payment:    return "💳 Payment"
        if col in risk:       return "⚠️ Risk"
        if col in collection: return "🏃 Collection Activity"
        if col in loan:       return "📋 Loan"
        return "🔧 Other"

    # ── Step 2: Render sidebar ────────────────────────────────────────────────

    def render_sidebar(self) -> list[FilterSelection]:
        """
        # * Render all filter widgets in the Streamlit sidebar.
        # * Groups filters by category with expanders.
        # * Returns the user's current selections.

        # ! Must be called inside a Streamlit `with st.sidebar:` block
        # ! or it will render in the main area.

        Returns:
            List of FilterSelection — one per rendered filter.
        """
        if not self._definitions:
            st.sidebar.info("No filters available for this dataset.")
            return []

        selections = []

        # * Group definitions by category
        categories: dict[str, list[FilterDefinition]] = {}
        for defn in self._definitions:
            cat = defn.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(defn)

        # * Render each category group
        for cat_label, defns in categories.items():
            with st.sidebar.expander(cat_label, expanded=True):
                for defn in defns:
                    sel = self._render_widget(defn)
                    if sel:
                        selections.append(sel)

        self._selections = selections
        active_count = sum(1 for s in selections if s.is_active)
        log.debug(f"[filters] Rendered {len(selections)} filters | {active_count} active")
        return selections

    def _render_widget(self, defn: FilterDefinition) -> Optional[FilterSelection]:
        """
        # * Render a single filter widget and return the user's selection.

        Args:
            defn : FilterDefinition for this filter

        Returns:
            FilterSelection with current value, or None on error.
        """
        try:
            # * Use a stable Streamlit key so widget state persists on rerun
            widget_key = f"filter_{defn.field.lower().replace(' ', '_').replace('/', '_')}"

            if defn.widget_type == "multiselect":
                selected = st.multiselect(
                    label   = defn.label,
                    options = [str(v) for v in defn.options],
                    default = [],
                    key     = widget_key,
                )
                # * Empty multiselect = "All"
                value = selected if selected else "All"

            else:
                # * Dropdown — single select with "All" as first option
                options_with_all = ["All"] + [str(v) for v in defn.options]
                selected = st.selectbox(
                    label   = defn.label,
                    options = options_with_all,
                    index   = 0,
                    key     = widget_key,
                )
                value = selected

            return FilterSelection(
                field       = defn.field,
                value       = value,
                widget_type = defn.widget_type,
            )

        except Exception as e:
            log.error(f"[filters] _render_widget failed for '{defn.field}' | error={e}")
            return None

    # ── Step 3: Apply filters ─────────────────────────────────────────────────

    def apply_filters(
        self,
        df:         pd.DataFrame,
        selections: Optional[list[FilterSelection]] = None,
    ) -> pd.DataFrame:
        """
        # * Apply user slicer selections to a DataFrame.
        # * Only applies active filters — "All" selections are skipped.
        # * Returns a filtered copy — never modifies the original DataFrame.

        Args:
            df         : Source DataFrame (from Power BI DAX result)
            selections : List of FilterSelection — uses self._selections if None

        Returns:
            Filtered DataFrame.
        """
        if df is None or df.empty:
            return df

        sels = selections or self._selections
        if not sels:
            return df

        with benchmark("filter_apply"):
            filtered = df.copy()
            original_len = len(filtered)

            for sel in sels:
                if not sel.is_active:
                    continue

                col = sel.field
                if col not in filtered.columns:
                    log.warning(f"[filters] Filter column '{col}' not in DataFrame — skipping")
                    continue

                try:
                    if sel.widget_type == "multiselect" and isinstance(sel.value, list):
                        # * Multiselect — keep rows matching ANY selected value
                        str_vals = [str(v) for v in sel.value]
                        filtered = filtered[
                            filtered[col].astype(str).isin(str_vals)
                        ]

                    else:
                        # * Dropdown — single value match
                        str_val = str(sel.value)
                        filtered = filtered[
                            filtered[col].astype(str) == str_val
                        ]

                    log.debug(
                        f"[filters] Applied '{col}' = '{sel.value}' | "
                        f"rows {original_len} → {len(filtered)}"
                    )

                except Exception as e:
                    log.error(f"[filters] Failed to apply filter '{col}' | error={e}")
                    continue

            rows_removed = original_len - len(filtered)
            log.info(
                f"[filters] apply_filters complete | "
                f"{original_len} → {len(filtered)} rows | "
                f"{rows_removed} rows filtered out"
            )
            return filtered

    # ── Step 4: Summary ───────────────────────────────────────────────────────

    def get_summary(
        self,
        selections: Optional[list[FilterSelection]] = None,
    ) -> str:
        """
        # * Build a human-readable summary of active filters.
        # * Used in chart titles and PDF export headers.

        Args:
            selections : Uses self._selections if None

        Returns:
            String like "Region: Pune  |  Op bucket: NPA  |  Portfolio new: VASTU"
            or "All Data" if no active filters.

        Example:
            manager.get_summary()
            # → "Region: Pune  |  Bounce status: Tech, Non Tech"
        """
        sels = selections or self._selections
        if not sels:
            return "All Data"

        active = {
            sel.field: (
                ", ".join(sel.value) if isinstance(sel.value, list)
                else str(sel.value)
            )
            for sel in sels
            if sel.is_active
        }
        return build_filter_summary(active)

    def get_active_count(self) -> int:
        """# * Returns how many filters are currently active (not All)."""
        return sum(1 for s in self._selections if s.is_active)

    def get_selections_as_dict(self) -> dict[str, Any]:
        """
        # * Returns current selections as a plain dict.
        # * Used by session.py to persist filter state and by export functions.

        Returns:
            {"Region": "Pune", "Op bucket": ["NPA", "60-89 DPD"], ...}
            Only active (non-All) filters are included.
        """
        return {
            sel.field: sel.value
            for sel in self._selections
            if sel.is_active
        }

    def reset(self):
        """
        # * Clear all current selections.
        # * Does NOT clear Streamlit widget state — user must re-interact with sidebar.
        # ? Call this when a new query is loaded to start fresh.
        """
        self._selections  = []
        self._definitions = []
        log.debug("[filters] FilterManager reset")