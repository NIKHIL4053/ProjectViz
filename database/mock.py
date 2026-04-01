"""
database/mock.py
----------------
# * Mock data layer for the Loan Dashboard project.
# * Returns realistic fake DataFrames when USE_MOCK_DATA=true in .env.
# * Allows full UI development and demo without a live PostgreSQL connection.

# ? How mock routing works:
# ?   client.py calls get_mock_data(sql)
# ?   get_mock_data() reads the SQL to detect query intent
# ?   Returns a DataFrame that matches what the real query would return
# ?   All 43 columns from the data dictionary are represented

# ? Mock data is seeded with numpy.random.seed(42) for reproducibility.
# ? The same question always returns the same data — important for demos.

# ! Mock data is NOT stored to disk — it is generated fresh per call.
# ! Do not use mock data for benchmarking query performance.

Exports:
    - get_mock_data()   : Main entry point — detects intent, returns DataFrame
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# * Fixed seed for reproducible demo data
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# * MASTER DATA — shared across all mock generators
# * These are the actual real-world values from the data dictionary.
# ──────────────────────────────────────────────────────────────────────────────

_REGIONS   = ["Pune", "NCR", "Rajasthan", "MP", "Karnataka", "Telangana"]
_BRANCHES  = [
    "PUNE", "NASHIK", "JAIPUR", "UDAIPUR", "LUCKNOW",
    "KANPUR", "BANGALORE", "MYSORE", "HYDERABAD", "DELHI NCR",
    "BHOPAL", "INDORE", "CHENNAI", "AHMEDABAD"
]
_REGION_MAP = {
    "PUNE": "Pune", "NASHIK": "Pune",
    "JAIPUR": "Rajasthan", "UDAIPUR": "Rajasthan",
    "LUCKNOW": "NCR", "KANPUR": "NCR", "DELHI NCR": "NCR",
    "BANGALORE": "Karnataka", "MYSORE": "Karnataka",
    "HYDERABAD": "Telangana",
    "BHOPAL": "MP", "INDORE": "MP",
    "CHENNAI": "Telangana",
    "AHMEDABAD": "Rajasthan",
}

_OP_BUCKETS    = ["Current", "Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD", "NPA", "Write-off"]
_OP_WEIGHTS    = [0.45, 0.15, 0.15, 0.10, 0.07, 0.05, 0.03]

_CUST_STATUSES = ["Current", "Norm", "Flow", "Stab", "Roll Back", "Risk NPA", "NPA", "Write-off"]
_CUST_WEIGHTS  = [0.42, 0.18, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02]

_BOUNCE_STATUS = ["PAID", "Tech", "Non Tech"]
_BOUNCE_WEIGHTS= [0.65, 0.20, 0.15]

_PAYMENT_MODES = ["Cash", "Cheque", "NEFT", "RTGS", "Demand Draft", "Blank"]
_PAYMENT_WEIGHTS=[0.20, 0.10, 0.35, 0.20, 0.05, 0.10]

_PAY_DAY_BUCKETS = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-27", "28-31", "NA"]
_MOB_BUCKETS   = ["1-6 MOB", "7-12 MOB", "13-18 MOB", "19-24 MOB", "24+ MOB"]

_PORTFOLIOS    = [
    "VASTU", "State Bank of India DA 1-3",
    "Tata Capital DA01", "Other"
]
_PRODUCTS      = ["Retail", "Micro Housing"]
_TX_TYPES      = [
    "RESIDENTIAL LAP", "COMMERCIAL LAP",
    "BALANCE TRANSFER - RESIDENTIAL LAP",
    "SELF CONSTRUCTION", "COMMERCIAL PURCHASE"
]
_SCHEMES       = [
    "PD ASSESSMENT-FLOATING", "BANKING 360-FLOATING",
    "RENTAL INCOME PROGRAM-FLOATING", "IEP-FLOATING"
]
_ARRANGEMENTS  = [
    "VASTU", "State Bank of India - DA01",
    "Tata Capital - DA01", "Dhansafal"
]

_VISITED_VALUES = [
    "Visited before due date", "Visit After Due date",
    "Visit before and after due date", "Not visited"
]
_VISITED_WEIGHTS = [0.25, 0.30, 0.15, 0.30]

_COLL_SALES    = ["Collection", "Sales"]

_TLS = [
    "Ramesh Kumar", "Priya Shah", "Amit Verma",
    "Sunita Rao", "Vikram Nair", "No TL"
]
_SHS = ["Anil Mehta", "Kavita Joshi", "Suresh Pillai"]


# ──────────────────────────────────────────────────────────────────────────────
# * FULL TABLE GENERATOR
# * Builds a realistic 89K-row base DataFrame matching all 43 columns.
# ──────────────────────────────────────────────────────────────────────────────

def _generate_full_table(n: int = 89255) -> pd.DataFrame:
    """
    # * Generate a full mock dataset matching the real loan_dashboard table schema.
    # * All 43 columns populated with realistic correlated values.

    Args:
        n : Number of rows to generate (default matches real dataset size)

    Returns:
        DataFrame with all 43 columns.
    """
    rng = np.random.default_rng(42)

    # * Generate base fields
    branches   = rng.choice(_BRANCHES, n, p=[1/len(_BRANCHES)] * len(_BRANCHES))
    regions    = [_REGION_MAP.get(b, "Pune") for b in branches]
    op_buckets = rng.choice(_OP_BUCKETS,    n, p=_OP_WEIGHTS)
    dpd_values = _generate_dpd(op_buckets, rng)
    bounce     = rng.choice(_BOUNCE_STATUS, n, p=_BOUNCE_WEIGHTS)
    emi_inc    = rng.choice([0, 1],         n, p=[0.85, 0.15])

    # * Closing bucket correlated with op bucket
    closing    = _generate_closing_bucket(op_buckets, rng)

    # * Cust wise status derived from op/closing bucket
    cust_status = _generate_cust_status(op_buckets, closing)

    # * Risk NPA — only relevant in 60-89 DPD
    risk_npa = np.where(
        (np.array(op_buckets) == "60-89 DPD") & (rng.random(n) < 0.3),
        1, 0
    )

    # * Asset classification
    asset_class = np.where(
        np.isin(op_buckets, ["NPA", "Write-off"]),
        rng.choice(["Substandard Asset", "Doubtful Asset"], n),
        "Standard Asset"
    )

    # * Financial fields
    loan_amounts  = rng.lognormal(13.0, 0.8, n).round(2)
    dpd_casewise  = np.maximum(0, dpd_values + rng.integers(-5, 5, n))
    tod           = np.where(
        np.array(dpd_values) > 0,
        (loan_amounts * 0.03 * dpd_values / 30).round(2),
        0.0
    )
    bal_prin      = (loan_amounts * rng.uniform(0.3, 0.99, n)).round(2)

    # * Collection team
    allocation    = _generate_allocations(n, rng)
    allocated_not = ["Allocated" if a != "NA" else "Non Allocated" for a in allocation]
    tls           = rng.choice(_TLS, n, p=[0.18, 0.17, 0.17, 0.17, 0.17, 0.14])
    shs           = [_SHS[0] if t in _TLS[:2] else
                     _SHS[1] if t in _TLS[2:4] else
                     _SHS[2] for t in tls]

    # * Visit fields
    visited_vals  = rng.choice(_VISITED_VALUES, n, p=_VISITED_WEIGHTS)
    visit_or_not  = ["Visited" if v != "Not visited" else "Not Visited" for v in visited_vals]
    visit_count   = np.where(
        np.array(visit_or_not) == "Visited",
        rng.integers(1, 6, n),
        0
    )

    # * Payment fields
    payment_mode  = rng.choice(_PAYMENT_MODES,    n, p=_PAYMENT_WEIGHTS)
    pay_day       = rng.choice(_PAY_DAY_BUCKETS,  n)
    digital_cash  = np.where(
        np.isin(payment_mode, ["Blank"]), None,
        np.where(payment_mode == "Cash", "Cash", "Digital")
    )
    loan_bounce   = np.where(np.isin(bounce, ["Tech", "Non Tech"]), 1, 0)
    cust_bounce   = loan_bounce  # * simplified — same for mock

    # * Bounce charges
    bounce_charges   = np.where(loan_bounce == 1, rng.uniform(500, 2500, n).round(2), 0.0)
    bounce_collected = np.where(
        loan_bounce == 1,
        (bounce_charges * rng.uniform(0.4, 1.0, n)).round(2),
        0.0
    )
    overdue_amt   = (bounce_charges - bounce_collected).round(2)

    # * NPA fields
    npa_mob       = np.where(np.isin(op_buckets, ["NPA", "Write-off"]),
                             rng.integers(1, 24, n), None)
    add_npa       = np.where(
        np.isin(op_buckets, ["60-89 DPD"]) & (rng.random(n) < 0.2), 1, 0
    )
    npa_orig_date = np.where(
        npa_mob is not None,
        pd.Timestamp("2025-01-01") - pd.to_timedelta(
            np.where(npa_mob is not None, npa_mob, 0) * 30, unit="D"
        ),
        None
    )

    # * Other fields
    mob_bucket    = rng.choice(_MOB_BUCKETS, n)
    installment   = rng.integers(1, 60, n)
    schemes       = rng.choice(_SCHEMES, n)
    products      = rng.choice(_PRODUCTS, n, p=[0.6, 0.4])
    portfolios    = rng.choice(_PORTFOLIOS, n, p=[0.35, 0.30, 0.25, 0.10])
    arrangements  = rng.choice(_ARRANGEMENTS, n)
    tx_types      = rng.choice(_TX_TYPES, n)
    coll_sales    = rng.choice(_COLL_SALES, n, p=[0.80, 0.20])

    # * Build loanappno — LAP or HL prefix
    loan_type_flag = rng.random(n)
    loanappnos    = [
        f"LAP{str(i).zfill(12)}" if loan_type_flag[i] < 0.6 else f"HL{str(i).zfill(13)}"
        for i in range(n)
    ]
    cust_ids      = [f"CN{str(rng.integers(10000, 99999)):>5}" for _ in range(n)]

    # * Date fields
    next_dates = pd.Timestamp("2026-02-01")
    due_dates  = rng.choice(pd.date_range("2026-02-01", "2026-02-28"), n)

    df = pd.DataFrame({
        "loanappno":                  loanappnos,
        "dpd":                        dpd_values.astype(int),
        "dpd_casewise":               dpd_casewise.astype(int),
        "day_wise_asset_classification": asset_class,
        "TOD":                        tod,
        "bal_prin":                   bal_prin,
        "Next Date":                  next_dates,
        "Bounce status":              bounce,
        "Emi increase":               emi_inc.astype(int),
        "Cust ID":                    cust_ids,
        "Risk NPA":                   risk_npa.astype(int),
        "Op bucket":                  op_buckets,
        "Closing bucket":             closing,
        "Cust wise status":           cust_status,
        "Due Date":                   due_dates,
        "Installment No":             installment.astype(int),
        "Branch":                     branches,
        "Type of Arrangement":        arrangements,
        "Scheme":                     schemes,
        "Payment mode":               payment_mode,
        "Payment Day bucket":         pay_day,
        "Digital/cash":               digital_cash,
        "Loan level bounce":          loan_bounce.astype(int),
        "Cust level bounce":          cust_bounce.astype(int),
        "Region":                     regions,
        "MOB Bucket":                 mob_bucket,
        "overdue_amount":             overdue_amt,
        "Bounce charges":             bounce_charges,
        "Bounce charge collected":    bounce_collected,
        "Visited":                    visited_vals,
        "Allocation 1":               allocation,
        "SH":                         shs,
        "TL":                         tls,
        "Allocated or not":           allocated_not,
        "Visit or not":               visit_or_not,
        "Visit Count":                visit_count.astype(int),
        "Add NPA":                    add_npa.astype(int),
        "Coll/Sales":                 coll_sales,
        "NPA_Origination_Date":       npa_orig_date,
        "NPA MOB":                    npa_mob,
        "Portfolio new":              portfolios,
        "Transaction Type":           tx_types,
        "Product":                    products,
    })

    return df


# ── Helper generators ─────────────────────────────────────────────────────────

def _generate_dpd(op_buckets: np.ndarray, rng) -> np.ndarray:
    """# * Generate DPD values correlated with Op bucket."""
    dpd = np.zeros(len(op_buckets), dtype=float)
    bucket_dpd = {
        "Current":   (0,   0),
        "Risk X":    (0,   0),
        "1-29 DPD":  (1,  29),
        "30-59 DPD": (30, 59),
        "60-89 DPD": (60, 89),
        "NPA":       (90, 360),
        "Write-off": (90, 720),
    }
    for bucket, (low, high) in bucket_dpd.items():
        mask = np.array(op_buckets) == bucket
        if mask.any():
            dpd[mask] = rng.integers(low, max(high, low + 1), mask.sum())
    return dpd


def _generate_closing_bucket(op_buckets: list, rng) -> list:
    """# * Generate closing bucket with realistic movement from opening bucket."""
    closing = []
    for ob in op_buckets:
        if ob == "Current":
            closing.append(rng.choice(["Current", "Risk X"], p=[0.88, 0.12]))
        elif ob == "Risk X":
            closing.append(rng.choice(
                ["Current", "Risk X", "1-29 DPD"], p=[0.55, 0.30, 0.15]
            ))
        elif ob == "1-29 DPD":
            closing.append(rng.choice(
                ["Current", "Risk X", "1-29 DPD", "30-59 DPD"],
                p=[0.30, 0.25, 0.30, 0.15]
            ))
        elif ob == "30-59 DPD":
            closing.append(rng.choice(
                ["Current", "Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD"],
                p=[0.20, 0.15, 0.20, 0.30, 0.15]
            ))
        elif ob == "60-89 DPD":
            closing.append(rng.choice(
                ["Current", "Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD", "NPA"],
                p=[0.10, 0.08, 0.12, 0.20, 0.30, 0.20]
            ))
        elif ob == "NPA":
            closing.append(rng.choice(["NPA", "Write-off"], p=[0.90, 0.10]))
        else:
            closing.append("Write-off")
    return closing


def _generate_cust_status(op_buckets: list, closing_buckets: list) -> list:
    """# * Derive Cust wise status from Op → Closing bucket movement."""
    status = []
    current_like = {"Current", "Risk X"}
    for ob, cb in zip(op_buckets, closing_buckets):
        if ob == "Current":
            status.append("Current")
        elif ob in {"Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD", "NPA", "Write-off"}:
            if cb in current_like:
                status.append("Norm")
            elif cb in {"1-29 DPD"} and ob in {"30-59 DPD", "60-89 DPD"}:
                status.append("Roll Back")
            elif cb == ob:
                status.append("Stab")
            elif cb in {"NPA", "Write-off", "60-89 DPD"}:
                status.append("Risk NPA")
            else:
                status.append("Flow")
        else:
            status.append("Current")
    return status


def _generate_allocations(n: int, rng) -> list:
    """# * Generate field executive names — 20% unallocated."""
    fes = [
        "Rajesh Patil", "Suman Gupta", "Praveen Kumar",
        "Deepa Nair", "Arjun Singh", "Meena Sharma",
        "Karan Mehta", "Pooja Iyer", "NA"
    ]
    weights = [0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10, 0.10]
    return rng.choice(fes, n, p=weights).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# * INTENT DETECTION
# * Reads the SQL and decides which mock scenario to return.
# ──────────────────────────────────────────────────────────────────────────────

def _detect_intent(sql: str) -> str:
    """
    # * Detect query intent from SQL keywords and return a scenario tag.
    # * Used to select the right mock data shape.

    Args:
        sql : SQL string from sql_generator.py

    Returns:
        Intent string — one of the scenario keys below.
    """
    sql_lower = sql.lower()

    # * Check GROUP BY columns to understand aggregation shape
    if "group by" in sql_lower:
        if "op bucket" in sql_lower and "closing bucket" in sql_lower:
            return "bucket_movement"
        if "op bucket" in sql_lower:
            return "bucket_distribution"
        if "branch" in sql_lower and "region" in sql_lower:
            return "branch_region"
        if "branch" in sql_lower:
            return "by_branch"
        if "region" in sql_lower:
            return "by_region"
        if "tl" in sql_lower or "team leader" in sql_lower:
            return "by_tl"
        if "allocation 1" in sql_lower:
            return "fe_scorecard"
        if "cust wise status" in sql_lower:
            return "status_summary"
        if "mob bucket" in sql_lower:
            return "by_mob"
        if "portfolio new" in sql_lower:
            return "by_portfolio"
        if "digital/cash" in sql_lower:
            return "payment_channel"

    # * Single value queries
    if "bounce %" in sql_lower or "bounce_percent" in sql_lower:
        return "single_bounce_pct"
    if "resolution %" in sql_lower or "resolution_percent" in sql_lower:
        return "single_resolution_pct"
    if "coverage %" in sql_lower or "coverage_percent" in sql_lower:
        return "single_coverage_pct"

    # * Distribution queries (no GROUP BY, many rows)
    if "dpd_casewise" in sql_lower and "group by" not in sql_lower:
        return "dpd_distribution"

    # * Default — return grouped branch data
    return "by_branch"


# ──────────────────────────────────────────────────────────────────────────────
# * MOCK DATA GENERATORS — one per scenario
# ──────────────────────────────────────────────────────────────────────────────

def _mock_by_branch(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Branch")
        .agg(
            Bounce_Count    = ("Bounce status", lambda x: (x.isin(["Tech","Non Tech"])).sum()),
            Total_Customers = ("Cust ID", "nunique"),
            Resolved_Count  = ("Cust wise status", lambda x: (x=="Norm").sum()),
            Balance_Principal = ("bal_prin", "sum"),
        )
        .round(2)
        .reset_index()
        .rename(columns={"Branch": "Branch"})
    )


def _mock_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region")
        .agg(
            Bounce_Count    = ("Bounce status", lambda x: (x.isin(["Tech","Non Tech"])).sum()),
            Total_Customers = ("Cust ID", "nunique"),
            Balance_Principal = ("bal_prin", "sum"),
        )
        .round(2)
        .reset_index()
    )


def _mock_bucket_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Op bucket")
        .agg(
            Loan_Count      = ("loanappno", "count"),
            Customer_Count  = ("Cust ID", "nunique"),
            Balance_Principal = ("bal_prin", "sum"),
            Total_Overdue   = ("TOD", "sum"),
        )
        .round(2)
        .reset_index()
    )


def _mock_bucket_movement(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Op bucket", "Closing bucket"])
        .agg(
            Loan_Count     = ("loanappno", "count"),
            Customer_Count = ("Cust ID", "nunique"),
        )
        .reset_index()
    )


def _mock_by_tl(df: pd.DataFrame) -> pd.DataFrame:
    filt = df[df["TL"] != "No TL"]
    result = (
        filt.groupby("TL")
        .apply(lambda g: pd.Series({
            "Visited Count":    ((g["Visit or not"]=="Visited") & (g["Allocated or not"]=="Allocated")).sum(),
            "Allocated Count":  (g["Allocated or not"]=="Allocated").sum(),
            "Resolved Count":   (g["Cust wise status"]=="Norm").nunique(),
            "Total Customers":  g["Cust ID"].nunique(),
        }))
        .reset_index()
    )
    result["Coverage %"] = (
        result["Visited Count"] * 100.0 / result["Allocated Count"].replace(0, np.nan)
    ).round(2)
    result["Resolution %"] = (
        result["Resolved Count"] * 100.0 / result["Total Customers"].replace(0, np.nan)
    ).round(2)
    return result.sort_values("Coverage %", ascending=False)


def _mock_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Cust wise status")
        .agg(
            Customer_Count = ("Cust ID", "nunique"),
            Loan_Count     = ("loanappno", "count"),
        )
        .reset_index()
        .sort_values("Customer_Count", ascending=False)
    )


def _mock_fe_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    filt = df[df["Allocation 1"] != "NA"]
    return (
        filt.groupby(["Allocation 1", "TL"])
        .apply(lambda g: pd.Series({
            "Resolution %":  round((g["Cust wise status"]=="Norm").sum() * 100.0 / max(g["Cust ID"].nunique(), 1), 2),
            "Coverage %":    round(((g["Visit or not"]=="Visited") & (g["Allocated or not"]=="Allocated")).sum() * 100.0 / max((g["Allocated or not"]=="Allocated").sum(), 1), 2),
            "Intensity":     round(g["Visit Count"].sum() / max(g["Cust ID"].nunique(), 1), 2),
            "Allocated Count": (g["Allocated or not"]=="Allocated").sum(),
        }))
        .reset_index()
        .sort_values("Resolution %", ascending=False)
    )


def _mock_by_mob(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("MOB Bucket")
        .agg(
            Loan_Count      = ("loanappno", "count"),
            Bounce_Count    = ("Bounce status", lambda x: x.isin(["Tech","Non Tech"]).sum()),
            Balance_Principal = ("bal_prin", "sum"),
        )
        .round(2)
        .reset_index()
    )


def _mock_by_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Portfolio new")
        .agg(
            Loan_Count      = ("loanappno", "count"),
            Customer_Count  = ("Cust ID", "nunique"),
            Balance_Principal = ("bal_prin", "sum"),
            Bounce_Count    = ("Bounce status", lambda x: x.isin(["Tech","Non Tech"]).sum()),
        )
        .round(2)
        .reset_index()
    )


def _mock_payment_channel(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["Digital/cash"].notna()]
        .groupby(["MOB Bucket", "Digital/cash"])
        .agg(Customer_Count = ("Cust ID", "nunique"))
        .reset_index()
    )


def _mock_dpd_distribution(df: pd.DataFrame, sample: int = 2000) -> pd.DataFrame:
    return (
        df[["dpd_casewise", "Op bucket", "Branch", "Region"]]
        .dropna()
        .sample(min(sample, len(df)), random_state=42)
        .reset_index(drop=True)
    )


def _mock_branch_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Branch", "Region"])
        .agg(
            Bounce_Count    = ("Bounce status", lambda x: x.isin(["Tech","Non Tech"]).sum()),
            Customer_Count  = ("Cust ID", "nunique"),
        )
        .reset_index()
    )


def _mock_single_value(metric: str, df: pd.DataFrame) -> pd.DataFrame:
    """# * Return a single-row DataFrame for scalar KPI metrics."""
    if metric == "single_bounce_pct":
        bounced = df["Bounce status"].isin(["Tech","Non Tech"]).sum()
        total   = df["Cust ID"].nunique()
        val     = round(bounced * 100.0 / max(total, 1), 2)
        return pd.DataFrame({"Bounce %": [val]})

    elif metric == "single_resolution_pct":
        resolved = (df["Cust wise status"] == "Norm").sum()
        total    = df["Cust ID"].nunique()
        val      = round(resolved * 100.0 / max(total, 1), 2)
        return pd.DataFrame({"Resolution %": [val]})

    elif metric == "single_coverage_pct":
        visited  = ((df["Visit or not"]=="Visited") & (df["Allocated or not"]=="Allocated")).sum()
        alloc    = (df["Allocated or not"]=="Allocated").sum()
        val      = round(visited * 100.0 / max(alloc, 1), 2)
        return pd.DataFrame({"Coverage %": [val]})

    return pd.DataFrame({"Value": [0]})


# ──────────────────────────────────────────────────────────────────────────────
# * MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

# * Cache the base DataFrame — generate once per app session
_base_df: pd.DataFrame = None


def _get_base_df() -> pd.DataFrame:
    """# * Returns the cached base DataFrame, generating it if needed."""
    global _base_df
    if _base_df is None:
        log.info("[mock] Generating base mock dataset...")
        _base_df = _generate_full_table()
        log.info(f"[mock] Base dataset ready | rows={len(_base_df)} | cols={len(_base_df.columns)}")
    return _base_df


def get_mock_data(sql: str) -> pd.DataFrame:
    """
    # * Main entry point — detect SQL intent and return matching mock DataFrame.
    # * Called by client.py when USE_MOCK_DATA=true.

    Args:
        sql : SQL string from sql_generator.py

    Returns:
        DataFrame matching the shape the real query would return.
    """
    df     = _get_base_df()
    intent = _detect_intent(sql)

    log.info(f"[mock] Routing | intent={intent} | sql_preview=\"{sql[:80]}\"")

    scenario_map = {
        "by_branch":         lambda: _mock_by_branch(df),
        "by_region":         lambda: _mock_by_region(df),
        "bucket_distribution": lambda: _mock_bucket_distribution(df),
        "bucket_movement":   lambda: _mock_bucket_movement(df),
        "by_tl":             lambda: _mock_by_tl(df),
        "status_summary":    lambda: _mock_status_summary(df),
        "fe_scorecard":      lambda: _mock_fe_scorecard(df),
        "by_mob":            lambda: _mock_by_mob(df),
        "by_portfolio":      lambda: _mock_by_portfolio(df),
        "payment_channel":   lambda: _mock_payment_channel(df),
        "dpd_distribution":  lambda: _mock_dpd_distribution(df),
        "branch_region":     lambda: _mock_branch_region(df),
        "single_bounce_pct": lambda: _mock_single_value("single_bounce_pct", df),
        "single_resolution_pct": lambda: _mock_single_value("single_resolution_pct", df),
        "single_coverage_pct":   lambda: _mock_single_value("single_coverage_pct", df),
    }

    generator = scenario_map.get(intent, lambda: _mock_by_branch(df))
    result    = generator()

    log.info(f"[mock] Returned | intent={intent} | rows={len(result)} | cols={list(result.columns)}")
    return result