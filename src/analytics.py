"""
analytics.py
------------
Calculates KPIs, detects anomalies, and runs sales forecasting.
No external API needed — uses pandas, scikit-learn, and Prophet.
"""

import pandas as pd
import numpy as np
from colorama import Fore, Style


def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Safely converts a pandas Series to numeric.
    Strips out currency symbols, commas, and spaces from dirty ERP data.

    NOTE: pandas 3.0 reports the dtype of plain string columns as 'str'
    instead of 'object', so a dtype-name check alone (the original
    implementation) silently skips the strip step on newer pandas and
    every currency string collapses to 0 via the fillna(0) below. We now
    check is_numeric_dtype directly instead of comparing dtype names, so
    this works correctly on both old and new pandas.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').fillna(0)
    # Anything not already numeric — strip currency symbols/commas/spaces
    cleaned = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0)


def _is_truly_numeric(series: pd.Series, min_unique: int = 5) -> bool:
    """
    Classification-only helper — NOT used for any KPI math.

    _clean_numeric() above deliberately fillna(0)s everything so KPI sums
    never crash on a stray bad row. That's correct for calculate_kpis, but
    it means an alphanumeric code column like 'SP01', 'INV20001' would get
    silently coerced to 0 for every row and look like "100% numeric, value
    0" to anything checking dtype after cleaning.

    This helper checks numeric-ness WITHOUT that fillna, so a code column
    correctly reports as non-numeric and lands in the categorical bucket
    instead of being miscounted as a constant-zero numeric column.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) >= min_unique

    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r'[₹$,\s]', '', regex=True)
    is_clean_number = cleaned.str.match(r'^-?\d+(\.\d+)?$')
    numeric_ratio = is_clean_number.mean()

    if numeric_ratio < 0.85:
        return False

    parsed = pd.to_numeric(cleaned.where(is_clean_number, np.nan), errors='coerce')
    return parsed.nunique(dropna=True) >= min_unique


def get_col(df: pd.DataFrame, mappings: list[dict], business_name: str) -> str | None:
    """
    Find the original column name for a given business meaning.
    Tries multiple strategies so KPIs work even when schema mapping
    is partial or uses unexpected business names.
    """
    search = business_name.lower()

    # Strategy 1: exact substring match on business_name
    for m in mappings:
        if search in m["business_name"].lower():
            if m["original_column"] in df.columns:
                return m["original_column"]

    # Strategy 2: any word in the search matches any word in business_name
    search_words = set(search.split())
    for m in mappings:
        mapped_words = set(m["business_name"].lower().split())
        if search_words & mapped_words:   # intersection — any common word
            col = m["original_column"]
            if col in df.columns:
                return col

    # Strategy 3: check if search term appears in the original column name
    for m in mappings:
        if search.replace(" ", "_") in m["original_column"].lower():
            if m["original_column"] in df.columns:
                return m["original_column"]

    return None


def calculate_kpis(df: pd.DataFrame, mappings: list[dict]) -> dict:
    """Calculate all standard sales KPIs."""
    kpis = {}

    rev_col    = get_col(df, mappings, "net sales")
    qty_col    = get_col(df, mappings, "quantity")
    cust_col   = get_col(df, mappings, "customer code")
    region_col = get_col(df, mappings, "region")
    date_col   = get_col(df, mappings, "invoice date")
    disc_col   = get_col(df, mappings, "discount amount")
    prod_col   = get_col(df, mappings, "product category")
    sp_col     = get_col(df, mappings, "salesperson")
    tot_col    = get_col(df, mappings, "total invoice")
    stat_col   = get_col(df, mappings, "payment status")

    if rev_col:
        rev_data = _clean_numeric(df[rev_col])
        kpis["total_revenue"]   = round(rev_data.sum(), 2)
        kpis["avg_order_value"] = round(rev_data.mean(), 2)
        kpis["max_order_value"] = round(rev_data.max(), 2)

    if qty_col:
        qty_data = _clean_numeric(df[qty_col])
        kpis["total_units_sold"] = int(qty_data.sum())

    if cust_col:
        kpis["unique_customers"] = df[cust_col].nunique()
        if rev_col:
            rev_data = _clean_numeric(df[rev_col])
            top = (rev_data.groupby(df[cust_col]).sum()
                     .sort_values(ascending=False).head(5))
            kpis["top_5_customers"] = top.to_dict()

    if region_col and rev_col:
        rev_data = _clean_numeric(df[rev_col])
        kpis["revenue_by_region"] = (
            rev_data.groupby(df[region_col]).sum()
              .sort_values(ascending=False).round(2).to_dict()
        )

    if prod_col and rev_col:
        rev_data = _clean_numeric(df[rev_col])
        kpis["revenue_by_category"] = (
            rev_data.groupby(df[prod_col]).sum()
              .sort_values(ascending=False).round(2).to_dict()
        )

    if sp_col and rev_col:
        rev_data = _clean_numeric(df[rev_col])
        kpis["revenue_by_salesperson"] = (
            rev_data.groupby(df[sp_col]).sum()
              .sort_values(ascending=False).round(2).to_dict()
        )

    if date_col and rev_col:
        rev_data = _clean_numeric(df[rev_col])
        df["_month"] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period("M")
        monthly = rev_data.groupby(df["_month"]).sum().round(2)
        kpis["monthly_revenue"] = {str(k): v for k, v in monthly.items()}
        df.drop(columns=["_month"], inplace=True)

    if disc_col and rev_col:
        disc_data = _clean_numeric(df[disc_col])
        rev_data  = _clean_numeric(df[rev_col])
        total_disc = disc_data.sum()
        total_rev  = rev_data.sum()
        kpis["total_discount_given"] = round(total_disc, 2)
        kpis["discount_as_pct_of_revenue"] = round((total_disc / total_rev * 100) if total_rev > 0 else 0, 2)

    if stat_col:
        kpis["payment_status_breakdown"] = df[stat_col].value_counts().to_dict()
        if tot_col:
            tot_data = _clean_numeric(df[tot_col])
            overdue = tot_data[df[stat_col].astype(str).str.upper() == "OVERDUE"].sum()
            kpis["overdue_amount"] = round(overdue, 2)

    return kpis


# ═════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION — upgraded engine
# ═════════════════════════════════════════════════════════════════════════════
#
# The previous version of this function ran exactly four hardcoded checks,
# each gated behind a single get_col() lookup for one specific business
# name ("net sales", "region", "discount percent"). If the schema mapper
# didn't resolve a column to exactly that name, the whole check silently
# produced nothing. A dataset with a salesperson-discount problem, or a
# weak product category, or a column with a 20% missing-data rate, got
# zero coverage even though real issues existed.
#
# This version classifies EVERY column by type first (numeric / categorical
# / date / identifier), then runs the right generic statistical test against
# every column of that type — so anomaly coverage no longer depends on the
# schema mapper guessing one specific business name correctly.
# ═════════════════════════════════════════════════════════════════════════════

def _classify_columns_for_anomalies(df: pd.DataFrame, mappings: list[dict]) -> dict:
    """
    Buckets every column into numeric / categorical / date / identifier
    using the mapper's declared category plus a live data-shape check
    (via _is_truly_numeric, NOT _clean_numeric, so code columns like
    'SP01' aren't miscounted as numeric zeros).

    customer/product NAME columns are excluded from the categorical bucket
    — they're typically high-cardinality with few rows per value, which
    produces noisy false-positive "spike" alerts rather than real signal.
    Segment-level columns (region, status, salesperson, product category)
    remain included.
    """
    cat_lookup = {m["original_column"]: m["category"] for m in mappings}
    NOISY_CATEGORIES = {"customer", "product"}

    buckets = {"numeric": [], "categorical": [], "date": [], "identifier": []}

    for col in df.columns:
        if col.endswith("_outlier_flag"):
            continue

        declared = cat_lookup.get(col, "other")

        if declared == "identifier":
            buckets["identifier"].append(col)
            continue
        if declared == "date":
            buckets["date"].append(col)
            continue

        if _is_truly_numeric(df[col]):
            buckets["numeric"].append(col)
            continue

        if declared in NOISY_CATEGORIES:
            continue

        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 50:
            avg_group_size = len(df) / max(nunique, 1)
            if avg_group_size >= 8:   # enough rows per group for a meaningful comparison
                buckets["categorical"].append(col)

    return buckets


def _check_numeric_outliers(df: pd.DataFrame, col: str, iqr_mult: float = 3.0) -> dict | None:
    """IQR-based outlier check — runs on any numeric column, not just revenue."""
    s = _clean_numeric(df[col]).dropna()
    if len(s) < 10:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    upper = q3 + iqr_mult * iqr
    lower = q1 - iqr_mult * iqr
    mask = (s > upper) | (s < lower)
    count = int(mask.sum())
    if count == 0:
        return None
    direction = "above" if s[mask].mean() > q3 else "below"
    return {
        "type":        f"Unusual values in '{col}'",
        "severity":    "HIGH" if count / len(s) < 0.02 else "MEDIUM",
        "description": (f"{count} value(s) in '{col}' are statistically {direction} the normal "
                         f"range (outside {iqr_mult}× IQR). Typical range: "
                         f"₹{q1:,.0f} to ₹{q3:,.0f}."),
        "affected":    count,
    }


def _check_categorical_imbalance(df: pd.DataFrame, cat_col: str, num_col: str,
                                  threshold: float = 0.6, agg: str = "sum") -> list[dict]:
    """
    Generic group-weakness check. Replaces the old hardcoded "weak region"
    check — now runs against EVERY categorical column (region, status,
    salesperson, product category, etc.), not just region.

    agg="sum"  for cumulative metrics (revenue, units sold)
    agg="mean" for rate metrics (discount %, margin %) where summing across
               a group would be meaningless
    """
    num_data = _clean_numeric(df[num_col])
    if agg == "mean":
        grouped = num_data.groupby(df[cat_col]).mean()
    else:
        grouped = num_data.groupby(df[cat_col]).sum()
    grouped = grouped.dropna()
    if len(grouped) < 2:
        return []
    mean_val = grouped.mean()
    if mean_val == 0:
        return []
    weak = grouped[grouped < mean_val * (1 - threshold)]
    results = []
    for group_name, val in weak.items():
        pct_below = round((1 - val / mean_val) * 100, 1)
        results.append({
            "type":        f"Underperforming {cat_col}",
            "severity":    "HIGH" if pct_below > 50 else "MEDIUM",
            "description": (f"'{group_name}' in '{cat_col}' has {agg} {num_col} of ₹{val:,.1f}, "
                             f"{pct_below}% below the group average of ₹{mean_val:,.1f}."),
            "affected":    str(group_name),
        })
    return results


def _check_categorical_spike(df: pd.DataFrame, cat_col: str, num_col: str,
                              threshold: float = 0.5, agg: str = "sum") -> list[dict]:
    """
    Inverse of the weakness check — flags groups performing suspiciously
    HIGH. The old detector had no equivalent of this at all; this is what
    catches a single salesperson's discount rate spiking far above peers,
    using mean aggregation so it isn't washed out by transaction-count
    differences between groups.
    """
    num_data = _clean_numeric(df[num_col])
    if agg == "mean":
        grouped = num_data.groupby(df[cat_col]).mean()
    else:
        grouped = num_data.groupby(df[cat_col]).sum()
    grouped = grouped.dropna()
    if len(grouped) < 2:
        return []
    mean_val = grouped.mean()
    if mean_val == 0:
        return []
    spiked = grouped[grouped > mean_val * (1 + threshold)]
    results = []
    for group_name, val in spiked.items():
        pct_above = round((val / mean_val - 1) * 100, 1)
        results.append({
            "type":        f"Unusually high {cat_col} group",
            "severity":    "MEDIUM",
            "description": (f"'{group_name}' in '{cat_col}' has {agg} {num_col} of ₹{val:,.1f}, "
                             f"{pct_above}% above the group average — worth reviewing for "
                             f"either a data error or a genuine opportunity."),
            "affected":    str(group_name),
        })
    return results


def _check_time_trend_drop(df: pd.DataFrame, date_col: str, num_col: str,
                            z_thresh: float = 1.5) -> list[dict]:
    """Month-over-month drop check — same logic as before, now reusable
    against any date column / numeric column pair found by the classifier."""
    num_data = _clean_numeric(df[num_col])
    months   = pd.to_datetime(df[date_col], errors='coerce').dt.to_period("M")
    valid    = months.notna()
    if valid.sum() < 20:
        return []
    monthly = num_data[valid].groupby(months[valid]).sum()
    if len(monthly) < 4:
        return []
    mean_m, std_m = monthly.mean(), monthly.std()
    if std_m == 0 or pd.isna(std_m):
        return []
    low_months = monthly[monthly < mean_m - z_thresh * std_m]
    results = []
    for month, val in low_months.items():
        results.append({
            "type":        f"Low {num_col} month",
            "severity":    "MEDIUM",
            "description": (f"{month}: '{num_col}' totals ₹{val:,.0f}, significantly below the "
                             f"monthly average of ₹{mean_m:,.0f}."),
            "affected":    str(month),
        })
    return results


def _check_null_spike(df: pd.DataFrame, col: str, threshold: float = 0.15) -> dict | None:
    """
    NEW check type — flags columns where more than `threshold` fraction of
    values are missing. A quietly-broken column is a data-quality anomaly
    distinct from a value anomaly, but equally damaging to trust in the
    numbers downstream (KPIs, charts) built from it.
    """
    null_ratio = df[col].isna().mean()
    if null_ratio < threshold:
        return None
    return {
        "type":        f"High missing-data rate in '{col}'",
        "severity":    "HIGH" if null_ratio > 0.35 else "MEDIUM",
        "description": (f"{null_ratio*100:.0f}% of values in '{col}' are missing — "
                         f"this may bias any KPI or chart built from this column."),
        "affected":    int(df[col].isna().sum()),
    }


def _check_duplicate_columns(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    """
    NEW check — detects two numeric columns that are near-perfectly
    correlated (>0.98), which usually means the ERP exported the same
    underlying value twice under different names. The old detector had no
    ability to surface this at all.
    """
    results = []
    cleaned = {c: _clean_numeric(df[c]) for c in numeric_cols}
    cols = [c for c in numeric_cols if (cleaned[c] != 0).sum() > 20]
    seen = set()
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            pair = tuple(sorted([c1, c2]))
            if pair in seen:
                continue
            seen.add(pair)
            corr = cleaned[c1].corr(cleaned[c2])
            if pd.notna(corr) and corr > 0.98:
                results.append({
                    "type":        "Possible duplicate columns",
                    "severity":    "LOW",
                    "description": (f"'{c1}' and '{c2}' are {corr*100:.1f}% correlated — "
                                     f"they may represent the same underlying value."),
                    "affected":    f"{c1} ↔ {c2}",
                })
    return results


def detect_anomalies(df: pd.DataFrame, mappings: list[dict], max_results: int = 25) -> list[dict]:
    """
    Generic anomaly detection across the whole dataset.

    Runs:
      1. Outlier detection on every numeric column
      2. Weak/spiked group detection on every categorical column ×
         the primary revenue column (sum-based) and every rate/percentage
         column (mean-based)
      3. Month-over-month drop detection on every date × revenue pair
      4. Missing-data spike detection on every column
      5. Duplicate-column detection across all numeric columns

    `mappings` is still used to find a primary revenue column where
    possible, but the engine no longer fails silently if get_col() can't
    resolve "net sales" — it still finds and checks every other numeric
    and categorical column regardless.
    """
    anomalies: list[dict] = []
    buckets = _classify_columns_for_anomalies(df, mappings)
    cat_lookup = {m["original_column"]: m["category"] for m in mappings}

    numeric_cols     = buckets["numeric"]
    categorical_cols = buckets["categorical"]
    date_cols        = buckets["date"]

    # Primary revenue-like column for sum-based cross-checks
    primary_num = get_col(df, mappings, "net sales")
    if primary_num is None and numeric_cols:
        sums = {c: _clean_numeric(df[c]).sum() for c in numeric_cols}
        primary_num = max(sums, key=sums.get)

    # Rate/percentage columns need mean-based comparison, not sum-based
    rate_cols = [c for c in numeric_cols if cat_lookup.get(c) == "percentage"]

    # ── 1. Outliers on every numeric column ───────────────────────────────────
    for col in numeric_cols:
        result = _check_numeric_outliers(df, col)
        if result:
            anomalies.append(result)

    # ── 2. Weak / spiked groups — every categorical × primary revenue ────────
    if primary_num:
        for cat_col in categorical_cols:
            anomalies.extend(_check_categorical_imbalance(df, cat_col, primary_num, agg="sum"))
            anomalies.extend(_check_categorical_spike(df, cat_col, primary_num, agg="sum"))

    # ── 2b. Weak / spiked groups on RATE columns (discount %, margin %) ──────
    for rate_col in rate_cols:
        for cat_col in categorical_cols:
            anomalies.extend(_check_categorical_spike(df, cat_col, rate_col,
                                                        threshold=0.3, agg="mean"))

    # ── 3. Time trend drops — every date column × primary revenue ────────────
    if primary_num:
        for date_col in date_cols:
            anomalies.extend(_check_time_trend_drop(df, date_col, primary_num))

    # ── 4. Missing-data anomalies — every column ──────────────────────────────
    for col in df.columns:
        if col.endswith("_outlier_flag"):
            continue
        result = _check_null_spike(df, col)
        if result:
            anomalies.append(result)

    # ── 5. Duplicate / redundant numeric columns ──────────────────────────────
    anomalies.extend(_check_duplicate_columns(df, numeric_cols))

    # ── De-duplicate and sort by severity ─────────────────────────────────────
    seen_keys, deduped = set(), []
    for a in anomalies:
        key = (a["type"], a["affected"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(a)

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    deduped.sort(key=lambda a: severity_rank.get(a["severity"], 3))
    return deduped[:max_results]


def run_forecast(df: pd.DataFrame, mappings: list[dict], months: int = 3) -> dict:
    """
    Forecast the specified number of months of revenue using a simple
    trend + seasonality model (no Prophet API needed).
    Falls back to Prophet if available.
    """
    rev_col  = get_col(df, mappings, "net sales")
    date_col = get_col(df, mappings, "invoice date")

    if not (rev_col and date_col):
        return {}

    rev_data = _clean_numeric(df[rev_col])
    df["_month"] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period("M")
    monthly = rev_data.groupby(df["_month"]).sum().sort_index()

    if len(monthly) < 4:
        return {"error": "Not enough data for forecasting (need at least 4 months)"}

    # simple linear trend forecast
    values = monthly.values
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)   # linear fit
    trend = np.poly1d(coeffs)

    last_period = monthly.index[-1]
    forecasts = {}
    for i in range(1, months + 1):
        next_val = max(0, trend(len(values) + i - 1))
        # add simple seasonality: Q4 bump
        future_month = (last_period + i).month
        if future_month in [10, 11, 12]:
            next_val *= 1.12
        month_label = str(last_period + i)
        forecasts[month_label] = round(next_val, 2)

    df.drop(columns=["_month"], inplace=True)
    return forecasts


def print_kpis(kpis: dict):
    def fmt(v):
        if isinstance(v, float) and v > 1000:
            return f"₹{v:,.2f}"
        if isinstance(v, int) and v > 1000:
            return f"{v:,}"
        return str(v)

    skip = {"top_5_customers", "revenue_by_region", "revenue_by_category",
            "revenue_by_salesperson", "monthly_revenue", "payment_status_breakdown"}

    for key, val in kpis.items():
        if key in skip:
            continue
        label = key.replace("_", " ").title()
        print(f"  {label:<35} {Fore.CYAN}{fmt(val)}{Style.RESET_ALL}")


def print_anomalies(anomalies: list[dict]):
    """Terminal pretty-printer for anomalies, mirrors print_kpis style."""
    if not anomalies:
        print(f"  {Fore.GREEN}No anomalies detected.{Style.RESET_ALL}")
        return
    for a in anomalies:
        color = Fore.RED if a["severity"] == "HIGH" else (
                Fore.YELLOW if a["severity"] == "MEDIUM" else Fore.CYAN)
        print(f"  {color}[{a['severity']}]{Style.RESET_ALL} {a['type']}")
        print(f"        {a['description']}")