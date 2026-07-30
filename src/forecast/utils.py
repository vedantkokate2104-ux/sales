"""
utils.py
--------
Small, reusable statistical helpers for the Forecast Intelligence Engine.

Numeric cleaning intentionally reuses analytics._clean_numeric so currency
stripping / coercion stays identical to the rest of the pipeline instead of
being re-implemented here.
"""

import numpy as np
import pandas as pd

from analytics import _clean_numeric   # reuse existing numeric-cleaning logic


# ── Frequency vocabulary ────────────────────────────────────────────────────
# Single source of truth for how a detected frequency name maps to (a) the
# seasonal cycle length used by Holt-Winters/SARIMA, (b) the pandas Period
# frequency code used to aggregate a raw date column, and (c) the pandas
# timestamp frequency alias used for date_range()/Prophet's
# make_future_dataframe(). Every model and the trainer read from this table
# rather than hardcoding "12" / "M" / "MS" individually, so frequency
# preservation is consistent everywhere it is used.

SEASONAL_PERIOD = {
    "daily":     7,     # weekly cycle within daily data
    "weekly":    52,    # yearly cycle within weekly data
    "monthly":   12,    # yearly cycle within monthly data
    "quarterly": 4,     # yearly cycle within quarterly data
    "yearly":    None,  # no sub-annual seasonal cycle is possible
    "irregular": None,
    "unknown":   None,
}

PERIOD_FREQ_CODE = {
    "daily":     "D",
    "weekly":    "W",
    "monthly":   "M",
    "quarterly": "Q",
    "yearly":    "Y",
}

TIMESTAMP_FREQ_CODE = {
    "daily":     "D",
    "weekly":    "W",
    "monthly":   "MS",
    "quarterly": "QS",
    "yearly":    "AS",
}


def seasonal_period_for_freq(frequency: str):
    """Seasonal cycle length (in periods) for a detected frequency name, or
    None if that frequency has no meaningful sub-cycle (yearly data) or is
    not a clean, regular cadence (irregular/unknown)."""
    return SEASONAL_PERIOD.get(frequency)


def period_freq_code(frequency: str, default: str = "M") -> str:
    """pandas Period frequency code (for .dt.to_period() / PeriodIndex /
    period_range()) for a detected frequency name. Falls back to monthly
    ("M") for irregular/unknown data — a fallback grain to make training
    possible at all, not a default override of a genuinely detected
    frequency."""
    return PERIOD_FREQ_CODE.get(frequency, default)


def timestamp_freq_code(frequency: str, default: str = "MS") -> str:
    """pandas timestamp frequency alias (for date_range() / Prophet's
    make_future_dataframe()) for a detected frequency name. Same
    irregular/unknown fallback rationale as period_freq_code()."""
    return TIMESTAMP_FREQ_CODE.get(frequency, default)


def effective_training_frequency(frequency: str, default: str = "monthly") -> str:
    """
    The CANONICAL frequency name a training series was actually built at —
    identical to `frequency` for any recognised cadence (daily/weekly/
    monthly/quarterly/yearly), but resolved to the same "monthly" fallback
    grain that period_freq_code()/timestamp_freq_code() already use
    whenever `frequency` is "irregular"/"unknown"/anything else not in the
    vocabulary. Model code should key seasonal-period/frequency-alias
    lookups off THIS value, not the raw detected label, so a model never
    sees a frequency name ("irregular") that doesn't match the actual
    PeriodIndex grain of the series it was handed.
    """
    return frequency if frequency in PERIOD_FREQ_CODE else default


def infer_frequency(dates: pd.Series) -> str:
    """
    Infer whether a sorted date series is daily / weekly / monthly /
    quarterly / yearly / irregular, based on the median gap between
    consecutive unique dates. Quarterly and yearly were added so a
    dataset that genuinely reports on a quarterly or yearly cadence is no
    longer misclassified as "irregular" (its median gap is a clean ~90 or
    ~365 days, not erratic — the old 4-bucket thresholds simply had nowhere
    to put it).
    """
    unique_dates = pd.Series(sorted(pd.to_datetime(dates).dropna().unique()))
    if len(unique_dates) < 2:
        return "unknown"
    gaps = unique_dates.diff().dropna().dt.days
    if gaps.empty:
        return "unknown"
    median_gap = gaps.median()
    if median_gap <= 1.5:
        return "daily"
    if median_gap <= 9:
        return "weekly"
    if median_gap <= 45:
        return "monthly"
    if median_gap <= 135:
        return "quarterly"
    if median_gap <= 450:
        return "yearly"
    return "irregular"


def count_missing_timestamps(dates: pd.Series, frequency: str) -> int:
    """
    Count how many expected periods (at the inferred frequency) between the
    first and last observation have no corresponding data point at all.
    Returns 0 for "irregular"/"unknown" frequencies, where there is no fixed
    expected cadence to compare against.
    """
    if frequency not in TIMESTAMP_FREQ_CODE:
        return 0
    code = timestamp_freq_code(frequency)
    period_freq = period_freq_code(frequency)

    unique_dates = pd.to_datetime(pd.Series(dates)).dropna().unique()
    unique_dates = pd.Series(sorted(unique_dates))
    if len(unique_dates) < 2:
        return 0

    full_range = pd.date_range(unique_dates.iloc[0], unique_dates.iloc[-1], freq=code)
    observed = set(pd.PeriodIndex(unique_dates, freq=period_freq).astype(str))
    expected = set(pd.PeriodIndex(full_range, freq=period_freq).astype(str))
    return max(0, len(expected) - len(observed))


def outlier_percentage(values: pd.Series, iqr_mult: float = 1.5) -> float:
    """
    Percentage of values that fall outside iqr_mult x IQR of the series.
    Mirrors the IQR convention already used by analytics.py's anomaly
    detector, at the standard 1.5x multiplier rather than the anomaly
    detector's intentionally wide 3x/4x — here it is a descriptive
    dataset-quality statistic, not an anomaly-flagging threshold.
    """
    s = _clean_numeric(values).dropna()
    if len(s) < 4:
        return 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    outliers = ((s < lower) | (s > upper)).sum()
    return round(float(outliers) / len(s) * 100, 2)


def trend_strength(values: np.ndarray) -> float:
    """
    Dependency-free trend-strength proxy: the absolute correlation between
    the time index and the value. 0 = no linear trend, 1 = a perfectly
    linear trend. This is a descriptive statistic used for classification
    only — it is NOT a forecasting model.
    """
    n = len(values)
    if n < 3 or np.std(values) == 0:
        return 0.0
    x = np.arange(n)
    corr = np.corrcoef(x, values)[0, 1]
    return 0.0 if np.isnan(corr) else round(abs(float(corr)), 4)


def seasonality_strength(values: np.ndarray, period: int) -> float:
    """
    Dependency-free seasonality-strength proxy: the share of total variance
    explained by the mean value at each position-in-cycle (e.g. each
    calendar month for a monthly series with period=12). 0 = no seasonal
    pattern, 1 = value is fully determined by its position in the cycle.
    """
    n = len(values)
    if period < 2 or n < period * 2:
        return 0.0
    total_var = np.var(values)
    if total_var == 0:
        return 0.0
    positions = np.arange(n) % period
    seasonal_means = np.array([values[positions == p].mean() for p in range(period)])
    predicted = seasonal_means[positions]
    residual_var = np.var(np.asarray(values) - predicted)
    strength = 1 - (residual_var / total_var)
    return round(max(0.0, min(1.0, float(strength))), 4)
