"""
preprocessing.py
-----------------
Automated data cleaning for the Auto Forecasting Engine (Tasks 2 and 11
of the production-grade forecasting pass). Forecasting should never run
directly on raw uploaded data — this module sits between the raw
(date, revenue) columns and forecast.trainer.build_series(), correcting
what can safely be corrected and disclosing everything it changed.

This does not replace or duplicate forecast.validation.validate_dataset()
(which reports data-quality WARNINGS about the original data, unchanged,
for transparency) — it performs the actual CORRECTIONS, and its own notes
are merged into that same warnings list by forecast_engine.py so the user
sees one coherent explanation of both what was found and what was done
about it.

Three stages, run in this order (each documented at its own function):
  1. clean_transactions()      — row-level cleanup, BEFORE aggregation
  2. (aggregation happens in forecast.trainer.build_series(), unchanged)
  3. cap_outliers() + interpolate_short_gaps() — period-level cleanup,
     AFTER aggregation, since "an outlier" and "a gap" are only
     meaningful once the data has a fixed periodicity.

Deterministic date parsing (runtime stability pass, Task 10): both
clean_transactions() (below) and forecast.trainer.build_series() used to
call `pd.to_datetime(dates, errors="coerce")` with no explicit format.
When pandas cannot infer one consistent format from the first few rows —
common with day-first, 2-digit-year formats like this project's own
sample dataset ("16/10/24") — it silently falls back to parsing every row
INDIVIDUALLY via `dateutil`, printing a `UserWarning: Could not infer
format`. That per-row fallback is not just noisy, it is a genuine
correctness risk: a value like "06/04/23" is valid as EITHER 6 April 2023
or 4 June 2023, and dateutil decides row by row (based on each row's own
heuristics) rather than applying one consistent rule for the whole
column — so two rows from the same column, using the same format, can end
up parsed with different day/month assignments. parse_dates_deterministic()
instead tries a fixed, ordered list of explicit formats against the WHOLE
column, keeps whichever format parses a clear majority of it, and parses
every row with that ONE format — every date in a column is interpreted
the same way, instead of format inference potentially flipping day/month
between rows.
"""

import logging
import warnings as _warnings_module
from typing import Dict, List, Tuple

import pandas as pd

from analytics import _clean_numeric

logger = logging.getLogger(__name__)

# Explicit candidate date formats, tried in this order against the WHOLE
# date column. ISO-style (unambiguous) formats come first. Day-first
# formats are prioritised over month-first ones for the remaining
# ambiguous cases — this project's own bundled sample dataset uses
# day/month/2-digit-year ("16/10/24"), and its schema mapper/data already
# skews toward Indian business conventions (city names like Mumbai/Pune
# seen elsewhere in this project), where day-first is the norm. This is a
# documented assumption, not a universal rule: a dataset that is
# genuinely month-first throughout will still parse correctly, because
# whichever format actually matches the most rows wins the vote below —
# the ordering here only breaks ties between formats that would otherwise
# match a similar share of ambiguous rows.
_CANDIDATE_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
    "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y", "%m/%d/%Y", "%m-%d-%y", "%m/%d/%y",
    "%m/%d/%Y %H:%M:%S",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
]

# An explicit format is only trusted if it parses at least this share of
# the column — otherwise a handful of coincidental matches would be a
# worse bet than pandas' own inference, so the deterministic parser backs
# off to inference (see parse_dates_deterministic()) rather than forcing
# a format that barely fits the data.
_MIN_FORMAT_MATCH_SHARE = 0.8


def parse_dates_deterministic(raw_dates: pd.Series) -> pd.Series:
    """
    Task 10 — deterministic date parsing. Tries every format in
    _CANDIDATE_DATE_FORMATS against the full column, keeps whichever
    format parses the largest share of values, and — if that share clears
    _MIN_FORMAT_MATCH_SHARE — parses the ENTIRE column with that one
    format, so every row is interpreted consistently. Falls back to
    pandas' own per-row inference (with the resulting "could not infer
    format" UserWarning suppressed, since it has already been tried
    against explicit alternatives above and lost) only when no candidate
    format clears that bar — an honest fallback for genuinely
    mixed/unusual date columns, rather than forcing a bad fit.

    Never raises: any failure while trying a candidate format is treated
    as "that format doesn't fit" (0 valid rows), not an error, and the
    final pandas-inference fallback uses `errors="coerce"`, so a value
    that truly cannot be parsed becomes NaT rather than stopping the
    whole column from being processed.
    """
    if len(raw_dates) == 0:
        return pd.to_datetime(raw_dates, errors="coerce")

    # Only strings can match an explicit strftime format — a column that
    # pandas already parsed as datetimes (or read as native Timestamp
    # objects) skips straight to inference, which is a no-op in that case.
    raw_str = raw_dates.astype(str)
    n = len(raw_str)

    best_format, best_valid = None, -1
    for fmt in _CANDIDATE_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(raw_str, format=fmt, errors="coerce")
        except Exception:
            continue
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid, best_format = valid, fmt

    if best_format is not None and best_valid >= max(1, int(n * _MIN_FORMAT_MATCH_SHARE)):
        return pd.to_datetime(raw_str, format=best_format, errors="coerce")

    with _warnings_module.catch_warnings():
        _warnings_module.simplefilter("ignore", UserWarning)
        return pd.to_datetime(raw_dates, errors="coerce")

# Implausible date bounds — guards against obvious data-entry errors (a
# typo'd year like 1900, or a placeholder date decades in the future)
# rather than a real transaction date. Wide enough to never reject
# genuine historical or near-future (e.g. pre-scheduled invoice) data.
MIN_PLAUSIBLE_YEAR = 2000
MAX_FUTURE_DAYS = 30

# Negative-revenue rows are kept (refunds/returns are real business
# events that sum correctly into their period) unless they make up more
# than this share of all rows, which points to a data-quality issue
# rather than ordinary refund activity.
NEGATIVE_REVENUE_WARN_THRESHOLD_PCT = 15.0

# Outlier capping (winsorization): a point beyond this many IQRs from the
# quartiles is capped to the boundary, not dropped — every period keeps a
# value (a forecasting model needs one at every period) while an extreme
# spike can no longer dominate a model's fit. Wider than the 1.5x
# "statistical outlier" convention forecast/utils.py's outlier_percentage
# uses for descriptive/audit purposes, because forecasting only needs to
# tame extremes, not flag every mild deviation.
OUTLIER_IQR_MULTIPLIER = 3.0

# A run of this many or fewer consecutive zero-filled periods is treated
# as a probable recording GAP (worth interpolating); longer runs are
# treated as genuine zero-activity (e.g. a seasonal closure) and left as
# zero rather than fabricated.
MAX_INTERPOLATION_GAP = 2


def clean_transactions(dates: pd.Series, values: pd.Series) -> Tuple[pd.Series, pd.Series, List[str], Dict[str, int]]:
    """
    Task 2 — clean raw (date, revenue) pairs BEFORE aggregation. Every
    step here only removes or corrects a row that is already present; it
    never invents a new observation (that only happens in
    interpolate_short_gaps(), after aggregation — see its docstring).

    Returns (clean_dates, clean_values, notes, counts). `notes` are
    business-friendly strings meant to be merged into the pipeline's
    existing warnings list. `counts` gives the raw numbers behind those
    notes (n_start, n_dropped_invalid, n_implausible_dates,
    n_exact_duplicates, n_negative) for anything downstream that wants to
    reason about data quality numerically (see compute_data_quality_score).
    """
    notes: List[str] = []
    dates = parse_dates_deterministic(dates)
    values = _clean_numeric(values)
    n_start = len(dates)

    # 1) Drop rows with an unparseable date or non-numeric revenue —
    #    nothing downstream can use them.
    valid = dates.notna() & values.notna()
    n_dropped_invalid = int((~valid).sum())
    dates, values = dates[valid], values[valid]

    # 2) Drop implausible dates (typo'd years, far-future placeholders).
    n_implausible = 0
    if len(dates) > 0:
        today = pd.Timestamp.now().normalize()
        max_future = today + pd.Timedelta(days=MAX_FUTURE_DAYS)
        min_date = pd.Timestamp(year=MIN_PLAUSIBLE_YEAR, month=1, day=1)
        plausible = (dates >= min_date) & (dates <= max_future)
        n_implausible = int((~plausible).sum())
        dates, values = dates[plausible], values[plausible]

    # 3) Drop EXACT duplicate (date, value) rows — a true duplicate
    #    record (e.g. the same invoice imported twice). Two DIFFERENT
    #    invoices sharing a date is normal and is handled correctly by
    #    summing at aggregation time, not here.
    n_exact_dupes = 0
    if len(dates) > 0:
        dup_mask = pd.DataFrame({"d": dates.values, "v": values.values}).duplicated()
        n_exact_dupes = int(dup_mask.sum())
        dates, values = dates[~dup_mask.values], values[~dup_mask.values]

    # 4) Negative revenue — kept as-is (see module docstring rationale),
    #    only flagged if unusually widespread.
    n_negative = int((values < 0).sum()) if len(values) > 0 else 0

    if n_dropped_invalid > 0:
        notes.append(
            f"{n_dropped_invalid} row(s) had an unparseable date or non-numeric revenue value and were excluded."
        )
    if n_implausible > 0:
        notes.append(
            f"{n_implausible} row(s) had a date outside a plausible range (before {MIN_PLAUSIBLE_YEAR} or "
            f"more than {MAX_FUTURE_DAYS} days in the future) and were excluded."
        )
    if n_exact_dupes > 0:
        notes.append(f"{n_exact_dupes} exact duplicate row(s) (same date and value) were removed.")
    if n_negative > 0 and n_start > 0:
        pct_negative = round(n_negative / n_start * 100, 1)
        if pct_negative > NEGATIVE_REVENUE_WARN_THRESHOLD_PCT:
            notes.append(
                f"{n_negative} row(s) ({pct_negative}%) have negative revenue — unusually high; this may "
                f"indicate a data quality issue rather than ordinary refunds. Negative values were kept "
                f"as-is and summed normally into their period."
            )

    counts = {
        "n_start": n_start,
        "n_dropped_invalid": n_dropped_invalid,
        "n_implausible_dates": n_implausible,
        "n_exact_duplicates": n_exact_dupes,
        "n_negative": n_negative,
    }
    return dates, values, notes, counts


def cap_outliers(series: pd.Series, multiplier: float = OUTLIER_IQR_MULTIPLIER) -> Tuple[pd.Series, List[str], int]:
    """
    Task 11 — winsorize (cap, don't remove) extreme spikes/drops in an
    already-aggregated per-period series, so one abnormal period cannot
    dominate a model's fit while every period keeps a real value (removing
    it entirely would create a fresh missing-timestamp gap). Needs at
    least 4 points for quartiles to be meaningful; shorter series are
    returned unchanged.
    """
    if len(series) < 4:
        return series, [], 0

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return series, [], 0

    lower = max(0.0, q1 - multiplier * iqr)
    upper = q3 + multiplier * iqr
    capped_mask = (series < lower) | (series > upper)
    n_capped = int(capped_mask.sum())
    if n_capped == 0:
        return series, [], 0

    clipped = series.clip(lower=lower, upper=upper)
    note = (
        f"{n_capped} period(s) had a revenue spike or drop far outside this series' normal range and were "
        f"capped to a more typical value before training, so a single unusual period cannot distort the forecast."
    )
    return clipped, [note], n_capped


def interpolate_short_gaps(series: pd.Series, max_gap: int = MAX_INTERPOLATION_GAP) -> Tuple[pd.Series, List[str]]:
    """
    Task 2 — after build_series() has zero-filled every missing period
    into a gapless index, reconsider short RUNS of zero-filled periods
    (length <= max_gap) that sit between real, non-zero activity on both
    sides: these look like a recording gap rather than genuine
    zero-activity, and are linearly interpolated between their neighbours
    instead of being left at zero (which would otherwise read as a real,
    and potentially misleading, demand collapse to any model). Longer
    runs of zero are left alone — a multi-period stretch of zero looks
    much more like real inactivity (e.g. a seasonal closure), and
    fabricating a long synthetic trend across it would be a stronger,
    less justifiable assumption than leaving it as-is.

    Must run AFTER aggregation (on the gapless per-period series), not on
    raw transaction rows — "a gap" is only meaningful once the data has a
    fixed periodicity to have gaps in.
    """
    if len(series) < 3:
        return series, []

    values = series.values.astype(float)
    n = len(values)
    is_zero = values == 0.0
    result = values.copy()
    i, n_runs = 0, 0

    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < n and is_zero[j]:
            j += 1
        run_len = j - i
        has_left = i > 0 and values[i - 1] > 0
        has_right = j < n and values[j] > 0
        if run_len <= max_gap and has_left and has_right:
            left_val, right_val = values[i - 1], values[j]
            for k in range(run_len):
                frac = (k + 1) / (run_len + 1)
                result[i + k] = left_val + frac * (right_val - left_val)
            n_runs += 1
        i = j

    if n_runs == 0:
        return series, []

    note = (
        f"{n_runs} short gap(s) of {max_gap} or fewer periods, surrounded by real activity on both sides, "
        f"were filled by interpolation instead of being left at zero (likely a recording gap rather than "
        f"genuine inactivity)."
    )
    return pd.Series(result, index=series.index), [note]


def compute_data_quality_score(
    counts: Dict[str, int],
    n_capped: int,
    n_series_points: int,
    missing_timestamp_ratio: float,
) -> float:
    """
    A single 0-1 summary of how much correction the raw data needed
    before it could be trusted for forecasting: 1.0 means the data
    arrived essentially clean; lower scores mean more of it had to be
    dropped, capped, or was structurally missing. Consumed by
    selector.py's confidence calculation (Task 8) as ONE of several
    inputs — never the sole determinant of confidence.
    """
    n_start = max(1, counts.get("n_start", 1))
    pct_dropped = (
        counts.get("n_dropped_invalid", 0)
        + counts.get("n_implausible_dates", 0)
        + counts.get("n_exact_duplicates", 0)
    ) / n_start * 100
    pct_capped = (n_capped / max(1, n_series_points)) * 100

    penalty = (
        min(0.4, pct_dropped / 100 * 2)
        + min(0.3, pct_capped / 100 * 3)
        + min(0.3, missing_timestamp_ratio)
    )
    return round(max(0.0, min(1.0, 1 - penalty)), 4)
