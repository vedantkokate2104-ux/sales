"""
segments.py
------------
Group (segment) discovery and resolution for grouped forecasting
(Phase 5).

WHAT ALREADY WORKED
--------------------
`forecast_engine.run_auto_forecast()` has always accepted `group_col` and
`group_value`, and — importantly — it filters BEFORE aggregation:

    working_df = df[df[group_col].astype(str) == str(group_value)]
    ...
    construction, grain_selection, stats = prepare_forecast_series(
        working_df[date_col], working_df[rev_col], ...)

so a group's transactions are aggregated into their own series, its own
grain is selected, its own walk-forward validation runs, and its own
prediction intervals are built. The forecasting pipeline was already
reusable; nothing about it needs duplicating per dimension. What was
missing was everything AROUND it: no way to discover which columns are
worth grouping by, no way to list their values, no history classification,
no API surface, no UI, and two failure modes that were actively
dangerous.

THE TWO BUGS THIS MODULE EXISTS TO CLOSE
-----------------------------------------
1. AN INVALID GROUP COLUMN SILENTLY RETURNED THE COMPANY FORECAST.
   The filter was guarded by `if group_col and group_value is not None
   and group_col in df.columns`. When the column did not exist the guard
   fell through, no filtering happened, and the caller received the
   full-company forecast — which the UI would then label with the group
   the user had asked for. Verified on the bundled dataset: requesting a
   nonexistent column returned a total of 4,009,262, identical to the
   company forecast to the rupee. A user reading "Region — West" would
   have been shown company numbers.

2. AN EMPTY GROUP RETURNED AN EMPTY FORECAST WITH NO EXPLANATION.
   Filtering to a value that matches no rows produced
   `forecast: [], selected_model: None` and no warning — indistinguishable
   from a crash to anything consuming the response.

`resolve_group()` turns both into explicit, typed statuses that the API
and UI can render honestly.

DISCOVERY IS DELIBERATELY LIGHTWEIGHT
--------------------------------------
Nothing here fits a model. Discovery is value counts and date ranges over
already-cleaned data, so listing the segments of a dataset with thousands
of customers costs one pass, not thousands of forecasts. The forecast for
a segment is only ever computed when that specific segment is requested
(Phase 5, Step 17).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics import get_col
from .backtest import MIN_FOLDS, plan_validation
from .utils import period_freq_code, seasonal_period_for_freq

logger = logging.getLogger(__name__)


# ── Group-resolution statuses ──────────────────────────────────────────────
GROUP_OK = "OK"
GROUP_UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
GROUP_UNKNOWN_VALUE = "UNKNOWN_VALUE"
GROUP_EMPTY = "EMPTY_GROUP"
GROUP_NOT_GROUPABLE = "NOT_GROUPABLE"
GROUP_MISSING_VALUE = "MISSING_VALUE"

# ── History classification (Step 7) ────────────────────────────────────────
# These names describe how much VALIDATION a segment can support, and the
# thresholds are derived from Phase 2's own fold planner rather than
# invented here — see classify_history().
SUFFICIENT_HISTORY = "SUFFICIENT_HISTORY"
LIMITED_HISTORY = "LIMITED_HISTORY"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"

# ── Discovery safeguards (Step 3) ──────────────────────────────────────────
# A column needs at least two distinct values to segment anything.
MIN_CARDINALITY = 2

# Above this many distinct values a column is an identifier, not a
# business dimension. Invoice numbers (180 distinct on the bundled
# dataset) are excluded by this; customer codes (30) survive it.
MAX_CARDINALITY = 50

# A column where this share of rows carries its own distinct value is
# behaving like a key even if its raw cardinality is modest.
MAX_UNIQUE_RATIO = 0.5

# Columns missing more than this share of their values cannot segment the
# data reliably.
MAX_MISSING_RATIO = 0.5

# Hard cap on values returned per column, so a response can never balloon.
MAX_VALUES_RETURNED = 200

# Mapped categories that are quantities or timestamps rather than
# dimensions. Grouping revenue BY revenue is meaningless, and grouping by
# a date duplicates the time axis the forecast already has.
_NON_GROUPABLE_CATEGORIES = {"date", "amount", "quantity", "percentage"}

# Suffixes of columns the pipeline itself adds to the frame. data_cleaner
# writes `<column>_outlier_flag` for every numeric column it screens, and
# those flags pass every statistical safeguard below — two distinct
# values, no missing, low cardinality — so without this they surface as
# offered business dimensions. "Forecast revenue grouped by whether
# revenue was flagged an outlier" is not a segment a user should be
# offered, and it is not present in the customer's own CSV.
_INTERNAL_COLUMN_SUFFIXES = ("_outlier_flag",)


@dataclass
class GroupColumnInfo:
    """One candidate grouping dimension."""
    name: str
    label: str
    category: str
    cardinality: int
    n_records: int
    missing_ratio: float
    unique_ratio: float
    usable: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "label":         self.label,
            "category":      self.category,
            "cardinality":   self.cardinality,
            "n_records":     self.n_records,
            "missing_ratio": self.missing_ratio,
            "unique_ratio":  self.unique_ratio,
            "usable":        self.usable,
            "reason":        self.reason,
        }


@dataclass
class GroupValueInfo:
    """One value of a grouping dimension, with enough context for a user
    to tell which segments are worth forecasting before running one."""
    value: str
    n_records: int
    n_periods: int
    first_date: Optional[str]
    last_date: Optional[str]
    history_status: str
    revenue_share: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "value":          self.value,
            "n_records":      self.n_records,
            "n_periods":      self.n_periods,
            "first_date":     self.first_date,
            "last_date":      self.last_date,
            "history_status": self.history_status,
            "revenue_share":  self.revenue_share,
        }


@dataclass
class GroupResolution:
    """Outcome of resolving a (group_col, group_value) request."""
    status: str
    group_col: Optional[str] = None
    group_value: Optional[str] = None
    label: str = "Company Total"
    n_records: int = 0
    message: Optional[str] = None
    frame: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == GROUP_OK

    @property
    def is_company(self) -> bool:
        return self.group_col is None and self.status == GROUP_OK

    def to_dict(self) -> dict:
        return {
            "group_status": self.status,
            "group_col":    self.group_col,
            "group_value":  self.group_value,
            "group_label":  self.label,
            "group_records": self.n_records,
            "group_message": self.message,
        }


# ── History classification ─────────────────────────────────────────────────

def classify_history(n_periods: int, horizon: int = 3, frequency: str = "monthly") -> Tuple[str, str]:
    """
    Classify how much validation a segment of `n_periods` modelled periods
    can support, returning (status, explanation).

    The thresholds are NOT invented here. They are read out of Phase 2's
    own fold planner: a segment is SUFFICIENT when
    `plan_validation()` yields at least MIN_FOLDS folds at the full
    production horizon with no degradation, LIMITED when it yields folds
    but had to shorten the horizon or fell below MIN_FOLDS, and
    INSUFFICIENT when it yields none at all. That keeps the definition of
    "enough history" tied to the one place that actually decides what can
    be validated, so it stays correct if that planner changes.
    """
    plan = plan_validation(
        n_periods, requested_horizon=horizon,
        seasonal_period=seasonal_period_for_freq(frequency),
    )
    if plan.n_folds == 0:
        return INSUFFICIENT_HISTORY, (
            f"{n_periods} {frequency} period(s) cannot support any walk-forward validation fold "
            f"at a {horizon}-period horizon. A forecast can still be produced, but it would carry "
            f"no independent accuracy estimate and no prediction interval."
        )
    if plan.degraded or plan.n_folds < MIN_FOLDS:
        return LIMITED_HISTORY, (
            f"{n_periods} {frequency} period(s) support only {plan.n_folds} validation fold(s) "
            f"at horizon {plan.horizon}"
            + (f" (reduced from {plan.requested_horizon})" if plan.horizon != plan.requested_horizon else "")
            + f". Metrics and intervals for this segment are indicative rather than reliable."
        )
    return SUFFICIENT_HISTORY, (
        f"{n_periods} {frequency} period(s) support {plan.n_folds} validation folds at horizon "
        f"{plan.horizon} — enough for independent metrics and prediction intervals."
    )


# ── Column discovery (Step 3) ──────────────────────────────────────────────

def _category_of(column: str, mappings: Sequence[dict]) -> str:
    for mapping in mappings or []:
        if mapping.get("original_column") == column:
            return str(mapping.get("category", "")).lower()
    return ""


def _label_of(column: str, mappings: Sequence[dict]) -> str:
    for mapping in mappings or []:
        if mapping.get("original_column") == column:
            name = mapping.get("business_name")
            if name:
                return str(name)
    return column


def discover_group_columns(
    df: pd.DataFrame,
    mappings: Sequence[dict],
    max_cardinality: int = MAX_CARDINALITY,
    include_unusable: bool = False,
) -> List[GroupColumnInfo]:
    """
    Find the columns of `df` that are meaningful business dimensions to
    forecast by (Step 3).

    Nothing is hard-coded to product or region. A column qualifies on its
    own statistics:

      - its mapped category is not a date, amount, quantity or percentage
        (those are the measure and the time axis, not dimensions);
      - it has at least MIN_CARDINALITY distinct values, so there is
        something to contrast;
      - it has at most `max_cardinality` distinct values, which is what
        separates a dimension from a key — invoice number has 180 distinct
        values on the bundled dataset and is excluded, customer code has
        30 and is kept;
      - no more than MAX_UNIQUE_RATIO of rows carry their own distinct
        value, catching key-like columns whose raw cardinality happens to
        be modest on a small extract;
      - it is populated in at least half of rows.

    Every rejected column is still returned with `usable=False` and a
    reason when `include_unusable` is set, so the decision is inspectable
    rather than a silent omission.
    """
    results: List[GroupColumnInfo] = []
    n_rows = int(len(df))
    if n_rows == 0:
        return results

    for column in df.columns:
        series = df[column]
        category = _category_of(column, mappings)
        non_null = series.dropna()
        missing_ratio = round(1.0 - (len(non_null) / n_rows), 4) if n_rows else 1.0
        cardinality = int(non_null.astype(str).str.strip().replace("", np.nan).dropna().nunique())
        unique_ratio = round(cardinality / n_rows, 4) if n_rows else 0.0

        reason = None
        if str(column).endswith(_INTERNAL_COLUMN_SUFFIXES):
            reason = "internal column added by the cleaning pipeline, not a field from the source data"
        elif category in _NON_GROUPABLE_CATEGORIES:
            reason = f"'{category}' columns are the measure or the time axis, not a business dimension"
        elif pd.api.types.is_datetime64_any_dtype(series):
            reason = "date column — the forecast already has a time axis"
        elif pd.api.types.is_numeric_dtype(series) and cardinality > max_cardinality:
            reason = f"numeric with {cardinality} distinct values — a measure, not a dimension"
        elif cardinality < MIN_CARDINALITY:
            reason = f"only {cardinality} distinct value(s) — nothing to segment"
        elif cardinality > max_cardinality:
            reason = (
                f"{cardinality} distinct values exceeds the {max_cardinality} limit — this behaves "
                f"like an identifier, and exposing it would invite thousands of segment forecasts"
            )
        elif unique_ratio > MAX_UNIQUE_RATIO:
            reason = (
                f"{round(unique_ratio * 100, 1)}% of rows carry their own distinct value — "
                f"identifier-like"
            )
        elif missing_ratio > MAX_MISSING_RATIO:
            reason = f"{round(missing_ratio * 100, 1)}% of rows are missing this column"

        info = GroupColumnInfo(
            name=column, label=_label_of(column, mappings), category=category,
            cardinality=cardinality, n_records=n_rows, missing_ratio=missing_ratio,
            unique_ratio=unique_ratio, usable=(reason is None), reason=reason,
        )
        if info.usable or include_unusable:
            results.append(info)

    results.sort(key=lambda c: (not c.usable, c.cardinality))
    return results


# ── Value discovery (Step 4) ───────────────────────────────────────────────

def discover_group_values(
    df: pd.DataFrame,
    group_col: str,
    mappings: Sequence[dict],
    frequency: str = "monthly",
    horizon: int = 3,
    limit: int = MAX_VALUES_RETURNED,
) -> List[GroupValueInfo]:
    """
    List the values of `group_col` present in the data, each with the
    context needed to judge it before spending a forecast on it: record
    count, number of periods of history, date span, revenue share, and the
    history classification above.

    Lightweight by construction — value counts and min/max dates over
    already-cleaned data. No model is fit (Step 17).

    Missing values, blanks and whitespace-only entries are dropped rather
    than surfaced as a phantom segment, and everything is normalised to
    string form so it matches the engine's own
    `df[col].astype(str) == str(value)` filter exactly.
    """
    if group_col not in df.columns or len(df) == 0:
        return []

    date_col = get_col(df, mappings, "invoice date")
    rev_col = get_col(df, mappings, "net sales")

    work = df[[group_col]].copy()
    work["_key"] = work[group_col].astype(str).str.strip()
    work = work[(work["_key"] != "") & (work["_key"].str.lower() != "nan")]
    if work.empty:
        return []

    if date_col and date_col in df.columns:
        work["_date"] = pd.to_datetime(df.loc[work.index, date_col], errors="coerce")
    else:
        work["_date"] = pd.NaT
    if rev_col and rev_col in df.columns:
        work["_rev"] = pd.to_numeric(df.loc[work.index, rev_col], errors="coerce").fillna(0.0)
    else:
        work["_rev"] = 0.0

    total_revenue = float(work["_rev"].sum())
    code = period_freq_code(frequency)

    infos: List[GroupValueInfo] = []
    for value, chunk in work.groupby("_key", sort=False):
        dates = chunk["_date"].dropna()
        n_periods = int(dates.dt.to_period(code).nunique()) if len(dates) else 0
        status, _explanation = classify_history(n_periods, horizon=horizon, frequency=frequency)
        revenue = float(chunk["_rev"].sum())
        infos.append(GroupValueInfo(
            value=str(value),
            n_records=int(len(chunk)),
            n_periods=n_periods,
            first_date=str(dates.min().date()) if len(dates) else None,
            last_date=str(dates.max().date()) if len(dates) else None,
            history_status=status,
            revenue_share=round(revenue / total_revenue, 4) if total_revenue else None,
        ))

    # Largest segments first — the ones a user is most likely to want.
    infos.sort(key=lambda v: (-(v.revenue_share or 0.0), -v.n_records, v.value))
    return infos[:limit]


# ── Resolution (the two bug fixes) ─────────────────────────────────────────

def resolve_group(
    df: pd.DataFrame,
    mappings: Sequence[dict],
    group_col: Optional[str] = None,
    group_value: Optional[str] = None,
    max_cardinality: int = MAX_CARDINALITY,
) -> GroupResolution:
    """
    Turn a (group_col, group_value) request into either a filtered frame
    or an explicit, typed failure — never a silent fallback.

    Returns GROUP_OK with `frame=None` and `group_col=None` when no group
    was requested, which is the company-total path and must behave exactly
    as it did before Phase 5.

    Every other outcome carries a status the API and UI can act on:
    UNKNOWN_COLUMN, MISSING_VALUE, NOT_GROUPABLE, UNKNOWN_VALUE, or
    EMPTY_GROUP. The first of those is the bug described in the module
    docstring — previously it produced the company forecast under a group
    label.
    """
    if not group_col:
        return GroupResolution(status=GROUP_OK, label="Company Total", n_records=int(len(df)))

    if group_col not in df.columns:
        available = [c.name for c in discover_group_columns(df, mappings, max_cardinality)]
        return GroupResolution(
            status=GROUP_UNKNOWN_COLUMN, group_col=group_col, group_value=group_value,
            label=str(group_col),
            message=(
                f"'{group_col}' is not a column in this dataset, so no group forecast can be "
                f"produced. Available grouping columns: "
                f"{', '.join(available) if available else 'none in this dataset'}."
            ),
        )

    label_col = _label_of(group_col, mappings)

    if group_value is None or str(group_value).strip() == "":
        return GroupResolution(
            status=GROUP_MISSING_VALUE, group_col=group_col, group_value=group_value,
            label=label_col,
            message=(
                f"A grouping column ('{label_col}') was given without a value to forecast. "
                f"Supply a value, or omit the column entirely for the company total."
            ),
        )

    candidates = {c.name: c for c in discover_group_columns(df, mappings, max_cardinality)}
    if group_col not in candidates:
        rejected = {
            c.name: c for c in discover_group_columns(df, mappings, max_cardinality, include_unusable=True)
        }
        reason = rejected[group_col].reason if group_col in rejected else "not a usable dimension"
        return GroupResolution(
            status=GROUP_NOT_GROUPABLE, group_col=group_col, group_value=str(group_value),
            label=label_col,
            message=f"'{label_col}' cannot be used as a forecast dimension: {reason}.",
        )

    key = df[group_col].astype(str).str.strip()
    target = str(group_value).strip()
    mask = key == target
    n_matched = int(mask.sum())

    if n_matched == 0:
        known = [v.value for v in discover_group_values(df, group_col, mappings, limit=8)]
        return GroupResolution(
            status=GROUP_UNKNOWN_VALUE, group_col=group_col, group_value=target, label=label_col,
            message=(
                f"'{target}' is not a value of '{label_col}' in this dataset. "
                f"Known values include: {', '.join(known) if known else 'none'}."
            ),
        )

    frame = df[mask.to_numpy()].copy()
    if frame.empty:
        return GroupResolution(
            status=GROUP_EMPTY, group_col=group_col, group_value=target, label=label_col,
            message=f"'{label_col} — {target}' matched no usable transactions.",
        )

    return GroupResolution(
        status=GROUP_OK, group_col=group_col, group_value=target,
        label=f"{label_col} — {target}", n_records=n_matched, frame=frame,
    )


# ── Metadata payload (Step 13) ─────────────────────────────────────────────

def group_metadata(
    df: pd.DataFrame,
    mappings: Sequence[dict],
    frequency: str = "monthly",
    horizon: int = 3,
    max_cardinality: int = MAX_CARDINALITY,
    values_per_column: int = MAX_VALUES_RETURNED,
) -> dict:
    """
    Everything the UI needs to offer a Forecast Level selector: the usable
    dimensions and, for each, its values with history context.

    One pass of value counts per candidate column. No forecasting.
    """
    columns = discover_group_columns(df, mappings, max_cardinality)
    payload = []
    for column in columns:
        values = discover_group_values(
            df, column.name, mappings, frequency=frequency,
            horizon=horizon, limit=values_per_column,
        )
        entry = column.to_dict()
        entry["values"] = [v.to_dict() for v in values]
        entry["n_values_returned"] = len(values)
        entry["truncated"] = column.cardinality > len(values)
        payload.append(entry)

    return {
        "group_columns": payload,
        "n_group_columns": len(payload),
        "frequency_basis": frequency,
        "max_cardinality": max_cardinality,
        "history_thresholds": {
            "sufficient": SUFFICIENT_HISTORY,
            "limited": LIMITED_HISTORY,
            "insufficient": INSUFFICIENT_HISTORY,
            "derived_from": "forecast.backtest.plan_validation — the same planner that decides "
                            "how many walk-forward folds a series can support",
        },
    }
