"""
features.py
------------
Frequency-aware, leakage-safe feature engineering with an explicit
availability contract (Phase 3, Steps 2-7).

WHY A NEW MODULE
-----------------
Feature construction previously lived inside models/xgboost.py and was
built for that one model. The Phase 3 audit found two structural problems
that no amount of hyperparameter tuning could fix:

  1. TOO MANY FEATURES FOR THE DATA. 48 features were generated for a
     48-period series. After lag_12 consumed the first year, the full
     series left 36 usable training rows against 47 feature columns
     (p/n = 1.31) — and the FIRST walk-forward fold trained on 12 rows
     with 40 columns (p/n = 3.33). That is the most likely explanation
     for XGBoost scoring 49.41% WAPE against Simple Exponential
     Smoothing's 40.63%.

  2. WINDOWS BORROWED FROM A DAY-ORIENTED LIST. Rolling windows of 7 and
     14 and a lag of 7 are meaningful for daily data. At monthly grain
     they mean "seven months" and "fourteen months", which correspond to
     no business cycle at all.

Redundancy compounded both: nine feature pairs correlated above 0.95,
including one exact duplicate (roll_volatility_3 and short_term_cv,
r = 1.000).

This module builds a SMALL, frequency-appropriate feature set instead,
and makes every feature declare what it is and when it is available.

LEAKAGE MODEL
--------------
Every target-derived feature is computed from `series.shift(1)` before
any window is applied:

    correct    target.shift(1).rolling(3).mean()
    incorrect  target.rolling(3).mean()

so the value being predicted can never appear in its own feature row.
Calendar features are exempt: the calendar position of a future period
is genuinely known in advance, which is the same information SARIMA and
Holt-Winters already use through their seasonal terms.

This shift-first construction is applied to the whole frame at once,
which is both easier to verify than index slicing and identical under
recursive prediction — `build_feature_frame()` is the single entry point
used for training rows and for forecast rows alike.

AVAILABILITY CONTRACT
----------------------
A feature is only usable in a production forecast if its value is
actually knowable at forecast time. The audit of this project's own ERP
export found no usable exogenous driver: the four columns most correlated
with net sales (`rev`, `gross_amt`, `gst_amt`, `trx_v`) are arithmetic
restatements of the target itself, and everything else (quantity, margin,
discount, customer counts) is a business outcome whose future value is
unknown. `classify_business_column()` and `FeatureSpec.availability`
encode that judgement so a future contributor cannot quietly wire one of
them in and produce a spectacular in-sample result with a worthless
forecast.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Availability contract (Steps 6 & 7) ────────────────────────────────────

KNOWN_FUTURE = "KNOWN_FUTURE"          # value is knowable in advance (calendar)
FORECAST_REQUIRED = "FORECAST_REQUIRED"  # would itself need forecasting first
SCENARIO_INPUT = "SCENARIO_INPUT"      # a plan/assumption the user supplies
HISTORICAL_ONLY = "HISTORICAL_ONLY"    # only ever observable after the fact
UNKNOWN = "UNKNOWN"                    # cannot be classified — never usable

# Only these may enter a production forecast. FORECAST_REQUIRED is
# admissible ONLY for lagged transforms of the target itself, which are
# knowable at forecast time because the model carries its own predictions
# forward recursively — that case is marked KNOWN_FUTURE at the point of
# use (see `_target_spec`), not blanket-approved here.
USABLE_AVAILABILITY = {KNOWN_FUTURE, SCENARIO_INPUT}


@dataclass
class FeatureSpec:
    """One feature's contract: what it is, where it comes from, and
    whether it can legitimately be known when the forecast is made."""
    name: str
    kind: str              # "target_lag" | "target_rolling" | "target_trend" | "calendar" | "business_driver"
    availability: str
    aggregation: str       # how it is aggregated to the forecast grain
    leakage_safe: bool
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "feature_name": self.name,
            "feature_type": self.kind,
            "availability": self.availability,
            "aggregation":  self.aggregation,
            "leakage_safe": self.leakage_safe,
            "description":  self.description,
        }


def _target_spec(name: str, kind: str, description: str) -> FeatureSpec:
    """
    A transform of the target's OWN PAST. Marked KNOWN_FUTURE because a
    recursive forecaster genuinely has these values at forecast time: the
    history is real up to the origin, and beyond it the model's own prior
    predictions carry forward — no external information is assumed. The
    leakage guard is the shift(1) applied before any window.
    """
    return FeatureSpec(
        name=name, kind=kind, availability=KNOWN_FUTURE,
        aggregation="target series at the selected forecast grain",
        leakage_safe=True, description=description,
    )


# ── Frequency-aware definitions (Step 2) ───────────────────────────────────
# Lags are counted in PERIODS, never in days. lag_12 on monthly data means
# twelve months; the same number on daily data would mean twelve days,
# which is why these are keyed by frequency rather than shared.

LAGS_BY_FREQUENCY: Dict[str, List[int]] = {
    "monthly":   [1, 2, 3, 6, 12],
    "weekly":    [1, 2, 4, 13, 52],
    "daily":     [1, 7, 14, 28, 365],
    "quarterly": [1, 2, 4],
}

# Rolling windows, also in periods. Deliberately two per frequency: a
# short window for recent level and a longer one covering roughly half a
# seasonal cycle. More windows produced redundant columns (see module
# docstring) without improving out-of-sample error.
ROLLING_WINDOWS_BY_FREQUENCY: Dict[str, List[int]] = {
    "monthly":   [3, 6],
    "weekly":    [4, 13],
    "daily":     [7, 28],
    "quarterly": [2, 4],
}

# Calendar features that carry real variation AFTER aggregation to the
# grain. Day-of-week and day-of-month are omitted for monthly and
# quarterly data because every period would take a near-constant value
# (Step 5) — a column that cannot vary cannot inform a split.
CALENDAR_BY_FREQUENCY: Dict[str, List[str]] = {
    "monthly":   ["year", "quarter", "month"],
    "quarterly": ["year", "quarter"],
    "weekly":    ["year", "quarter", "month", "weekofyear"],
    "daily":     ["year", "quarter", "month", "dayofweek", "dayofmonth"],
}

# A lag is only created when the series still leaves at least this many
# usable training rows after that lag consumes the earliest observations.
# Without this a long lag silently empties the training set.
MIN_TRAIN_ROWS = 8


def resolve_lags(frequency: str, n_observations: int, min_train_rows: int = MIN_TRAIN_ROWS) -> List[int]:
    """
    Lags appropriate to `frequency`, restricted to those the series can
    actually support: lag L needs `n - L >= min_train_rows` rows left over.
    Returns at least [1] whenever the series has two points, so a model is
    never handed an empty feature set.
    """
    candidates = LAGS_BY_FREQUENCY.get(frequency, LAGS_BY_FREQUENCY["monthly"])
    usable = [lag for lag in candidates if n_observations - lag >= min_train_rows]
    if not usable and n_observations >= 2:
        return [1]
    return usable


def resolve_rolling_windows(frequency: str, n_observations: int, min_train_rows: int = MIN_TRAIN_ROWS) -> List[int]:
    """Rolling windows appropriate to `frequency`, same sufficiency rule."""
    candidates = ROLLING_WINDOWS_BY_FREQUENCY.get(frequency, ROLLING_WINDOWS_BY_FREQUENCY["monthly"])
    return [w for w in candidates if n_observations - w >= min_train_rows]


@dataclass
class FeatureConfig:
    """
    Which feature families to build. Every flag defaults to OFF so an
    experiment has to opt in to each family explicitly — that is what
    makes the Step 8 ablations honest rather than a comparison against a
    moving target.
    """
    use_lags: bool = False
    use_rolling: bool = False
    use_trend: bool = False
    use_calendar: bool = False
    min_train_rows: int = MIN_TRAIN_ROWS

    def describe(self) -> str:
        parts = [n for n, on in (
            ("lags", self.use_lags), ("rolling", self.use_rolling),
            ("trend", self.use_trend), ("calendar", self.use_calendar),
        ) if on]
        return "+".join(parts) if parts else "none"


# ── Feature construction ───────────────────────────────────────────────────

def _calendar_frame(index: pd.PeriodIndex, frequency: str) -> Tuple[pd.DataFrame, List[FeatureSpec]]:
    """
    Calendar features for the period BEING predicted. Legitimately known
    in advance, so no shift is applied. Only fields that vary at this
    grain are produced (Step 5).
    """
    stamps = index.to_timestamp()
    wanted = CALENDAR_BY_FREQUENCY.get(frequency, CALENDAR_BY_FREQUENCY["monthly"])
    available = {
        "year":       stamps.year.to_numpy(dtype=float),
        "quarter":    stamps.quarter.to_numpy(dtype=float),
        "month":      stamps.month.to_numpy(dtype=float),
        "weekofyear": np.array([float(ts.isocalendar()[1]) for ts in stamps]),
        "dayofweek":  stamps.dayofweek.to_numpy(dtype=float),
        "dayofmonth": stamps.day.to_numpy(dtype=float),
    }
    data, specs = {}, []
    for name in wanted:
        if name not in available:
            continue
        column = f"cal_{name}"
        data[column] = available[name]
        specs.append(FeatureSpec(
            name=column, kind="calendar", availability=KNOWN_FUTURE,
            aggregation="calendar position of the forecast period",
            leakage_safe=True,
            description=f"{name} of the period being forecast — knowable in advance",
        ))
    return pd.DataFrame(data, index=index), specs


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """OLS slope over a trailing window of an ALREADY-SHIFTED series."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered ** 2))

    def _slope(chunk: np.ndarray) -> float:
        if np.isnan(chunk).any() or denominator == 0:
            return np.nan
        return float(np.dot(x_centered, chunk - chunk.mean()) / denominator)

    return values.rolling(window).apply(_slope, raw=True)


def build_feature_frame(
    series: pd.Series,
    frequency: str,
    config: FeatureConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[FeatureSpec]]:
    """
    Build the supervised learning frame for `series` at `frequency`.

    Returns (X, y, specs) where row *t* of X describes period *t* using
    ONLY information from periods strictly before *t* (target-derived
    features) plus the calendar position of *t* itself. Rows still
    containing NaN — the earliest periods, where the longest lag has no
    value yet — are dropped, and `y` is aligned to the surviving rows.

    Every target-derived family is built from `shifted = series.shift(1)`,
    so the leakage guarantee is a property of the construction rather than
    something a caller has to remember (Step 3).
    """
    index = series.index
    n = len(series)
    shifted = series.shift(1)  # THE leakage guard — applied once, before every window

    frames: List[pd.DataFrame] = []
    specs: List[FeatureSpec] = []

    if config.use_lags:
        lags = resolve_lags(frequency, n, config.min_train_rows)
        data = {f"lag_{lag}": series.shift(lag) for lag in lags}
        if data:
            frames.append(pd.DataFrame(data, index=index))
            specs.extend(
                _target_spec(f"lag_{lag}", "target_lag",
                             f"value {lag} period(s) before the period being forecast")
                for lag in lags
            )

    if config.use_rolling:
        windows = resolve_rolling_windows(frequency, n, config.min_train_rows)
        data = {}
        for w in windows:
            data[f"roll_mean_{w}"] = shifted.rolling(w).mean()
            data[f"roll_std_{w}"] = shifted.rolling(w).std()
            specs.append(_target_spec(f"roll_mean_{w}", "target_rolling",
                                      f"mean of the {w} periods before the forecast period"))
            specs.append(_target_spec(f"roll_std_{w}", "target_rolling",
                                      f"standard deviation of the {w} periods before the forecast period"))
        data["expanding_mean"] = shifted.expanding(min_periods=2).mean()
        data["expanding_std"] = shifted.expanding(min_periods=2).std()
        specs.append(_target_spec("expanding_mean", "target_rolling",
                                  "mean of all periods before the forecast period"))
        specs.append(_target_spec("expanding_std", "target_rolling",
                                  "standard deviation of all periods before the forecast period"))
        if data:
            frames.append(pd.DataFrame(data, index=index))

    if config.use_trend:
        windows = resolve_rolling_windows(frequency, n, config.min_train_rows)
        span = windows[0] if windows else 3
        long_window = windows[-1] if windows else 3
        data = {
            "ema": shifted.ewm(span=span, adjust=False).mean(),
            "momentum": shifted - shifted.shift(1),
            "growth_rate": (shifted / shifted.shift(1).replace(0.0, np.nan)) - 1.0,
            "trend_slope": _rolling_slope(shifted, long_window),
        }
        frames.append(pd.DataFrame(data, index=index))
        specs.extend([
            _target_spec("ema", "target_trend",
                         f"exponentially weighted mean (span {span}) of periods before the forecast period"),
            _target_spec("momentum", "target_trend",
                         "change between the two periods before the forecast period"),
            _target_spec("growth_rate", "target_trend",
                         "relative change between the two periods before the forecast period"),
            _target_spec("trend_slope", "target_trend",
                         f"OLS slope over the {long_window} periods before the forecast period"),
        ])

    if config.use_calendar:
        calendar_frame, calendar_specs = _calendar_frame(index, frequency)
        if not calendar_frame.empty:
            frames.append(calendar_frame)
            specs.extend(calendar_specs)

    if not frames:
        return pd.DataFrame(index=index), series.copy(), []

    X = pd.concat(frames, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    return X.loc[valid], series.loc[valid], specs


# ── Feature selection (Step 12) ────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    correlation_threshold: float = 0.98,
    min_unique: int = 2,
) -> Tuple[pd.DataFrame, List[FeatureSpec], Dict[str, str]]:
    """
    Drop features that cannot carry information on THIS training window:
    constants, near-constants, and one of each near-duplicate pair.

    Applied to a fold's own training matrix, so it is a training-time
    decision like any other and does not consult the validation window.
    Out-of-sample validation remains the final authority on whether a
    surviving feature was worth keeping — this only removes columns that
    are provably uninformative or provably redundant.

    Returns (reduced_X, surviving_specs, {dropped_name: reason}).
    """
    dropped: Dict[str, str] = {}
    if X.empty or X.shape[1] == 0:
        return X, list(specs), dropped

    keep = list(X.columns)

    for column in list(keep):
        if X[column].nunique(dropna=False) < min_unique:
            dropped[column] = "constant on this training window"
            keep.remove(column)

    if len(keep) > 1:
        correlations = X[keep].corr().abs()
        for i, a in enumerate(list(correlations.columns)):
            if a in dropped:
                continue
            for b in list(correlations.columns)[i + 1:]:
                if b in dropped:
                    continue
                value = correlations.loc[a, b]
                if pd.notna(value) and value >= correlation_threshold:
                    dropped[b] = f"correlated {value:.3f} with {a}"
                    keep.remove(b)

    surviving = [s for s in specs if s.name in keep]
    return X[keep], surviving, dropped


# ── Business column classification (Step 6) ────────────────────────────────

# Business names whose values are arithmetic restatements of net sales in
# this project's ERP export. Verified numerically during the Phase 3
# audit: `rev` equals `nsv` in 96.9% of rows, `nsv = gross_amt - disc_val`
# in 93.5%, `gst_amt = 0.18 x nsv` in 93.5%, and `trx_v = nsv + gst_amt`.
# Feeding any of them to a model is target leakage wearing a business
# label, so they are named here rather than left to a correlation filter
# to rediscover.
_TARGET_RESTATEMENTS = {
    "net sales", "revenue", "gross amount", "gst amount",
    "total invoice value", "tax amount",
}

# Business outcomes that are measurable historically but whose FUTURE
# value is not known when the forecast is made.
_OUTCOME_DRIVERS = {
    "quantity sold", "unit price", "margin percentage",
    "customer code", "product id", "customer name",
}

# Levers a business plans in advance. Usable only if the caller supplies
# an explicit future assumption — this dataset contains no such input,
# so in practice they remain unusable here.
_PLANNED_DRIVERS = {"discount percentage", "discount amount"}


def classify_business_column(business_name: str) -> Tuple[str, str]:
    """
    Classify an ERP column by its availability at forecast time
    (Step 6). Returns (availability, reason).

    Deliberately conservative: anything unrecognised is UNKNOWN, which is
    not usable. A column has to be positively identified as safe before
    it can be used, rather than being assumed safe until proven harmful.
    """
    name = (business_name or "").strip().lower()

    if any(marker in name for marker in _TARGET_RESTATEMENTS):
        return HISTORICAL_ONLY, (
            "arithmetic restatement of the forecast target — using it would be leakage, "
            "not a business driver"
        )
    if any(marker in name for marker in _PLANNED_DRIVERS):
        return SCENARIO_INPUT, (
            "a lever the business plans in advance; usable only if an explicit future "
            "assumption is supplied alongside the forecast request"
        )
    if any(marker in name for marker in _OUTCOME_DRIVERS):
        return FORECAST_REQUIRED, (
            "a business outcome measurable only after the fact; its own future value would "
            "have to be forecast first"
        )
    if "date" in name:
        return KNOWN_FUTURE, "calendar position, knowable in advance"
    return UNKNOWN, "not recognised as a forecast-time-available driver"


def audit_business_columns(mappings: Sequence[dict]) -> List[dict]:
    """
    Classify every mapped ERP column and report whether it may be used as
    a forecast feature. Consumed by tests and by the Phase 3 report; the
    forecasting pipeline itself uses no business drivers on this dataset,
    because none of them clear the contract.
    """
    audited = []
    for mapping in mappings or []:
        business_name = mapping.get("business_name", "")
        availability, reason = classify_business_column(business_name)
        audited.append({
            "column":       mapping.get("original_column", ""),
            "business_name": business_name,
            "availability": availability,
            "usable":       availability in USABLE_AVAILABILITY,
            "reason":       reason,
        })
    return audited


def assert_no_unavailable_features(specs: Sequence[FeatureSpec]) -> None:
    """
    Step 7 — the pipeline must refuse features that would not be known at
    forecast time. Raises ValueError naming every offender rather than
    silently dropping them, so a mistake surfaces loudly during
    development instead of quietly degrading a production forecast.
    """
    offenders = [s.name for s in specs if s.availability not in USABLE_AVAILABILITY]
    if offenders:
        raise ValueError(
            "These features are not available at forecast time and must not be used: "
            + ", ".join(sorted(offenders))
        )
