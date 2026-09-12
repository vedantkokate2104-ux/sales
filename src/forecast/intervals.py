"""
intervals.py
-------------
Prediction intervals from honest out-of-sample forecast errors (Phase 4).

WHAT EXISTED BEFORE
--------------------
Nothing in `src/` produced a prediction interval. `BaseForecastModel`
exposes only `fit()` / `predict()` / `forecast()`, and no model, the
trainer, the selector or the engine ever computed a bound.

The only band the user ever saw was drawn in the browser:

    const bandPct = data.evaluation_metrics
        ? Math.min(0.5, data.evaluation_metrics.mape / 100)
        : 0.15;

— a symmetric ribbon of plus/minus MAPE around the point forecast, capped
at 50%, falling back to a hardcoded plus/minus 15% when no metrics
existed. That is not a prediction interval: it has no nominal coverage
level, it is derived from an average error rather than a quantile, it is
symmetric on a right-skewed quantity, and its coverage was never measured
against anything. A user reading it as "the revenue will land in here"
had no basis for doing so.

METHOD SELECTED: SPLIT-CONFORMAL QUANTILES OF RELATIVE ERRORS
-------------------------------------------------------------
For every walk-forward fold the Phase 2 backtest already retains the
actual and the prediction at each horizon step. Those pairs are genuine
out-of-sample errors produced by a model that never saw the period it was
predicting. This module turns them into intervals:

    r = (actual - predicted) / predicted        one relative error per pair

    lower = point x (1 + Q_low(r))
    upper = point x (1 + Q_high(r))

Chosen over the alternatives for four concrete reasons:

  1. SCALE-FREE. Relative errors transfer across revenue magnitudes, so
     the same code serves a company invoicing thousands and one invoicing
     crores. Raw absolute-error quantiles would bake this dataset's rupee
     scale into the method (Step 11).

  2. NON-NEGATIVITY COMES FOR FREE. Since actuals are non-negative and
     predictions are positive, r >= -1 always, so (1 + Q_low) >= 0 and
     the lower bound cannot go negative. This is a property of the
     construction, not a clip applied afterwards (Step 7) — no
     log transform is needed, and none is introduced.

  3. NATURALLY ASYMMETRIC. Empirical quantiles of a right-skewed error
     distribution give a longer upper tail, which is what revenue data
     actually does. A symmetric plus/minus band cannot represent that.

  4. VALID IN SMALL SAMPLES. The quantile is taken at the SPLIT-CONFORMAL
     order statistic rather than by naive interpolation, which is what
     gives finite-sample coverage guarantees under exchangeability. With
     n errors, a two-sided interval at level (1 - alpha) needs the
     ceil((n+1)(1-alpha/2))-th order statistic to exist — i.e. n >= 39
     for 95%. When it does not exist, this module says so
     (`EXTRAPOLATED`) instead of quietly interpolating and calling the
     result a 95% interval.

Deliberately NOT implemented: parametric residual intervals (assume
normality this data does not have), bootstrapped simulation paths
(heavier, and no better on 33 points), and quantile-regression
forecasters (a new model family, out of scope and forbidden this phase).

LEAKAGE
--------
Errors come exclusively from `BacktestResult.folds`, where every
prediction was produced by a model fit on data strictly preceding its
origin, with preprocessing fitted inside the fold. No actual value from
the forecast horizon, and no full-series residual, enters an interval.

Coverage is measured PREQUENTIALLY (see `evaluate_coverage`): the
interval tested against fold k is built only from folds before k, so the
reported coverage is itself out-of-sample rather than a re-description of
the errors the interval was fitted on.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Interval levels the system produces.
LEVEL_80 = 0.80
LEVEL_95 = 0.95
DEFAULT_LEVELS: Tuple[float, ...] = (LEVEL_80, LEVEL_95)

# Below this many out-of-sample errors, no interval is produced at all.
# Four points cannot describe a distribution's tails, and a bound implied
# by three numbers would be worse than an honest "unavailable".
MIN_ERRORS_FOR_INTERVAL = 5

# Horizon-specific quantiles are only used when each horizon has enough
# errors for the tightest requested level to land on an INTERIOR order
# statistic — see required_errors_interior(). A hard-coded floor here
# (the first version used 8) selects horizon-specific estimation at
# sample sizes where the estimator degenerates: with 11 errors the 80%
# conformal index is exactly (min, max), so the "80%" and "95%" intervals
# come out identical and both equal the full observed error range.
MIN_ERRORS_PER_HORIZON_FLOOR = 8

# At or above this many errors the calibration claim is treated as
# reasonably supported. 40 is where a two-sided 95% conformal interval
# first becomes attainable (n >= 39), so below it the wider level is
# always extrapolated and the whole calibration is reported as
# LIMITED_DATA.
MIN_ERRORS_FOR_CALIBRATED = 40

# A prediction below this fraction of the series mean is treated as too
# close to zero to divide by — including it would produce a meaningless
# relative error of enormous magnitude.
MIN_PREDICTION_FRACTION = 1e-3

# Coverage within this many percentage points of nominal is described as
# on target. Wider than it looks: with ~20 prequential observations the
# standard error on a coverage estimate is around 9pp.
COVERAGE_TOLERANCE_PP = 10.0

# Calibration statuses.
CALIBRATED = "CALIBRATED"
LIMITED_DATA = "LIMITED_DATA"
UNAVAILABLE = "UNAVAILABLE"

# Per-level attainability of the quantile estimate.
#   CONFORMAL       the order statistic exists and is strictly interior —
#                   a genuine quantile with the finite-sample guarantee.
#   CONFORMAL_RANGE the guarantee still holds, but the order statistic is
#                   the observed minimum/maximum, so the bound is the full
#                   observed error range and cannot widen further however
#                   extreme the true tail is.
#   EXTRAPOLATED    the level needs more errors than exist; the empirical
#                   quantile is used and WILL understate the true spread.
CONFORMAL = "conformal"
CONFORMAL_RANGE = "conformal_range"
EXTRAPOLATED = "extrapolated"

# Basis for the error pool.
HORIZON_SPECIFIC = "HORIZON_SPECIFIC"
POOLED = "POOLED"


@dataclass
class IntervalBound:
    """One nominal level's bounds for one forecast period."""
    level: float
    lower: float
    upper: float
    width: float
    status: str          # CONFORMAL | EXTRAPOLATED
    n_errors: int

    def to_dict(self) -> dict:
        return {
            "level":    self.level,
            "lower":    round(self.lower, 2),
            "upper":    round(self.upper, 2),
            "width":    round(self.width, 2),
            "status":   self.status,
            "n_errors": self.n_errors,
        }


@dataclass
class PeriodInterval:
    """All levels for one forecast period, alongside its point forecast."""
    period: str
    horizon: int
    point: float
    bounds: Dict[str, IntervalBound] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {
            "period": self.period,
            "horizon": self.horizon,
            "point_forecast": round(self.point, 2),
        }
        for key, bound in self.bounds.items():
            out[f"lower_{key}"] = round(bound.lower, 2)
            out[f"upper_{key}"] = round(bound.upper, 2)
            out[f"interval_width_{key}"] = round(bound.width, 2)
            out[f"prediction_interval_{key}"] = [round(bound.lower, 2), round(bound.upper, 2)]
        return out


@dataclass
class IntervalReport:
    """Everything Phase 4 produces: the per-period intervals plus the
    evidence for how far they can be trusted."""
    available: bool
    method: str
    calibration_status: str
    basis: str
    periods: List[PeriodInterval] = field(default_factory=list)
    n_errors_total: int = 0
    n_errors_by_horizon: Dict[int, int] = field(default_factory=dict)
    coverage: Dict[str, dict] = field(default_factory=dict)
    sharpness: Dict[str, dict] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available":            self.available,
            "interval_method":      self.method,
            "interval_calibration_status": self.calibration_status,
            "interval_basis":       self.basis,
            "periods":              [p.to_dict() for p in self.periods],
            "n_errors_total":       self.n_errors_total,
            "n_errors_by_horizon":  {str(k): v for k, v in self.n_errors_by_horizon.items()},
            "coverage":             self.coverage,
            "sharpness":            self.sharpness,
            "notes":                self.notes,
            "unavailable_reason":   self.unavailable_reason,
        }


# ── Error collection ───────────────────────────────────────────────────────

def collect_relative_errors(folds: Sequence, series_mean: Optional[float] = None) -> Dict[int, List[float]]:
    """
    Extract out-of-sample relative errors from backtest folds, keyed by
    HORIZON STEP (1-based position within the fold's forecast window).

    Each `FoldOutcome` carries aligned `actuals` and `predictions` for the
    periods it scored, in chronological order, so position i within a fold
    is the (i+1)-step-ahead forecast from that fold's origin. Grouping by
    that position is what allows horizon-specific uncertainty (Step 5) —
    a 3-step forecast is genuinely less certain than a 1-step one, and
    pooling them would understate the far horizon and overstate the near.

    Predictions too close to zero are skipped: dividing by them yields a
    relative error of arbitrary magnitude that describes the denominator,
    not the model.
    """
    by_horizon: Dict[int, List[float]] = {}
    floor = None
    if series_mean is not None and np.isfinite(series_mean) and series_mean > 0:
        floor = MIN_PREDICTION_FRACTION * series_mean

    for fold in folds or []:
        actuals = getattr(fold, "actuals", []) or []
        predictions = getattr(fold, "predictions", []) or []
        for position, (actual, predicted) in enumerate(zip(actuals, predictions), start=1):
            actual, predicted = float(actual), float(predicted)
            if not np.isfinite(actual) or not np.isfinite(predicted):
                continue
            if predicted <= 0 or (floor is not None and predicted < floor):
                continue
            by_horizon.setdefault(position, []).append((actual - predicted) / predicted)

    return by_horizon


def conformal_indices(n: int, level: float) -> Tuple[Optional[int], Optional[int]]:
    """
    Order-statistic positions (0-based) for a two-sided split-conformal
    interval at `level` from `n` exchangeable errors.

    The upper bound uses the ceil((n+1)(1 - alpha/2))-th smallest error
    and the lower the floor((n+1)(alpha/2))-th. When the upper index
    exceeds n the level is NOT attainable from this many errors — 95%
    two-sided needs n >= 39 — and (None, None) is returned so the caller
    can fall back explicitly and label the result EXTRAPOLATED rather
    than pretending the guarantee holds.
    """
    if n < 1:
        return None, None
    alpha = 1.0 - level
    # Nudge both ranks by a tolerance before rounding. `1.0 - 0.80` is
    # 0.19999999999999996 in binary floating point, so at n = 9 the lower
    # rank evaluates to floor(0.9999999999999998) = 0 rather than the
    # exact floor(1.0) = 1, and a level that IS attainable gets rejected.
    # The error is only ~1e-16, far smaller than the gap between adjacent
    # integer ranks, so absorbing it cannot change a genuinely different
    # result — it only stops representation noise from silently demanding
    # one more error than the mathematics requires.
    tolerance = 1e-9
    upper_rank = int(np.ceil((n + 1) * (1.0 - alpha / 2.0) - tolerance))
    lower_rank = int(np.floor((n + 1) * (alpha / 2.0) + tolerance))
    if upper_rank > n or lower_rank < 1:
        return None, None
    return lower_rank - 1, upper_rank - 1


def required_errors_for_level(level: float, interior: bool = False) -> int:
    """
    Smallest error count at which `level` is estimable.

    With `interior=False`: the smallest n for which the split-conformal
    order statistic exists at all (9 for 80%, 39 for 95%).

    With `interior=True`: the smallest n for which that order statistic is
    strictly inside the sample rather than its minimum and maximum. This
    is the threshold that matters for choosing horizon-specific over
    pooled estimation — at n below it the "quantile" is just the observed
    range, and every requested level collapses onto the same bounds.
    """
    for n in range(1, 4000):
        low, high = conformal_indices(n, level)
        if low is None or high is None:
            continue
        if not interior:
            return n
        if low > 0 and high < n - 1:
            return n
    return 4000


def error_quantiles(errors: Sequence[float], level: float) -> Tuple[float, float, str]:
    """
    Lower and upper relative-error quantiles at `level`, plus the status
    describing how they were obtained (see the status constants above).
    """
    ordered = np.sort(np.asarray(list(errors), dtype=float))
    n = len(ordered)
    if n == 0:
        return 0.0, 0.0, EXTRAPOLATED

    low_index, high_index = conformal_indices(n, level)
    if low_index is not None and high_index is not None:
        status = CONFORMAL if (low_index > 0 and high_index < n - 1) else CONFORMAL_RANGE
        return float(ordered[low_index]), float(ordered[high_index]), status

    alpha = 1.0 - level
    low = float(np.quantile(ordered, alpha / 2.0))
    high = float(np.quantile(ordered, 1.0 - alpha / 2.0))
    return low, high, EXTRAPOLATED


# ── Interval construction ──────────────────────────────────────────────────

def _errors_for(by_horizon: Dict[int, List[float]], horizon: int, basis: str) -> List[float]:
    if basis == HORIZON_SPECIFIC and horizon in by_horizon:
        return by_horizon[horizon]
    pooled: List[float] = []
    for values in by_horizon.values():
        pooled.extend(values)
    return pooled


def choose_basis(
    by_horizon: Dict[int, List[float]],
    horizons_needed: Sequence[int],
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> Tuple[str, str]:
    """
    Decide between horizon-specific and pooled error distributions
    (Step 5), returning (basis, explanation).

    Horizon-specific estimation genuinely is the more informative choice —
    a 3-step forecast is less certain than a 1-step one, and pooling
    understates the far horizon while overstating the near. But it is only
    used when each horizon holds enough of its own errors for the TIGHTEST
    requested level to be a real interior quantile rather than the
    observed min/max. Splitting a small error pool three ways to estimate
    tails from ~11 points each does not produce three distributions; it
    produces three copies of the observed range, which is exactly the
    fabricated per-horizon structure Step 5 forbids.

    Pooling is therefore the documented fallback, and on a series of this
    length it is what actually gets used.
    """
    if not by_horizon:
        return POOLED, "no out-of-sample errors were available"

    tightest = min(levels) if levels else LEVEL_80
    needed = max(MIN_ERRORS_PER_HORIZON_FLOOR, required_errors_for_level(tightest, interior=True))
    counts = {h: len(by_horizon.get(h, [])) for h in horizons_needed}
    total = sum(len(v) for v in by_horizon.values())

    if counts and all(c >= needed for c in counts.values()):
        detail = ", ".join(f"h{h}: {c}" for h, c in sorted(counts.items()))
        return HORIZON_SPECIFIC, (
            f"each forecast horizon has at least {needed} of its own out-of-sample errors "
            f"({detail}) — enough for the {int(tightest * 100)}% quantile to be a genuine interior "
            f"order statistic — so uncertainty is estimated separately per horizon"
        )

    thin = ", ".join(f"h{h}: {c}" for h, c in sorted(counts.items()) if c < needed)
    return POOLED, (
        f"a {int(tightest * 100)}% interval needs at least {needed} errors per horizon to be an "
        f"interior quantile rather than the observed min/max, and this series has fewer ({thin}), "
        f"so all {total} errors are pooled across horizons. Pooling understates uncertainty at the "
        f"far horizon and overstates it at the near one, but it is the honest choice at this "
        f"sample size"
    )


def compute_level_bounds(
    point: float,
    errors_by_level: Dict[str, Sequence[float]],
    levels: Sequence[float] = DEFAULT_LEVELS,
    non_negative: bool = True,
) -> Dict[str, IntervalBound]:
    """
    THE single implementation of "given a point forecast and its error
    pool, what are the bounds" — shared by `build_intervals()` (what gets
    published) and `evaluate_coverage()` (what gets measured).

    Factoring this out is not tidying. Coverage evaluation originally
    computed its own bounds and skipped the nesting guard, so it measured
    intervals that were never published: at small samples the 80% level
    lands on the conformal min/max while the 95% falls back to an
    INTERPOLATED empirical quantile sitting strictly inside that range,
    making the "95%" interval narrower than the "80%". Coverage then came
    back lower for the wider nominal level — 100% at 80% and 77.8% at 95%
    on a quarterly series — which is impossible for nested intervals and
    made the whole coverage table untrustworthy. One code path removes
    the possibility of the two disagreeing again.
    """
    bounds: Dict[str, IntervalBound] = {}
    for level in levels:
        key = f"{int(round(level * 100))}"
        errors = list(errors_by_level.get(key, []))
        low_q, high_q, status = error_quantiles(errors, level)
        lower = point * (1.0 + low_q)
        upper = point * (1.0 + high_q)

        if lower > upper:
            lower, upper = upper, lower
        if non_negative:
            # Structurally redundant (r >= -1 implies lower >= 0) but kept
            # as an explicit assertion of the revenue domain constraint.
            lower = max(0.0, lower)
        # A point forecast must always lie inside its own interval.
        lower = min(lower, point)
        upper = max(upper, point)

        bounds[key] = IntervalBound(
            level=level, lower=lower, upper=upper, width=upper - lower,
            status=status, n_errors=len(errors),
        )

    _enforce_nesting_bounds(bounds, levels)
    return bounds


def _enforce_nesting_bounds(bounds: Dict[str, IntervalBound], levels: Sequence[float]) -> None:
    """Ensure higher nominal levels contain lower ones. Violations arise
    only when levels fall back to different estimators (conformal
    order statistic vs interpolated empirical quantile), so the WIDER
    level is widened to contain the narrower — never the reverse, which
    would silently shrink a level below its own quantile."""
    ordered = sorted(levels)
    for tighter, wider in zip(ordered, ordered[1:]):
        a = bounds.get(f"{int(round(tighter * 100))}")
        b = bounds.get(f"{int(round(wider * 100))}")
        if not a or not b:
            continue
        b.lower = min(b.lower, a.lower)
        b.upper = max(b.upper, a.upper)
        b.width = b.upper - b.lower


def build_intervals(
    forecast_points: Sequence[dict],
    by_horizon: Dict[int, List[float]],
    levels: Sequence[float] = DEFAULT_LEVELS,
    non_negative: bool = True,
) -> Tuple[List[PeriodInterval], str, str]:
    """
    Attach intervals to each forecast period. Returns
    (periods, basis, basis_explanation).
    """
    horizons = list(range(1, len(forecast_points) + 1))
    basis, explanation = choose_basis(by_horizon, horizons, levels)

    periods: List[PeriodInterval] = []
    for horizon, point in zip(horizons, forecast_points):
        value = float(point.get("value", 0.0))
        interval = PeriodInterval(
            period=str(point.get("period", "")), horizon=horizon, point=value,
        )
        errors = _errors_for(by_horizon, horizon, basis)
        errors_by_level = {f"{int(round(l * 100))}": errors for l in levels}
        interval.bounds = compute_level_bounds(
            value, errors_by_level, levels=levels, non_negative=non_negative,
        )
        periods.append(interval)

    return periods, basis, explanation


# ── Coverage and sharpness (Steps 8 & 9) ───────────────────────────────────

def evaluate_coverage(
    folds: Sequence,
    levels: Sequence[float] = DEFAULT_LEVELS,
    min_training_folds: int = 4,
    series_mean: Optional[float] = None,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Measure how often the intervals actually contained the truth, and how
    wide they had to be to do it (Steps 8 & 9).

    PREQUENTIAL, not in-sample: the interval evaluated against fold k is
    built ONLY from the errors of folds strictly before k. That mirrors
    how the interval would have been used in production and keeps the
    coverage figure genuinely out-of-sample — computing quantiles from all
    folds and then checking those same folds would measure how well a
    quantile describes the sample it came from, which is not a
    calibration result.

    Returns (coverage_by_level, sharpness_by_level).
    """
    ordered_folds = sorted(folds or [], key=lambda f: getattr(f, "fold", 0))
    hits: Dict[str, List[bool]] = {f"{int(round(l * 100))}": [] for l in levels}
    widths: Dict[str, List[float]] = {f"{int(round(l * 100))}": [] for l in levels}
    normalised: Dict[str, List[float]] = {f"{int(round(l * 100))}": [] for l in levels}
    scores: Dict[str, List[float]] = {f"{int(round(l * 100))}": [] for l in levels}

    for position, fold in enumerate(ordered_folds):
        if position < min_training_folds:
            continue
        history = ordered_folds[:position]
        by_horizon = collect_relative_errors(history, series_mean=series_mean)
        total_errors = sum(len(v) for v in by_horizon.values())
        if total_errors < MIN_ERRORS_FOR_INTERVAL:
            continue

        actuals = getattr(fold, "actuals", []) or []
        predictions = getattr(fold, "predictions", []) or []
        horizons = list(range(1, len(actuals) + 1))
        basis, _ = choose_basis(by_horizon, horizons, levels)

        for horizon, (actual, predicted) in enumerate(zip(actuals, predictions), start=1):
            actual, predicted = float(actual), float(predicted)
            if not np.isfinite(actual) or not np.isfinite(predicted) or predicted <= 0:
                continue
            errors = _errors_for(by_horizon, horizon, basis)
            if len(errors) < MIN_ERRORS_FOR_INTERVAL:
                continue

            # Identical bound computation to the published intervals —
            # including the nesting guard — so coverage measures the thing
            # the user actually receives. See compute_level_bounds().
            errors_by_level = {f"{int(round(l * 100))}": errors for l in levels}
            bounds = compute_level_bounds(predicted, errors_by_level, levels=levels)

            for level in levels:
                key = f"{int(round(level * 100))}"
                bound = bounds[key]
                hits[key].append(bool(bound.lower <= actual <= bound.upper))
                widths[key].append(bound.width)
                if predicted > 0:
                    normalised[key].append(bound.width / predicted)
                scores[key].append(_interval_score(actual, bound.lower, bound.upper, level))

    coverage: Dict[str, dict] = {}
    sharpness: Dict[str, dict] = {}
    for level in levels:
        key = f"{int(round(level * 100))}"
        observations = hits[key]
        nominal_pct = round(level * 100, 1)
        if not observations:
            coverage[key] = {
                "nominal": nominal_pct, "observed": None, "coverage_error_pp": None,
                "n": 0, "note": "not enough validation folds to measure coverage out-of-sample",
            }
            sharpness[key] = {
                "mean_width": None, "median_width": None,
                "mean_normalized_width": None, "interval_score": None, "n": 0,
            }
            continue

        observed_pct = round(100.0 * sum(observations) / len(observations), 1)
        coverage[key] = {
            "nominal":           nominal_pct,
            "observed":          observed_pct,
            "coverage_error_pp": round(observed_pct - nominal_pct, 1),
            "n":                 len(observations),
            "note":              _coverage_note(observed_pct, nominal_pct, len(observations)),
        }
        sharpness[key] = {
            "mean_width":            round(float(np.mean(widths[key])), 2),
            "median_width":          round(float(np.median(widths[key])), 2),
            "mean_normalized_width": round(float(np.mean(normalised[key])), 4) if normalised[key] else None,
            "interval_score":        round(float(np.mean(scores[key])), 2) if scores[key] else None,
            "n":                     len(widths[key]),
        }

    return coverage, sharpness


def _interval_score(actual: float, lower: float, upper: float, level: float) -> float:
    """
    Winkler interval score — a proper scoring rule for interval forecasts
    (Step 9). Charges the interval's width, plus a penalty proportional to
    how far outside the actual fell, scaled by 2/alpha. Lower is better.

    This is what stops "make it very wide" from looking like success: an
    interval that achieves coverage purely through width pays for that
    width on every single observation.
    """
    alpha = 1.0 - level
    score = upper - lower
    if actual < lower:
        score += (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        score += (2.0 / alpha) * (actual - upper)
    return score


def _coverage_note(observed: float, nominal: float, n: int) -> str:
    # Binomial standard error of the coverage estimate, in percentage
    # points — the honest way to say "this number is itself uncertain".
    p = observed / 100.0
    standard_error_pp = 100.0 * float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))
    gap = observed - nominal
    verdict = "close to nominal" if abs(gap) <= COVERAGE_TOLERANCE_PP else (
        "wider than nominal" if gap > 0 else "narrower than nominal"
    )
    return (
        f"{verdict}; measured on {n} out-of-sample point(s), "
        f"standard error about {standard_error_pp:.1f}pp"
    )


# ── Top-level entry point ──────────────────────────────────────────────────

def build_interval_report(
    forecast_points: Sequence[dict],
    backtest,
    levels: Sequence[float] = DEFAULT_LEVELS,
    series_mean: Optional[float] = None,
    non_negative: bool = True,
) -> IntervalReport:
    """
    Produce the full Phase 4 result for a set of point forecasts.

    `backtest` is the winning model's `BacktestResult` from Phase 2 (or
    the evaluated ensemble); anything exposing `.folds` works. When there
    is no usable error history the report comes back `available=False`
    with a stated reason, so the caller shows an honest "unavailable"
    message rather than a fabricated band.
    """
    method = "split-conformal quantiles of out-of-sample relative errors"

    if not forecast_points:
        return IntervalReport(
            available=False, method=method, calibration_status=UNAVAILABLE, basis=POOLED,
            unavailable_reason="No point forecast was produced, so there is nothing to bound.",
        )

    folds = list(getattr(backtest, "folds", []) or []) if backtest is not None else []
    by_horizon = collect_relative_errors(folds, series_mean=series_mean)
    total_errors = sum(len(v) for v in by_horizon.values())

    if total_errors < MIN_ERRORS_FOR_INTERVAL:
        return IntervalReport(
            available=False, method=method, calibration_status=UNAVAILABLE, basis=POOLED,
            n_errors_total=total_errors,
            n_errors_by_horizon={h: len(v) for h, v in by_horizon.items()},
            unavailable_reason=(
                f"Prediction interval unavailable — only {total_errors} out-of-sample forecast "
                f"error(s) exist, and at least {MIN_ERRORS_FOR_INTERVAL} are needed before any "
                f"interval can be estimated. More history is required."
            ),
        )

    periods, basis, explanation = build_intervals(
        forecast_points, by_horizon, levels=levels, non_negative=non_negative,
    )
    coverage, sharpness = evaluate_coverage(folds, levels=levels, series_mean=series_mean)

    status = CALIBRATED if total_errors >= MIN_ERRORS_FOR_CALIBRATED else LIMITED_DATA

    notes: List[str] = [f"Error basis: {explanation}."]
    if status == LIMITED_DATA:
        notes.append(
            f"Calibration is based on {total_errors} out-of-sample error(s), below the "
            f"{MIN_ERRORS_FOR_CALIBRATED} at which a two-sided 95% conformal interval first "
            f"becomes attainable. Treat the bounds as indicative rather than guaranteed."
        )
    extrapolated = sorted({
        f"{int(round(b.level * 100))}%"
        for p in periods for b in p.bounds.values() if b.status == EXTRAPOLATED
    })
    range_bound = sorted({
        f"{int(round(b.level * 100))}%"
        for p in periods for b in p.bounds.values() if b.status == CONFORMAL_RANGE
    })
    if extrapolated:
        notes.append(
            f"The {', '.join(extrapolated)} level(s) exceed what {total_errors} errors can support "
            f"as a conformal guarantee, so their bounds are extrapolated from the observed error "
            f"range and will understate the true uncertainty."
        )
    if range_bound:
        notes.append(
            f"The {', '.join(range_bound)} level(s) are bounded by the full observed error range "
            f"(the conformal order statistic is the sample minimum/maximum), so they cannot widen "
            f"further no matter how extreme the true tail is."
        )
    for key, entry in coverage.items():
        if entry.get("observed") is None:
            notes.append(
                f"Coverage at the {key}% level could not be measured out-of-sample — "
                f"too few validation folds to build an interval from earlier folds alone."
            )

    return IntervalReport(
        available=True, method=method, calibration_status=status, basis=basis,
        periods=periods, n_errors_total=total_errors,
        n_errors_by_horizon={h: len(v) for h, v in sorted(by_horizon.items())},
        coverage=coverage, sharpness=sharpness, notes=notes,
    )
