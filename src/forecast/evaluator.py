"""
evaluator.py
------------
Evaluation Engine: pure metric functions shared by every model's
evaluate() method (see forecast/models/base.py) and by forecast.selector
when ranking trained models. Centralising the metric math here means no
model implementation duplicates it.

Scientific-correctness pass: the previous implementation guarded every
percentage-metric denominator with a fixed epsilon (1e-9). That does not
fix division-by-near-zero — it just prevents a crash. Dividing a real
error (e.g. a few hundred thousand rupees) by 1e-9 produces an
astronomical, meaningless ratio (this is exactly the mechanism that
produced MAPE values like 3,217,990,222,222,288% observed against the
bundled sample dataset's weekly data, which legitimately contains
zero-sales periods). The fix here is the textbook-correct treatment, not
a bigger epsilon:

  - MAPE is now computed ONLY over points whose actual value is genuinely
    non-zero (a point with actual == 0 has an undefined, not "very
    large," percentage error), with a per-point cap as a second line of
    defence against a merely tiny-but-nonzero actual still producing an
    outsized single-point contribution. It stays a plain finite float in
    the overwhelming majority of real cases and is documented as
    informational only — selector.py's ranking never uses it as the
    deciding metric.
  - WAPE (Weighted Absolute Percentage Error, aka the MAD/Mean ratio) is
    new: it is a ratio of SUMS (sum of absolute errors / sum of absolute
    actuals), not an average of per-point ratios, so a handful of
    zero/near-zero actual periods cannot make one point dominate or
    explode it the way MAPE can. This is why mature forecasting tooling
    (e.g. Nixtla's StatsForecast, most retail/business forecasting teams)
    treats WAPE, not MAPE, as the primary accuracy metric for intermittent
    or volatile revenue data — see selector.py, where WAPE is now the
    primary ranking metric.
  - SMAPE now treats an exact (actual == 0, predicted == 0) pair as zero
    error rather than running it through an epsilon-guarded division.
  - Every metric drops non-finite (NaN/Inf) (actual, predicted) pairs
    before computing anything, and returns None (JSON null) for a metric
    that is genuinely not computable on the data given (e.g. every actual
    value in the window was zero) rather than silently emitting 0.0 or an
    exploded number. mae/rmse/smape remain reliably computable in every
    case that reaches this function with at least one finite pair; only
    mape/wape can legitimately come back as None.
"""

from typing import Optional

import numpy as np

_EPS = 1e-9

# A point whose |actual| is at or below this is treated as a zero actual
# for MAPE purposes (its percentage error is undefined, not "large").
_ZERO_ACTUAL_THRESHOLD = 1e-6

# Defensive per-point ceiling (in percent) applied before averaging MAPE,
# so a merely tiny-but-nonzero actual can't single-handedly blow up the
# mean the way an unguarded division can. Well above what any reasonably
# performing model would produce, so real skill differences between
# models are unaffected.
_MAPE_POINT_CAP = 500.0


def _finite_pairs(y_true, y_pred):
    """Coerce to float arrays and drop any pair where either side is
    NaN/Inf — a single non-finite value would otherwise poison every
    aggregate (np.mean of anything containing NaN is NaN)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], int(len(y_true)), int(mask.sum())


def mae(y_true, y_pred) -> Optional[float]:
    y_true, y_pred, _, n_valid = _finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return None
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> Optional[float]:
    y_true, y_pred, _, n_valid = _finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return None
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> Optional[float]:
    """
    Weighted Absolute Percentage Error, as a percentage:
        sum(|actual - predicted|) / sum(|actual|) * 100
    A ratio of sums, not an average of per-point ratios — a small number
    of zero/near-zero actuals cannot dominate or explode it the way MAPE
    can. Undefined (returns None) only if every actual in the window is
    ~zero, in which case the denominator itself is ~zero.
    """
    y_true, y_pred, _, n_valid = _finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return None
    denom = float(np.sum(np.abs(y_true)))
    if denom <= _EPS:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def mape(y_true, y_pred) -> Optional[float]:
    """
    Mean Absolute Percentage Error, as a percentage — computed only over
    points with a genuinely non-zero actual (see module docstring).
    Returns None if no point in the window has a usable non-zero actual.
    Informational metric only; never used alone to rank models (see
    selector.py).
    """
    y_true, y_pred, _, n_valid = _finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return None
    nonzero = np.abs(y_true) > _ZERO_ACTUAL_THRESHOLD
    if not np.any(nonzero):
        return None
    pct_errors = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]) * 100
    pct_errors = np.minimum(pct_errors, _MAPE_POINT_CAP)
    return float(np.mean(pct_errors))


def smape(y_true, y_pred) -> Optional[float]:
    """Symmetric MAPE, as a percentage (0-200). A pair where both actual
    and predicted are exactly zero contributes zero error (a correct
    "no sales, forecast no sales" point) instead of running through an
    epsilon-guarded division."""
    y_true, y_pred, _, n_valid = _finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return None
    denom = np.abs(y_true) + np.abs(y_pred)
    both_zero = denom <= _EPS
    terms = np.where(
        both_zero, 0.0,
        np.abs(y_true - y_pred) / (denom + _EPS) * 2,
    )
    return float(np.mean(terms) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute all metrics at once, rounded for display/JSON. `mape`/`wape`
    may be None (JSON null) when genuinely not computable on the data
    given (see each function's docstring) — mae/rmse/smape are reliably
    computable whenever at least one finite (actual, predicted) pair
    exists. n_points/n_valid_points/n_mape_points are additive
    transparency fields describing exactly how many points went into
    each metric.
    """
    y_true_arr, y_pred_arr, n_total, n_valid = _finite_pairs(y_true, y_pred)

    mae_v = mae(y_true, y_pred)
    rmse_v = rmse(y_true, y_pred)
    wape_v = wape(y_true, y_pred)
    mape_v = mape(y_true, y_pred)
    smape_v = smape(y_true, y_pred)

    n_mape_points = 0
    if n_valid > 0:
        n_mape_points = int(np.sum(np.abs(y_true_arr) > _ZERO_ACTUAL_THRESHOLD))

    return {
        "mae":   round(mae_v, 2) if mae_v is not None else None,
        "rmse":  round(rmse_v, 2) if rmse_v is not None else None,
        "mape":  round(mape_v, 2) if mape_v is not None else None,
        "smape": round(smape_v, 2) if smape_v is not None else None,
        "wape":  round(wape_v, 2) if wape_v is not None else None,
        "n_points": n_total,
        "n_valid_points": n_valid,
        "n_mape_points": n_mape_points,
    }
