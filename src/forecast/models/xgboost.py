"""
xgboost.py
----------
XGBoost forecasting: turns the series into a supervised-learning table
(lag features, rolling statistics, EMA, trend/momentum, cyclical calendar
encoding, and a small set of interaction terms) and trains
xgboost.XGBRegressor on it. Recommended by Phase 1's model_selector.py for
"Highly Volatile" series, where a gradient-boosted tree can capture
non-linear structure a classical linear method cannot.

There is no native multi-step API for a tabular regressor, so forecasting
beyond one step uses RECURSIVE prediction (the standard approach — see
e.g. sktime's `make_reduction`/Darts' `RegressionModel`): predict one step
ahead, append that prediction to the working history, recompute features
from the extended history, predict the next step, and so on. Every
feature at every step (training AND recursive prediction) is built from
the SAME single entry point, `_build_feature_row()`, so there is exactly
one implementation of "what do the features look like at position t" —
this is a deliberate leakage guard: a training row for position t only
ever sees values strictly BEFORE t (`history[:t]`), and a prediction step
for a future position only ever sees real history plus the model's own
prior predictions, never a real future value.

Forecast-quality hardening pass:
  - The model is now trained on a RESIDUAL target — the raw value minus
    a naive+trend baseline (last observed value plus the current OLS
    trend slope) — rather than the raw level. This is the single most
    consequential change in this pass: a gradient-boosted tree cannot
    predict outside the range of target values seen during training (a
    leaf's output is an average of the training points that landed in
    it), so on a genuinely trending series a tree trained on the raw
    level systematically underpredicts once the series moves past
    whatever maximum it reached during training — a structural
    limitation of tree ensembles that no amount of feature engineering
    or hyperparameter tuning can fix, and, empirically, the most likely
    explanation for XGBoost sometimes losing to Naive/SES on trending
    business data. Modelling the residual keeps the training target
    close to stationary; the baseline (never a model input feature
    itself — see fit()/predict()) carries the trend forward, and is
    added back to reconstruct every real prediction.
  - Feature engineering widened again: an exponential moving average
    (EMA) per rolling window (recency-weighted, unlike the plain rolling
    mean), an OLS trend slope over the longest available window
    (a smooth, multi-point trend signal distinct from the existing
    single-lag momentum ratio), a scale-free short-term volatility
    figure, cyclical sin/cos encodings of every calendar feature
    (removes the artificial "December is far from January" discontinuity
    a raw integer encoding implies), and a small, deliberately bounded
    set of interaction terms (trend/momentum crossed with volatility and
    with seasonal position).
  - Target is now trained in log1p-space and inverse-transformed via
    expm1() whenever the series is entirely non-negative (falls back to
    raw-scale training otherwise) — compresses the right-skewed,
    heteroscedastic scale typical of business revenue data so squared-
    error loss is not dominated by the largest-scale periods alone.
  - suggest_hyperparameters() now tries Optuna's Bayesian (TPE) search
    first, over a wider space (adds subsample/colsample_bytree/
    min_child_weight/reg_alpha/reg_lambda to the existing max_depth/
    learning_rate search), and transparently falls back to an expanded
    manual grid search — unchanged in spirit, just a richer set of joint
    configurations — if Optuna is not installed or the search fails for
    any reason. Optuna remains a fully optional dependency; nothing here
    requires installing it.
  - Recursive multi-step prediction is now soft-bounded: each step's
    prediction is clipped to a plausible range derived from the series'
    own recent level and volatility, widening as the horizon grows (like
    a prediction interval widening with lead time) — this does not
    eliminate the inherent error-accumulation of recursive forecasting
    (a fundamentally different, direct multi-horizon approach would be
    needed for that, and would require a separate model per horizon
    step — out of scope for an implementation-only change), but it
    prevents an early error from compounding into an implausible runaway
    trajectory many steps out, which matters more for volatile business
    data than for smooth series.
  - Richer lag candidates (adds lag 7 and a half-seasonal lag) and
    rolling windows that also scale up for longer series (adds a longer
    window, and the seasonal period itself as a window when there is
    enough history), so short series still get a small, low-overfitting-
    risk feature set while longer series get more structure to learn
    from.
  - A conservative default `min_child_weight` and a wider regularisation
    search space address the "could be more robust... reduce overfitting"
    feedback directly, on top of the existing subsample/colsample_bytree/
    reg_lambda/reg_alpha regularisation and the automatic feature-
    selection refit (unchanged, see fit()).
  - Internals reorganised into small, single-purpose helper functions
    (lag / rolling / EMA / trend-momentum / calendar / interactions) that
    the single `_build_feature_row()` entry point composes, instead of
    one large function — the leakage-guard property ("training and
    recursive prediction always call the same code") is preserved
    exactly, just implemented more legibly.

Everything below is a full backward-compatible drop-in: the class name,
`is_available()`, `check_requirements()`, `suggest_hyperparameters()`,
`fit()`, and `predict()` all keep their existing names, signatures, and
return types, so trainer.py/selector.py/forecast_engine.py continue to
interact with XGBoostModel exactly as before. Depends on the optional
`xgboost` package; degrades gracefully via is_available() if it is not
installed, exactly like every other optional model in this library
(Holt-Winters/SARIMA -> statsmodels, Prophet -> prophet).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import seasonal_period_for_freq

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _try_import_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        return None


def _try_import_holidays():
    try:
        import holidays
        return holidays
    except ImportError:
        return None


def _try_import_optuna():
    """
    Optuna is an OPTIONAL dependency used only to make
    suggest_hyperparameters() search a wider space more efficiently than
    a manual grid. If it is not installed (or fails to import/initialise
    for any reason), suggest_hyperparameters() transparently falls back
    to the manual grid search below — the project continues to work with
    zero new hard dependencies, exactly like every other optional
    integration in this library (xgboost/prophet/statsmodels/holidays all
    degrade the same way).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except Exception:
        return None


# Cache of one country's holiday calendar object per process — country
# holiday calendars are static reference data, expensive to rebuild, and
# safe to reuse across every model instance/fit call.
_HOLIDAY_CALENDAR_CACHE: Dict[str, object] = {}


def _india_holiday_calendar():
    """
    Best-effort holiday calendar lookup — this project's recognised
    columns (region, delivery city names like Mumbai/Pune, currency)
    indicate India-based business data, so "IN" is used as a reasonable
    default rather than something configurable per dataset (there is no
    country field in the schema mapper's output to read one from). If the
    optional `holidays` package is not installed, or the lookup fails for
    any reason, this simply returns None and the holiday feature is
    omitted entirely — never fabricated.
    """
    if "IN" in _HOLIDAY_CALENDAR_CACHE:
        return _HOLIDAY_CALENDAR_CACHE["IN"]
    holidays_mod = _try_import_holidays()
    calendar = None
    if holidays_mod is not None:
        try:
            calendar = holidays_mod.country_holidays("IN")
        except Exception:
            calendar = None
    _HOLIDAY_CALENDAR_CACHE["IN"] = calendar
    return calendar


def _holiday_feature(period, frequency: str) -> Optional[float]:
    """1.0/0.0 holiday-period indicator, or None if not applicable/available
    (see module docstring). Only meaningful for daily (does this exact day
    fall on a holiday?) and weekly (does any day in this week?) data —
    monthly-or-coarser periods can't meaningfully align to a single
    calendar holiday."""
    if frequency not in ("daily", "weekly"):
        return None
    calendar = _india_holiday_calendar()
    if calendar is None:
        return None
    ts = period.to_timestamp()
    try:
        if frequency == "daily":
            return 1.0 if ts.date() in calendar else 0.0
        return 1.0 if any((ts + pd.Timedelta(days=d)).date() in calendar for d in range(7)) else 0.0
    except Exception:
        return None


# ── feature engineering: lag / rolling / EMA / trend / momentum ────────────

def _lag_features(history: List[float], lags: List[int]) -> Dict[str, float]:
    """Value exactly `lag` steps back. NaN if not enough history yet — rows
    with any NaN lag are dropped before training (see
    XGBoostModel._build_valid_xy), so a short lag-history never silently
    trains on a fabricated value."""
    n = len(history)
    return {f"lag_{lag}": (history[-lag] if n >= lag else np.nan) for lag in lags}


def _rolling_window_features(history: List[float], roll_windows: List[int]) -> Dict[str, float]:
    """Rolling mean/std/median/min/max plus a coefficient-of-variation
    ("volatility") over the last `w` points (or all available points if
    fewer than `w` exist yet, so early rows still get a usable, if
    noisier, estimate rather than NaN)."""
    feats: Dict[str, float] = {}
    n = len(history)
    for w in roll_windows:
        window = history[-w:] if n >= w else history[:]
        if window:
            arr = np.asarray(window, dtype=float)
            mean_w = float(np.mean(arr))
            std_w = float(np.std(arr)) if len(arr) > 1 else 0.0
            feats[f"roll_mean_{w}"] = mean_w
            feats[f"roll_std_{w}"] = std_w
            feats[f"roll_median_{w}"] = float(np.median(arr))
            feats[f"roll_min_{w}"] = float(np.min(arr))
            feats[f"roll_max_{w}"] = float(np.max(arr))
            feats[f"roll_volatility_{w}"] = std_w / (abs(mean_w) + _EPS)
        else:
            for suffix in ("mean", "std", "median", "min", "max", "volatility"):
                feats[f"roll_{suffix}_{w}"] = np.nan
    return feats


def _ema(window_values: np.ndarray, span: int) -> float:
    """Exponentially-weighted mean of `window_values` (oldest..newest),
    span-parameterised the same way as pandas' `.ewm(span=...)`
    (alpha = 2/(span+1)) — recency-weighted, unlike the plain rolling
    mean, so a recent regime shift shows up faster. Recomputed fresh from
    the given slice every call, the same "no persistent/stateful running
    average" design as every other feature here, so training-time and
    recursive-prediction-time values are always computed identically."""
    n = len(window_values)
    if n == 0:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    weights = (1.0 - alpha) ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()
    return float(np.dot(weights, window_values))


def _ema_features(history: List[float], roll_windows: List[int]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n = len(history)
    for w in roll_windows:
        window = history[-w:] if n >= w else history[:]
        feats[f"ema_{w}"] = _ema(np.asarray(window, dtype=float), span=w) if window else np.nan
    return feats


def _trend_slope(window_values: np.ndarray) -> float:
    """OLS slope of `window_values` against an evenly-spaced x-axis — a
    smooth, multi-point trend estimate, far more robust to a single noisy
    point than a two-point momentum ratio."""
    n = len(window_values)
    if n < 2:
        return 0.0
    try:
        slope, _intercept = np.polyfit(np.arange(n, dtype=float), window_values, 1)
        return float(slope)
    except Exception:
        return 0.0


def _trend_and_momentum_features(history: List[float], roll_windows: List[int]) -> Dict[str, float]:
    """
    Momentum (existing): a reactive, two-point relative-change ratio at a
    short lag. Defaults to a neutral 0.0 (not NaN) when there isn't
    enough history yet — "no measurable momentum yet" is a reasonable,
    honest default rather than a fabricated number, so this never forces
    an otherwise-usable row to be dropped.

    Trend slope (new): a smooth OLS slope over the longest available
    rolling window — captures genuine sustained direction rather than a
    single reactive blip, and its normalised counterpart
    (`trend_slope_pct`, slope as a fraction of the window's level) is
    scale-free so it is comparable across series of very different size.

    Short-term coefficient of variation (new): a small, ALWAYS-PRESENT
    volatility signal (independent of the dynamically-sized
    `roll_windows` list), used standalone and by `_interaction_features`.

    Trend baseline (new): a naive-plus-trend one-step extrapolation
    (last observed value + the OLS slope above) — NOT used as a model
    input feature (see fit()/predict()), but as the reference level the
    model is trained to correct rather than reproduce from scratch. A
    gradient-boosted tree cannot predict outside the range of target
    values it was trained on (a leaf's prediction is an average of
    training points that landed in it), so on a genuinely trending
    series a tree asked to predict the raw level directly will
    systematically underpredict once the series moves past whatever
    maximum it happened to reach during training — this is a structural
    limitation of tree ensembles, not something more features or better
    hyperparameters can fix. Training the model on the RESIDUAL from this
    baseline instead (see XGBoostModel.fit()) keeps the modelling target
    close to stationary (oscillating around zero rather than growing
    without bound), which trees handle natively, while the baseline
    itself carries the trend forward exactly like the always-available
    Naive/Holt-style baselines this project already ships.
    """
    feats: Dict[str, float] = {}
    n = len(history)

    momentum_lag = roll_windows[0] if roll_windows else 3
    if n >= momentum_lag + 1:
        prev = history[-momentum_lag - 1]
        feats["momentum"] = (history[-1] - prev) / (abs(prev) + _EPS)
    else:
        feats["momentum"] = 0.0

    slope_window = max(roll_windows) if roll_windows else 3
    window = history[-slope_window:] if n >= slope_window else history[:]
    slope = _trend_slope(np.asarray(window, dtype=float)) if len(window) >= 2 else 0.0
    feats["trend_slope"] = slope
    level = float(np.mean(window)) if window else 0.0
    feats["trend_slope_pct"] = slope / (abs(level) + _EPS)

    short_window = history[-3:] if n >= 3 else history[:]
    if short_window:
        arr = np.asarray(short_window, dtype=float)
        s_mean = float(np.mean(arr))
        s_std = float(np.std(arr)) if len(arr) > 1 else 0.0
        feats["short_term_cv"] = s_std / (abs(s_mean) + _EPS)
        # A MEDIAN (not the single last value, and not the mean) of a
        # short recent window — robust to a single outlying spike sitting
        # right at the end of `history`. The single most recent value
        # alone is a poor "current level" estimate for spike-prone
        # business data: if that one point happens to be a spike, a
        # baseline built from it would be wildly wrong, feeding the model
        # a noisier residual to learn than the raw level would have been
        # in the first place — defeating the point of detrending. The
        # median discards exactly that kind of single-point outlier while
        # still tracking genuine, sustained level shifts.
        robust_level = float(np.median(arr))
    else:
        feats["short_term_cv"] = 0.0
        robust_level = 0.0

    feats["trend_baseline"] = (robust_level + slope) if n >= 1 else 0.0

    return feats


def _point_features(history: List[float], lags: List[int], roll_windows: List[int]) -> Dict[str, float]:
    """
    Feature vector describing the point that comes immediately AFTER
    `history` (i.e. this is what we know looking backward from the point
    being predicted/trained on). `history` must contain ONLY values
    strictly before that point — the single leakage guard every caller
    relies on. Composes the smaller helpers above; kept as one function
    (rather than inlining each call site) so there is exactly one
    "what does history alone tell us" computation, separate from the
    calendar-only and EMA-only pieces added alongside it.
    """
    feats: Dict[str, float] = {}
    feats.update(_lag_features(history, lags))
    feats.update(_rolling_window_features(history, roll_windows))
    feats["expanding_mean"] = float(np.mean(history)) if history else np.nan
    feats.update(_trend_and_momentum_features(history, roll_windows))
    return feats


# ── calendar features (cyclical encoding) ───────────────────────────────────

def _cyclical_encode(value: float, period_length: float) -> Tuple[float, float]:
    """sin/cos pair placing `value` (0-indexed within a cycle of length
    `period_length`) on a unit circle — the standard trick for calendar
    features: it removes the artificial discontinuity a raw integer
    encoding implies (e.g. December=11 and January=0 are numerically far
    apart despite being adjacent in the yearly cycle) by making
    "distance around the cycle" continuous."""
    angle = 2.0 * np.pi * (value / period_length)
    return float(np.sin(angle)), float(np.cos(angle))


def _calendar_features(period, frequency: str) -> Dict[str, float]:
    """
    Calendar features derived purely from the timestamp of the point
    BEING forecast/trained — this is legitimate (not leakage): the
    calendar position of a future period (e.g. "this is a December") is
    always genuinely known in advance, exactly like the seasonal terms
    SARIMA/Prophet/Holt-Winters already use. Raw integer encodings are
    kept (trees can still split cleanly on them, e.g. "month <= 6") and
    cyclical sin/cos pairs are added alongside them — automatic feature
    selection in fit() prunes whichever form a given series doesn't find
    useful. Only frequencies where the calendar position is meaningful
    get features; others return {} rather than fabricating a signal that
    isn't really there.
    """
    ts = period.to_timestamp()
    feats: Dict[str, float] = {}
    if frequency == "daily":
        feats["cal_dayofweek"] = float(ts.dayofweek)
        feats["cal_day"] = float(ts.day)
        feats["cal_month"] = float(ts.month)
        feats["cal_dayofweek_sin"], feats["cal_dayofweek_cos"] = _cyclical_encode(ts.dayofweek, 7)
        feats["cal_month_sin"], feats["cal_month_cos"] = _cyclical_encode(ts.month - 1, 12)
    elif frequency == "weekly":
        week = float(ts.isocalendar()[1])
        feats["cal_weekofyear"] = week
        feats["cal_month"] = float(ts.month)
        feats["cal_weekofyear_sin"], feats["cal_weekofyear_cos"] = _cyclical_encode(week - 1, 52)
        feats["cal_month_sin"], feats["cal_month_cos"] = _cyclical_encode(ts.month - 1, 12)
    elif frequency == "monthly":
        feats["cal_month"] = float(ts.month)
        feats["cal_quarter"] = float(ts.quarter)
        feats["cal_month_sin"], feats["cal_month_cos"] = _cyclical_encode(ts.month - 1, 12)
    elif frequency == "quarterly":
        feats["cal_quarter"] = float(ts.quarter)
        feats["cal_quarter_sin"], feats["cal_quarter_cos"] = _cyclical_encode(ts.quarter - 1, 4)
    # yearly / irregular-fallback: no sub-annual calendar signal exists

    holiday_flag = _holiday_feature(period, frequency)
    if holiday_flag is not None:
        feats["cal_is_holiday_period"] = holiday_flag

    return feats


def _interaction_features(row: Dict[str, float]) -> Dict[str, float]:
    """
    A small, deliberately bounded set of interaction terms — e.g. "how
    strong is the recent trend, scaled by how volatile the series has
    recently been" tells a tree more in one split than the two features
    considered independently. Kept small (a handful of terms, not a
    combinatorial expansion of every feature pair) because this project's
    series are often short, and each additional column is one more
    opportunity to overfit on noise; automatic feature selection in fit()
    prunes any of these that turn out not to matter for a given series
    anyway. Reads only from keys `_point_features`/`_calendar_features`
    always populate when applicable, never from a dynamically-named
    window-specific column, so this never depends on which windows this
    particular series happened to resolve to.
    """
    feats: Dict[str, float] = {}
    momentum = row.get("momentum", 0.0)
    trend_slope_pct = row.get("trend_slope_pct", 0.0)
    short_cv = row.get("short_term_cv", 0.0)

    feats["momentum_x_volatility"] = momentum * short_cv
    feats["trend_x_volatility"] = trend_slope_pct * short_cv

    if "cal_month_sin" in row:
        level = row.get("expanding_mean", 0.0)
        feats["season_x_level_sin"] = row["cal_month_sin"] * level
        feats["season_x_level_cos"] = row["cal_month_cos"] * level

    return feats


def _build_feature_row(history: List[float], period, lags: List[int], roll_windows: List[int], frequency: str) -> Dict[str, float]:
    """
    Single entry point combining every feature category for one point in
    time — called identically at training-matrix-build time and at
    recursive-prediction time, so both always compute the exact same
    feature SET the exact same way. This is the project's core leakage
    guard, preserved and extended (rather than replaced) by this pass.
    """
    feats = _point_features(history, lags, roll_windows)
    feats.update(_ema_features(history, roll_windows))
    feats.update(_calendar_features(period, frequency))
    feats.update(_interaction_features(feats))
    return feats


# ── target transform ────────────────────────────────────────────────────────

def _prepare_target(y: np.ndarray, use_log: bool) -> Tuple[np.ndarray, bool]:
    """
    Apply (or not) a log1p transform to `y`, honouring `use_log` but never
    overriding a hard safety constraint: log1p is undefined below -1 and
    numerically unstable near it, and this project's preprocessing
    pipeline can legitimately produce a negative PERIOD TOTAL (heavy
    refund activity outweighing sales in that period) — so log1p is only
    ever actually applied when EVERY value is non-negative, regardless of
    what `use_log` requests. Returns (possibly-transformed y, whether it
    was actually transformed).

    `use_log` itself is deliberately NOT a blanket "apply whenever safe"
    rule (see `_choose_target_space`'s docstring for why): log1p
    compresses the right-skewed, heteroscedastic scale typical of
    business revenue data and helps when the series' volatility genuinely
    scales with its level, but it can actively hurt raw-scale accuracy on
    data whose volatility instead comes from roughly constant-magnitude
    additive shocks (occasional spikes riding on a stable base) — for
    such data, log-space squared error deprioritises getting a large
    shock's absolute size right. Whether it helps is decided empirically,
    once, in `_choose_target_space`, not assumed from non-negativity
    alone.
    """
    if use_log and y.size and np.all(y >= 0.0):
        return np.log1p(y), True
    return y, False


def _choose_target_space(X_all: np.ndarray, y_all: np.ndarray, XGBRegressor, min_train_rows: int) -> bool:
    """
    Decide, from evidence rather than assumption, whether training in
    log1p-space actually helps THIS series. log1p is only even considered
    when the target is entirely non-negative (see _prepare_target).

    A SINGLE train/validation split — even a moderately sized one — turned
    out to be too noisy for this particular decision during testing: which
    space "wins" on any one tail slice can flip depending on whether a
    handful of large, infrequent points (e.g. spikes) happen to land inside
    or outside that one slice, and picking the wrong space can swing
    raw-scale accuracy by a large margin (unlike a close call between two
    similar hyperparameter configs, where either choice is roughly fine).
    So this instead runs a SMALL rolling-origin comparison — a handful of
    small held-out windows walking backward from the end of the series,
    each with its own short training prefix — and averages RMSE (computed
    in the original, raw scale — the same scale WAPE/MAE/RMSE are reported
    in downstream, so the comparison is apples-to-apples with what actually
    gets shown to users) across all of them for each space. Averaging over
    several windows is far less sensitive to which particular points end
    up in any one window than a single split of any size. Ties or any
    failure default to the raw scale (the more conservative choice, since
    an unhelpful transform is a real, observed risk here, not a
    theoretical one).
    """
    n = len(y_all)
    if not (y_all.size and np.all(y_all >= 0.0)):
        return False

    val_width = max(1, min(3, n // 10))
    if n < min_train_rows + val_width:
        return False
    # (n - min_train_rows) // val_width >= 1 is now guaranteed by the
    # check above, so every window this produces actually has enough
    # rows before it to train on — no window silently no-ops.
    max_windows = 4
    n_windows = min(max_windows, (n - min_train_rows) // val_width)

    def _window_rmse(train_end: int, val_end: int, use_log: bool) -> float:
        X_tr, y_tr_raw = X_all[:train_end], y_all[:train_end]
        X_val, y_val_raw = X_all[train_end:val_end], y_all[train_end:val_end]
        if len(X_tr) < min_train_rows or len(X_val) < 1:
            return float("nan")
        try:
            model = XGBRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                reg_lambda=1.0, reg_alpha=0.1,
                random_state=42, verbosity=0, objective="reg:squarederror",
                n_jobs=1,
            )
            y_tr = np.log1p(y_tr_raw) if use_log else y_tr_raw
            model.fit(X_tr, y_tr)
            raw_preds = model.predict(X_val)
            if use_log:
                raw_preds = np.expm1(raw_preds)
            raw_preds = np.maximum(raw_preds, 0.0)
            return float(np.sqrt(np.mean((y_val_raw - raw_preds) ** 2)))
        except Exception:
            return float("nan")

    raw_scores: List[float] = []
    log_scores: List[float] = []
    for i in range(n_windows):
        val_end = n - i * val_width
        train_end = val_end - val_width
        if train_end < min_train_rows:
            break
        raw_scores.append(_window_rmse(train_end, val_end, use_log=False))
        log_scores.append(_window_rmse(train_end, val_end, use_log=True))

    raw_scores = [s for s in raw_scores if np.isfinite(s)]
    log_scores = [s for s in log_scores if np.isfinite(s)]
    if not raw_scores or not log_scores:
        return False

    return float(np.mean(log_scores)) < float(np.mean(raw_scores))


def _stabilize_forecast_step(raw_pred: float, history: List[float], step_index: int, seasonal_period: Optional[int]) -> float:
    """
    Recursive multi-step forecasting compounds each step's error into the
    next step's input. That cannot be eliminated without abandoning the
    recursive strategy entirely (a direct multi-horizon approach would
    need a separate model per horizon step — a much larger architectural
    change than "improve the implementation"), but its worst consequence
    — a small early error compounding into an implausible runaway
    trajectory many steps out — can be bounded: each step's raw
    prediction is clipped to a plausible range derived from the series'
    own recent behaviour, widening gradually with the step index (further
    -out steps are allowed more room to genuinely trend, exactly like a
    prediction interval widening with horizon) rather than being allowed
    to diverge without limit.

    This is deliberately a SOFT bound (a wide multiple of recent
    volatility, widening every step), not a hard cap at the historical
    max/min — a real trend or seasonal peak must still be forecastable.
    At step_index=0 (the only step walk-forward validation's `predict(1)`
    ever exercises) the band is already wide (3 standard deviations), so
    this is essentially a no-op for one-step accuracy and only actively
    engages for longer horizons where compounding is the real risk.
    """
    window_size = max(12, (seasonal_period or 0) * 2)
    window = history[-window_size:] if len(history) > window_size else history
    if len(window) < 2:
        return max(0.0, raw_pred)

    arr = np.asarray(window, dtype=float)
    recent_std = float(np.std(arr))
    recent_min = float(np.min(arr))
    recent_max = float(np.max(arr))

    widening = 1.0 + 0.5 * step_index
    margin = (recent_std + _EPS) * 3.0 * widening

    lower = max(0.0, recent_min - margin)
    upper = recent_max + margin
    return float(np.clip(raw_pred, lower, upper))


# Hyperparameter search space for suggest_hyperparameters(). Each grid
# entry is a full, curated joint configuration (not a sparse cartesian
# sweep of one or two parameters at a time) so the manual fallback still
# explores meaningfully different regularisation strengths, not just tree
# shape — reg_alpha/reg_lambda/min_child_weight/subsample/colsample_bytree
# all vary across entries now, not just max_depth/learning_rate. Each
# candidate is trained with early stopping (up to
# _MAX_ESTIMATORS_FOR_TUNING rounds), so this also implicitly searches the
# number of boosting rounds.
_TUNING_GRID = (
    {"max_depth": 2, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_alpha": 0.3, "reg_lambda": 1.5},
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0},
)
_MAX_ESTIMATORS_FOR_TUNING = 300
_EARLY_STOPPING_ROUNDS = 20
_OPTUNA_N_TRIALS = 25
_OPTUNA_TIMEOUT_SECONDS = 20


def _grid_search_hyperparameters(X_tr, y_tr, X_val, y_val, XGBRegressor) -> dict:
    best_cfg, best_rmse = {}, np.inf
    for grid_cfg in _TUNING_GRID:
        try:
            model = XGBRegressor(
                n_estimators=_MAX_ESTIMATORS_FOR_TUNING,
                random_state=42, verbosity=0,
                objective="reg:squarederror",
                n_jobs=1,
                early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
                eval_metric="rmse",
                **grid_cfg,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            rmse = float(model.best_score) if model.best_score is not None else np.inf
            best_iter = (int(model.best_iteration) + 1) if model.best_iteration is not None else _MAX_ESTIMATORS_FOR_TUNING
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_cfg = {**grid_cfg, "n_estimators": max(10, best_iter)}
        except Exception:
            continue
    return best_cfg


def _optuna_search_hyperparameters(X_tr, y_tr, X_val, y_val, XGBRegressor, optuna) -> dict:
    """Bayesian (TPE) search over a wider space than the manual grid —
    tree shape AND regularisation strength together, bounded to
    _OPTUNA_N_TRIALS trials / _OPTUNA_TIMEOUT_SECONDS wall-clock seconds
    so this stays cheap enough for an interactive dashboard. Individual
    trial failures are caught by Optuna itself (`catch=`) and simply
    scored as failed, rather than aborting the whole search."""

    def objective(trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
        }
        model = XGBRegressor(
            n_estimators=_MAX_ESTIMATORS_FOR_TUNING,
            random_state=42, verbosity=0,
            objective="reg:squarederror",
            n_jobs=1,
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            eval_metric="rmse",
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        rmse = float(model.best_score) if model.best_score is not None else float("inf")
        best_iter = (int(model.best_iteration) + 1) if model.best_iteration is not None else _MAX_ESTIMATORS_FOR_TUNING
        trial.set_user_attr("n_estimators", max(10, best_iter))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=_OPTUNA_N_TRIALS,
        timeout=_OPTUNA_TIMEOUT_SECONDS,
        show_progress_bar=False,
        catch=(Exception,),
    )

    if study.best_trial is None or not np.isfinite(study.best_value):
        return {}
    best_cfg = dict(study.best_trial.params)
    best_cfg["n_estimators"] = study.best_trial.user_attrs.get("n_estimators", _MAX_ESTIMATORS_FOR_TUNING)
    return best_cfg


class XGBoostModel(BaseForecastModel):
    name = "XGBoost"

    # Minimum number of supervised (feature, target) training rows left
    # after lag features consume the earliest points — a handful of rows
    # is the practical floor for a gradient-boosted tree to learn anything
    # at all; below this, XGBoost is skipped in favour of the always-safe
    # baselines rather than fit on effectively noise.
    MIN_TRAIN_ROWS = 5

    @classmethod
    def is_available(cls) -> bool:
        return _try_import_xgboost() is not None

    def _resolve_lags(self, n: int, frequency: str) -> List[int]:
        """Richer lag candidates than a fixed short list: adds lag 7
        (weekly pattern, useful for daily data) and a half-seasonal lag
        (e.g. 6 for a 12-period yearly cycle) alongside the existing
        short lags and the full seasonal period, still bounded by how
        much history is actually available."""
        seasonal_period = seasonal_period_for_freq(frequency)
        candidates = [1, 2, 3, 6, 7, 12]
        if seasonal_period and seasonal_period > 1:
            candidates.append(seasonal_period)
            if seasonal_period > 3:
                candidates.append(seasonal_period // 2)
        budget = max(1, n - self.MIN_TRAIN_ROWS)
        lags = sorted(set(l for l in candidates if 1 <= l <= budget))
        return lags or [1]

    def _resolve_roll_windows(self, n: int, frequency: str) -> List[int]:
        """Rolling-window sizes scale with how much history is actually
        available: a short series only supports a couple of small
        windows; a longer series additionally supports a longer window
        (captures slower-moving structure a short window/EMA would miss)
        and, when it spans at least two full seasonal cycles, the
        seasonal period itself as a window. Every downstream feature
        category (rolling stats, EMA, trend slope) is driven off this
        same list, so feature richness scales down automatically for
        short series without a second, separate adaptivity mechanism."""
        windows = [w for w in (3, 7) if w < n]
        seasonal_period = seasonal_period_for_freq(frequency)
        if n >= 30:
            windows.append(min(14, n - 1))
        if seasonal_period and seasonal_period > 7 and n >= seasonal_period * 2:
            windows.append(seasonal_period)
        windows = sorted(set(w for w in windows if w >= 1))
        return windows or [max(1, min(2, n - 1))]

    def _min_observations_required(self, frequency: str, n_hint: Optional[int] = None) -> int:
        # A stable estimate of "enough data": resolve lags against a
        # generous hint so the min-obs check doesn't depend on n in a
        # circular way, then require MIN_TRAIN_ROWS usable rows on top.
        hint = n_hint if n_hint is not None else 10_000
        lags = self._resolve_lags(hint, frequency)
        max_lag = max(lags) if lags else 1
        return max_lag + self.MIN_TRAIN_ROWS

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        min_needed = self._min_observations_required(frequency, n_hint=len(series))
        if len(series) < min_needed:
            return False, (
                f"XGBoost needs at least {min_needed} {frequency} observations (enough history to "
                f"build lag features plus {self.MIN_TRAIN_ROWS} training examples) — only "
                f"{len(series)} available — skipped."
            )
        return True, None

    def _build_training_matrix(self, values: np.ndarray, index: pd.PeriodIndex, frequency: str) -> pd.DataFrame:
        rows = []
        values_list = values.tolist()
        for t in range(len(values_list)):
            rows.append(_build_feature_row(values_list[:t], index[t], self.lags_, self.roll_windows_, frequency))
        return pd.DataFrame(rows)

    def _build_valid_xy(self, series: pd.Series, frequency: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Shared by suggest_hyperparameters() and fit() — builds the full
        feature matrix and drops rows still missing a lag value, so both
        use the identical, single feature-construction pipeline (no
        duplicated logic).

        Returns (X, y, baseline): `baseline` is the `trend_baseline`
        column (see `_trend_and_momentum_features`) pulled OUT of X — it
        is never given to the model as an input feature, only used to
        compute the residual target the model actually learns (see
        fit()) and to reconstruct real predictions from the model's
        residual output (see predict()). Keeping it out of X entirely
        avoids the model being able to partially "relearn" the baseline
        itself, which would reintroduce the extrapolation problem the
        residual-target design exists to avoid.
        """
        n = len(series)
        self.lags_ = self._resolve_lags(n, frequency)
        self.roll_windows_ = self._resolve_roll_windows(n, frequency)
        values = series.values.astype(float)

        features = self._build_training_matrix(values, series.index, frequency)
        lag_cols = [f"lag_{l}" for l in self.lags_]
        valid_mask = features[lag_cols].notna().all(axis=1)
        valid_features = features.loc[valid_mask].reset_index(drop=True)
        baseline = valid_features["trend_baseline"].to_numpy(dtype=float)
        X = valid_features.drop(columns=["trend_baseline"])
        y = values[valid_mask.values]
        return X, y, baseline

    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """
        Tunes against the RESIDUAL target fit() will actually train on
        (raw value minus the naive+trend `trend_baseline` — see fit()'s
        docstring for why), so a config that looks good here is a config
        that is actually good for what gets fit. First decides,
        empirically, whether log1p-space training helps THIS series'
        residual at all (see `_choose_target_space` — never assumed:
        residuals are usually mixed-sign, so log1p is mostly inapplicable
        by construction, and even when the series itself is non-negative
        it can genuinely hurt, e.g. data dominated by additive spikes
        rather than level-scaling volatility). Then tries Optuna's
        Bayesian search (wider space: tree shape AND regularisation
        strength together) in whichever space won, falling back to an
        expanded manual grid search if Optuna isn't installed or fails
        for any reason. Returns {} (defaults used) if there isn't enough
        data for a meaningful internal split on top of XGBoost's own
        lag/row requirements.
        """
        XGBRegressor = _try_import_xgboost()
        if XGBRegressor is None:
            return {}

        n = len(series)
        min_needed = self._min_observations_required(frequency, n_hint=n)
        if n < min_needed:
            return {}

        X_all, y_all, baseline_all = self._build_valid_xy(series, frequency)
        resid_all = y_all - baseline_all
        n_valid = len(X_all)
        val_size = max(1, min(3, n_valid // 5))
        split = n_valid - val_size
        if split < self.MIN_TRAIN_ROWS or n_valid - split < 1:
            return {}

        X_tr_raw, y_tr_raw = X_all.iloc[:split].values, resid_all[:split]
        X_val_raw, y_val_raw = X_all.iloc[split:].values, resid_all[split:]

        # The log-vs-raw decision uses a small rolling-origin comparison
        # over the TRAINING portion only (X_tr_raw/y_tr_raw) — deliberately
        # NOT the full X_all/resid_all, which would let its internal
        # windows overlap the same tail rows held out below as
        # X_val_raw/y_val_raw for the actual hyperparameter ranking. That
        # overlap wouldn't be temporal leakage (every window inside
        # _choose_target_space is still chronological), but it would mean
        # the space decision and the hyperparameter ranking are partly
        # judged against the same held-out points — an avoidable form of
        # validation-set reuse. Restricting it to the training split costs
        # nothing (it is barely smaller than the full matrix) and removes
        # that overlap entirely.
        try:
            use_log = _choose_target_space(X_tr_raw, y_tr_raw, XGBRegressor, self.MIN_TRAIN_ROWS)
        except Exception as exc:
            logger.debug("Target-space selection failed, defaulting to raw scale (%s)", exc)
            use_log = False

        y_tr, use_log_tr = _prepare_target(y_tr_raw, use_log)
        y_val, use_log_val = _prepare_target(y_val_raw, use_log)
        if use_log_tr != use_log_val:
            # Cannot currently happen — _choose_target_space only returns
            # True after confirming the WHOLE residual array is
            # non-negative, and y_tr_raw/y_val_raw are both subsets of
            # that same array, so both must pass (or both fail) the same
            # safety check in _prepare_target. Enforced explicitly anyway
            # rather than relying on that invariant implicitly: training
            # and validating in two different target spaces within the
            # same tuning run would silently corrupt early stopping and
            # the resulting RMSE-based ranking.
            logger.debug("Train/validation target-space mismatch during tuning, falling back to raw scale")
            use_log = False
            y_tr, use_log_tr = _prepare_target(y_tr_raw, False)
            y_val, use_log_val = _prepare_target(y_val_raw, False)

        best_cfg = {}
        optuna = _try_import_optuna()
        if optuna is not None:
            try:
                best_cfg = _optuna_search_hyperparameters(X_tr_raw, y_tr, X_val_raw, y_val, XGBRegressor, optuna)
            except Exception as exc:
                logger.debug("Optuna hyperparameter search failed, falling back to grid search (%s)", exc)
                best_cfg = {}

        if not best_cfg:
            best_cfg = _grid_search_hyperparameters(X_tr_raw, y_tr, X_val_raw, y_val, XGBRegressor)

        if best_cfg:
            best_cfg["log_transform"] = use_log
        return best_cfg

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "XGBoostModel":
        XGBRegressor = _try_import_xgboost()
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed — XGBoost is unavailable.")

        n = len(series)
        min_needed = self._min_observations_required(frequency, n_hint=n)
        if n < min_needed:
            raise ValueError(f"XGBoost needs at least {min_needed} observations — only {n} available.")

        self.series_ = series
        self.frequency_ = frequency
        X, y, baseline = self._build_valid_xy(series, frequency)

        if len(X) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"XGBoost needs at least {self.MIN_TRAIN_ROWS} usable training rows after lag "
                f"features are built — only {len(X)} available."
            )

        # Train on the RESIDUAL from a naive+trend baseline, not the raw
        # value (see `trend_baseline`'s docstring in
        # _trend_and_momentum_features): a gradient-boosted tree cannot
        # predict outside the range of target values it was trained on,
        # so on a trending series a tree trained on the raw level
        # systematically underpredicts once the series moves past
        # whatever maximum it happened to reach during training — a
        # structural limitation no amount of feature engineering or
        # hyperparameter tuning can fix, and the most likely reason
        # XGBoost was sometimes losing to Naive/SES on real, trending
        # business data. Modelling the residual keeps the target close to
        # stationary, which trees handle natively, while `baseline`
        # itself carries the trend forward and is added back in
        # predict().
        residual = y - baseline

        # A NaN/Inf target reaching XGBoost's native fit() surfaces as a
        # cryptic low-level XGBoostError with a C++ stack trace, not a
        # readable message — this project's preprocessing.py is
        # responsible for cleaning/interpolating the series before it
        # reaches any model, but a NaN target here (which would mean a
        # genuinely missing observation reached this far uncleaned, since
        # unlike features, XGBoost has no native handling for a missing
        # LABEL) is real, reachable input, not a purely theoretical case.
        # Fail with the same clear, anticipated ValueError style already
        # used for the other precondition checks above, instead of
        # letting the underlying library's internal error leak through.
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                "XGBoost cannot train on a series containing NaN, infinite, or "
                "otherwise non-finite values — this indicates missing or invalid "
                "data reached the model without being cleaned upstream."
            )

        # log1p(residual) training with automatic expm1() inversion, but
        # only when suggest_hyperparameters() already validated that it
        # actually helps THIS series' residual (see _choose_target_space)
        # — passed through as the "log_transform" hyperparameter. Absent
        # that hint (e.g. a series too short to tune), the conservative
        # default is NOT to transform: log1p is not universally
        # beneficial even before detrending, and a residual is usually
        # mixed-sign by construction (it oscillates around the baseline),
        # so it will typically fail the non-negativity safety check
        # inside _prepare_target anyway — that check is enforced
        # unconditionally regardless of the hint, since a shorter/
        # different walk-forward fold's residual could differ in sign
        # composition even when the tuning prefix's didn't.
        requested_log = bool(hyperparams.get("log_transform", False))
        y_train, use_log = _prepare_target(residual, requested_log)

        max_depth = hyperparams.get("max_depth", 3)
        learning_rate = hyperparams.get("learning_rate", 0.08)
        n_estimators = hyperparams.get("n_estimators", 150)
        subsample = hyperparams.get("subsample", 0.9)
        colsample_bytree = hyperparams.get("colsample_bytree", 0.9)
        # A conservative default (rather than XGBoost's permissive
        # implicit default of 1) — small, noisy business datasets benefit
        # from requiring a bit more evidence per leaf; suggest_hyperparameters()
        # tunes this properly whenever there's enough data to do so.
        min_child_weight = hyperparams.get("min_child_weight", 3)
        reg_alpha = hyperparams.get("reg_alpha", 0.1)
        reg_lambda = hyperparams.get("reg_lambda", 1.0)

        def _make_model(n_est: int) -> "XGBRegressor":
            return XGBRegressor(
                n_estimators=n_est, max_depth=max_depth, learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                random_state=42, verbosity=0, objective="reg:squarederror",
                n_jobs=1,
            )

        model = _make_model(n_estimators)
        model.fit(X.values, y_train)
        feature_columns = list(X.columns)

        # Automatic feature selection: one bounded refit (not an
        # iterative search) on whatever columns this fit found genuinely
        # useful (non-zero split importance), if doing so actually drops
        # at least one column.
        importances = model.feature_importances_
        keep_mask = importances > 0
        if 0 < int(keep_mask.sum()) < len(keep_mask):
            reduced_columns = [c for c, keep in zip(feature_columns, keep_mask) if keep]
            reduced_model = _make_model(n_estimators)
            reduced_model.fit(X.loc[:, reduced_columns].values, y_train)
            model, feature_columns = reduced_model, reduced_columns

        self._model = model
        self._feature_columns = feature_columns
        self.use_log_transform_ = use_log
        self.hyperparameters_ = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": round(float(subsample), 3),
            "colsample_bytree": round(float(colsample_bytree), 3),
            "min_child_weight": min_child_weight,
            "reg_alpha": round(float(reg_alpha), 4),
            "reg_lambda": round(float(reg_lambda), 4),
            "log_transform": use_log,
        }
        self.fitted = True
        return self

    def _feature_row(self, history: List[float], period) -> Tuple[np.ndarray, float]:
        """Returns (model input array, trend_baseline value) for the
        point immediately after `history`. The array is aligned strictly
        to the column order/set fixed at training time; a column this
        frequency never produces (e.g. a calendar feature only
        meaningful at a different cadence), or one feature selection
        dropped, is filled with 0 rather than left missing, since the
        underlying array-based predict call has no concept of named/
        missing columns. `trend_baseline` is returned separately (it is
        never one of `self._feature_columns` — see _build_valid_xy) since
        predict() needs its value to reconstruct a real prediction from
        the model's residual output regardless of whether it happened to
        be a useful model input."""
        row = _build_feature_row(history, period, self.lags_, self.roll_windows_, self.frequency_)
        array = np.array([row.get(c, 0.0) for c in self._feature_columns], dtype=float)
        baseline_value = float(row.get("trend_baseline", 0.0))
        return array, baseline_value

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("XGBoost must be fit() before predict().")

        seasonal_period = seasonal_period_for_freq(self.frequency_)

        # RECURSIVE multi-step forecasting: each step's prediction is
        # appended to `history` so the next step's lag/rolling/EMA/
        # expanding/momentum/trend features are computed from "real
        # history + our own prior forecasts," exactly the way every other
        # model in this library (Holt/SES/SARIMA/...) implicitly carries
        # its own state forward. `history` always stays in the series'
        # original units (predictions are inverse-transformed out of
        # log-space, if applicable, and detrended residual predictions
        # are added back onto their baseline, before being appended) so
        # every feature computed from it means the same thing at every
        # step.
        history = list(self.series_.values.astype(float))
        last_period = self.series_.index[-1]
        preds = []
        for i in range(steps):
            next_period = last_period + (i + 1)
            row, baseline_value = self._feature_row(history, next_period)
            raw_resid_pred = float(self._model.predict(row.reshape(1, -1))[0])
            resid_pred = float(np.expm1(raw_resid_pred)) if self.use_log_transform_ else raw_resid_pred
            # Reconstruct the real prediction from the naive+trend
            # baseline plus the model's (possibly inverse-transformed)
            # residual correction — see fit()'s docstring for why the
            # model is trained on the residual rather than the raw level.
            pred = baseline_value + resid_pred
            pred = max(0.0, pred)
            pred = _stabilize_forecast_step(pred, history, i, seasonal_period)
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)
