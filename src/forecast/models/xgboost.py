"""
xgboost.py
----------
XGBoost forecasting: turns the series into a supervised-learning table
(lag features, rolling statistics, an expanding mean, momentum/volatility
features, and calendar features) and trains xgboost.XGBRegressor on it.
Recommended by Phase 1's model_selector.py for "Highly Volatile" series,
where a gradient-boosted tree can capture non-linear structure a
classical linear method cannot.

There is no native multi-step API for a tabular regressor, so forecasting
beyond one step uses RECURSIVE prediction (the standard approach — see
e.g. sktime's `make_reduction`/Darts' `RegressionModel`): predict one step
ahead, append that prediction to the working history, recompute features
from the extended history, predict the next step, and so on. Every
feature at every step (training AND recursive prediction) is built from
the SAME single function, `_point_features()`, so there is exactly one
implementation of "what do the features look like at position t" — this
is a deliberate leakage guard: a training row for position t only ever
sees values strictly BEFORE t (`history[:t]`), and a prediction step for
a future position only ever sees real history plus the model's own prior
predictions, never a real future value.

Production-grade pass (Tasks 3, 5, 9):
  - Feature set widened: rolling median/min/max alongside the existing
    rolling mean/std, a rolling coefficient-of-variation ("volatility")
    feature, a momentum/recent-growth-rate feature, and (best-effort,
    only for daily/weekly data, only if the optional `holidays` package
    is importable) a holiday-period indicator.
  - suggest_hyperparameters() grid-searches (max_depth, learning_rate),
    each candidate trained with early stopping against a small internal
    chronological validation split, so both the tree shape AND the
    number of boosting rounds are chosen from data rather than fixed
    guesses — once per forecast run, not once per walk-forward fold (see
    base.py's suggest_hyperparameters() docstring).
  - fit() performs one bounded round of automatic feature selection: if
    the fitted model assigns exactly zero importance to any feature, it
    is dropped and the model is refit once on the reduced set (not an
    iterative search — at most one extra fit, cheap for XGBoost even
    though the model is refit inside every walk-forward fold too).
  - reg_alpha (L1) added alongside the existing reg_lambda (L2) and
    subsample/colsample_bytree — regularisation appropriate for the
    small, noisy datasets this project deals with.

Depends on the optional `xgboost` package. Degrades gracefully via
is_available() if it is not installed, exactly like every other optional
model in this library (Holt-Winters/SARIMA -> statsmodels, Prophet ->
prophet).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import seasonal_period_for_freq

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


def _point_features(history: List[float], lags: List[int], roll_windows: List[int]) -> Dict[str, float]:
    """
    Feature vector describing the point that comes immediately AFTER
    `history` (i.e. this is what we know looking backward from the point
    being predicted/trained on). `history` must contain ONLY values
    strictly before that point — the single leakage guard every caller
    below relies on.
    """
    n = len(history)
    feats: Dict[str, float] = {}

    # Lag features: value exactly `lag` steps back. NaN if not enough
    # history yet — rows with any NaN lag are dropped before training
    # (see XGBoostModel.fit), so a short lag-history never silently
    # trains on a fabricated value.
    for lag in lags:
        feats[f"lag_{lag}"] = history[-lag] if n >= lag else np.nan

    # Rolling statistics over the last `w` points (or all available
    # points if fewer than `w` exist yet, so early rows still get a
    # usable, if noisier, estimate rather than NaN).
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
            # Coefficient of variation — a scale-free "how volatile has
            # this window been" signal, distinct from the raw std above.
            feats[f"roll_volatility_{w}"] = std_w / (abs(mean_w) + _EPS)
        else:
            for suffix in ("mean", "std", "median", "min", "max", "volatility"):
                feats[f"roll_{suffix}_{w}"] = np.nan

    # Expanding mean: mean of everything seen so far. Always defined once
    # there is at least one prior point (captures the series' overall
    # level independent of any fixed window size).
    feats["expanding_mean"] = float(np.mean(history)) if history else np.nan

    # Momentum / recent growth rate: relative change from a short lag ago
    # to the most recent point. Defaults to a neutral 0.0 (not NaN) when
    # there isn't enough history yet — unlike a raw lag value, "no
    # measurable momentum yet" is a reasonable, honest default rather
    # than a fabricated number, so this never forces an otherwise-usable
    # row to be dropped.
    momentum_lag = roll_windows[0] if roll_windows else 3
    if n >= momentum_lag + 1:
        prev = history[-momentum_lag - 1]
        feats["momentum"] = (history[-1] - prev) / (abs(prev) + _EPS)
    else:
        feats["momentum"] = 0.0

    return feats


def _calendar_features(period, frequency: str) -> Dict[str, float]:
    """
    Calendar features derived purely from the timestamp of the point
    BEING forecast/trained — this is legitimate (not leakage): the
    calendar position of a future period (e.g. "this is a December") is
    always genuinely known in advance, exactly like the seasonal terms
    SARIMA/Prophet/Holt-Winters already use. Only frequencies where the
    calendar position is meaningful get features; others return {} rather
    than fabricating a signal that isn't really there.
    """
    ts = period.to_timestamp()
    feats: Dict[str, float] = {}
    if frequency == "daily":
        feats.update({"cal_dayofweek": float(ts.dayofweek), "cal_day": float(ts.day), "cal_month": float(ts.month)})
    elif frequency == "weekly":
        feats.update({"cal_weekofyear": float(ts.isocalendar()[1]), "cal_month": float(ts.month)})
    elif frequency == "monthly":
        feats.update({"cal_month": float(ts.month), "cal_quarter": float(ts.quarter)})
    elif frequency == "quarterly":
        feats.update({"cal_quarter": float(ts.quarter)})
    # yearly / irregular-fallback: no sub-annual calendar signal exists

    holiday_flag = _holiday_feature(period, frequency)
    if holiday_flag is not None:
        feats["cal_is_holiday_period"] = holiday_flag

    return feats


# Hyperparameter search space for suggest_hyperparameters(). Each
# candidate is trained with early stopping (up to _MAX_ESTIMATORS_FOR_TUNING
# rounds), so this also implicitly searches the number of boosting rounds
# — three (max_depth, learning_rate) combinations x early-stopped
# training is fast for XGBoost even though it happens once per forecast
# run on top of the per-fold refits walk-forward validation already does.
_TUNING_GRID = (
    {"max_depth": 2, "learning_rate": 0.10},
    {"max_depth": 3, "learning_rate": 0.08},
    {"max_depth": 4, "learning_rate": 0.05},
)
_MAX_ESTIMATORS_FOR_TUNING = 300
_EARLY_STOPPING_ROUNDS = 20


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
        seasonal_period = seasonal_period_for_freq(frequency)
        candidates = [1, 2, 3, 6]
        if seasonal_period and seasonal_period > 1:
            candidates.append(seasonal_period)
        budget = max(1, n - self.MIN_TRAIN_ROWS)
        lags = sorted(set(l for l in candidates if l <= budget))
        return lags or [1]

    def _resolve_roll_windows(self, n: int) -> List[int]:
        windows = [w for w in (3, 7) if w < n]
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
            row = _point_features(values_list[:t], self.lags_, self.roll_windows_)
            row.update(_calendar_features(index[t], frequency))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_valid_xy(self, series: pd.Series, frequency: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Shared by suggest_hyperparameters() and fit() — builds the
        full feature matrix and drops rows still missing a lag value, so
        both use the identical, single feature-construction pipeline
        (Task 15 — no duplicated logic)."""
        n = len(series)
        self.lags_ = self._resolve_lags(n, frequency)
        self.roll_windows_ = self._resolve_roll_windows(n)
        values = series.values.astype(float)

        features = self._build_training_matrix(values, series.index, frequency)
        lag_cols = [f"lag_{l}" for l in self.lags_]
        valid_mask = features[lag_cols].notna().all(axis=1)
        X = features.loc[valid_mask].reset_index(drop=True)
        y = values[valid_mask.values]
        return X, y

    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """
        Tasks 5, 9 — grid-search (max_depth, learning_rate); each
        candidate is trained with early stopping against a small internal
        chronological validation split of whatever series it is given,
        so the number of boosting rounds is chosen from data too. Returns
        {} (defaults used) if there isn't enough data for a meaningful
        internal split on top of XGBoost's own lag/row requirements.
        """
        XGBRegressor = _try_import_xgboost()
        if XGBRegressor is None:
            return {}

        n = len(series)
        min_needed = self._min_observations_required(frequency, n_hint=n)
        if n < min_needed:
            return {}

        X_all, y_all = self._build_valid_xy(series, frequency)
        n_valid = len(X_all)
        val_size = max(1, min(3, n_valid // 5))
        split = n_valid - val_size
        if split < self.MIN_TRAIN_ROWS or n_valid - split < 1:
            return {}

        X_tr, y_tr = X_all.iloc[:split].values, y_all[:split]
        X_val, y_val = X_all.iloc[split:].values, y_all[split:]

        best_cfg, best_rmse = {}, np.inf
        for grid_cfg in _TUNING_GRID:
            try:
                model = XGBRegressor(
                    n_estimators=_MAX_ESTIMATORS_FOR_TUNING,
                    subsample=0.9, colsample_bytree=0.9,
                    reg_lambda=1.0, reg_alpha=0.1,
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
        X, y = self._build_valid_xy(series, frequency)

        if len(X) < self.MIN_TRAIN_ROWS:
            raise ValueError(
                f"XGBoost needs at least {self.MIN_TRAIN_ROWS} usable training rows after lag "
                f"features are built — only {len(X)} available."
            )

        max_depth = hyperparams.get("max_depth", 3)
        learning_rate = hyperparams.get("learning_rate", 0.08)
        n_estimators = hyperparams.get("n_estimators", 150)

        def _make_model(n_est: int) -> "XGBRegressor":
            return XGBRegressor(
                n_estimators=n_est, max_depth=max_depth, learning_rate=learning_rate,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, reg_alpha=0.1,
                random_state=42, verbosity=0, objective="reg:squarederror",n_jobs=1
            )

        model = _make_model(n_estimators)
        model.fit(X.values, y)
        feature_columns = list(X.columns)

        # Task 9 — automatic feature selection: one bounded refit (not an
        # iterative search) on whatever columns this fit found genuinely
        # useful (non-zero split importance), if doing so actually drops
        # at least one column.
        importances = model.feature_importances_
        keep_mask = importances > 0
        if 0 < int(keep_mask.sum()) < len(keep_mask):
            reduced_columns = [c for c, keep in zip(feature_columns, keep_mask) if keep]
            reduced_model = _make_model(n_estimators)
            reduced_model.fit(X.loc[:, reduced_columns].values, y)
            model, feature_columns = reduced_model, reduced_columns

        self._model = model
        self._feature_columns = feature_columns
        self.hyperparameters_ = {"max_depth": max_depth, "learning_rate": learning_rate, "n_estimators": n_estimators}
        self.fitted = True
        return self

    def _feature_row(self, history: List[float], period) -> np.ndarray:
        row = _point_features(history, self.lags_, self.roll_windows_)
        row.update(_calendar_features(period, self.frequency_))
        # Align strictly to the column order/set fixed at training time;
        # a column this frequency never produces (e.g. a calendar feature
        # only meaningful at a different cadence), or one feature
        # selection dropped, is filled with 0 rather than left missing,
        # since the underlying array-based predict call has no concept
        # of named/missing columns.
        return np.array([row.get(c, 0.0) for c in self._feature_columns], dtype=float)

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("XGBoost must be fit() before predict().")

        # RECURSIVE multi-step forecasting: each step's prediction is
        # appended to `history` so the next step's lag/rolling/expanding/
        # momentum features are computed from "real history + our own
        # prior forecasts," exactly the way every other model in this
        # library (Holt/SES/SARIMA/...) implicitly carries its own state
        # forward.
        history = list(self.series_.values.astype(float))
        last_period = self.series_.index[-1]
        preds = []
        for i in range(steps):
            next_period = last_period + (i + 1)
            row = self._feature_row(history, next_period)
            pred = float(self._model.predict(row.reshape(1, -1))[0])
            pred = max(0.0, pred)
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)
