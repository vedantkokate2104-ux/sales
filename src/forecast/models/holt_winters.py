"""
holt_winters.py
----------------
Holt-Winters Exponential Smoothing: models level, trend, AND a repeating
seasonal cycle. Backed by statsmodels.tsa.holtwinters.ExponentialSmoothing.

Seasonal period is now frequency-aware (Task 1/4): it is looked up from
forecast.utils.SEASONAL_PERIOD for the series' detected frequency (7 for
daily/weekly-within-a-week, 52 for weekly/yearly-within-a-week, 12 for
monthly, 4 for quarterly) instead of always assuming 12-month seasonality.
Yearly data has no sub-annual seasonal cycle at all, so Holt-Winters is not
suitable for it — check_requirements() reports this.

Depends on the optional `statsmodels` package. If it is not installed,
is_available() returns False and forecast.trainer skips this model with a
warning instead of crashing — the same graceful-degradation pattern Phase
1's embedding_mapper.py uses for sentence-transformers.

Automatic trend/seasonality tuning (production-grade pass, Task 5):
suggest_hyperparameters() picks among a small set of trend configurations
(additive trend, damped additive trend, no trend) and — only when every
value is strictly positive — additive vs. multiplicative seasonality, by
AIC, once per forecast run. Previously trend="add"/seasonal="add" was
always used regardless of whether the data actually supported a damped or
multiplicative fit better.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import seasonal_period_for_freq

# Trend configurations tried by suggest_hyperparameters(). damped_trend is
# only meaningful alongside an actual trend component, so it is dropped
# from the kwargs entirely when trend=None (passing damped_trend=False
# with trend=None is accepted by statsmodels but meaningless).
_TREND_CONFIGS = (
    {"trend": "add", "damped_trend": False},
    {"trend": "add", "damped_trend": True},
    {"trend": None},
)


def _try_import_statsmodels():
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        return ExponentialSmoothing
    except ImportError:
        return None


class HoltWintersModel(BaseForecastModel):
    name = "Holt-Winters"

    @classmethod
    def is_available(cls) -> bool:
        return _try_import_statsmodels() is not None

    def _resolve_seasonal_periods(self, frequency: str) -> Optional[int]:
        # forecast.utils.SEASONAL_PERIOD returns None for "yearly" (no
        # sub-annual cycle) and for "irregular"/"unknown" (no clean cycle
        # to model). trainer.py always passes the CANONICAL frequency the
        # training series was actually built at (irregular/unknown data is
        # already resolved to a "monthly" fallback grain before it reaches
        # any model — see utils.effective_training_frequency()), so in
        # practice only "yearly" data reaches here as None.
        return seasonal_period_for_freq(frequency)

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        seasonal_periods = self._resolve_seasonal_periods(frequency)
        if seasonal_periods is None:
            return False, (
                f"Holt-Winters needs a repeating seasonal cycle, but '{frequency}' data has no "
                f"sub-annual cycle to model — skipped."
            )
        min_obs = seasonal_periods * 2
        if len(series) < min_obs:
            return False, (
                f"Holt-Winters needs at least {min_obs} {frequency} observations (2 full seasonal "
                f"cycles) — only {len(series)} available — skipped."
            )
        return True, None

    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """
        Task 5 — picks trend shape (additive / damped additive / none)
        and seasonal mode (additive, or multiplicative if every tuning
        value is strictly positive) by lowest AIC among a small set of
        statsmodels fits, once per forecast run. Multiplicative
        seasonality is only attempted when there is no zero/negative
        revenue in the data — it is mathematically undefined and
        numerically unstable otherwise, and this project's series can
        legitimately contain zero-sales periods.
        """
        ExponentialSmoothing = _try_import_statsmodels()
        if ExponentialSmoothing is None:
            return {}
        seasonal_periods = self._resolve_seasonal_periods(frequency)
        if seasonal_periods is None or len(series) < seasonal_periods * 2:
            return {}

        values = series.values.astype(float)
        seasonal_modes = ["add"] + (["mul"] if np.all(values > 0) else [])

        best_cfg, best_aic = {"trend": "add", "damped_trend": False, "seasonal": "add"}, np.inf
        for trend_cfg in _TREND_CONFIGS:
            for seasonal_mode in seasonal_modes:
                try:
                    result = ExponentialSmoothing(
                        values, seasonal=seasonal_mode, seasonal_periods=seasonal_periods,
                        initialization_method="estimated", **trend_cfg,
                    ).fit(optimized=True)
                    if np.isfinite(result.aic) and result.aic < best_aic:
                        best_aic = result.aic
                        best_cfg = {**trend_cfg, "seasonal": seasonal_mode}
                except Exception:
                    continue

        return best_cfg

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "HoltWintersModel":
        ExponentialSmoothing = _try_import_statsmodels()
        if ExponentialSmoothing is None:
            raise ImportError("statsmodels is not installed — Holt-Winters is unavailable.")

        seasonal_periods = self._resolve_seasonal_periods(frequency)
        if seasonal_periods is None:
            raise ValueError(f"Holt-Winters needs a sub-annual seasonal cycle — '{frequency}' data has none.")
        min_obs = seasonal_periods * 2
        if len(series) < min_obs:
            raise ValueError(
                f"Holt-Winters needs at least {min_obs} {frequency} observations "
                f"(2 full seasonal cycles) — only {len(series)} available."
            )

        trend = hyperparams.get("trend", "add")
        damped_trend = hyperparams.get("damped_trend", False) if trend is not None else False
        seasonal_mode = hyperparams.get("seasonal", "add")
        values = series.values.astype(float)
        # Multiplicative seasonality is undefined/unstable with
        # zero-or-negative values — fall back to additive rather than
        # letting statsmodels raise, even if a stale "mul" hyperparameter
        # were ever passed in against different data than it was tuned on.
        if seasonal_mode == "mul" and not np.all(values > 0):
            seasonal_mode = "add"

        self.series_ = series
        self.frequency_ = frequency
        self.seasonal_periods_ = seasonal_periods
        self.trend_config_ = {"trend": trend, "damped_trend": damped_trend, "seasonal": seasonal_mode}
        trend_kwargs = {"trend": trend, "damped_trend": damped_trend} if trend is not None else {"trend": None}
        model = ExponentialSmoothing(
            values, seasonal=seasonal_mode,
            seasonal_periods=seasonal_periods, initialization_method="estimated", **trend_kwargs,
        )
        self._result = model.fit(optimized=True)
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Holt-Winters must be fit() before predict().")
        return np.asarray(self._result.forecast(steps), dtype=float)
