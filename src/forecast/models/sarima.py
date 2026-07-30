"""
sarima.py
---------
SARIMA (Seasonal ARIMA): a classical statistical model that jointly
captures trend and seasonality through differencing. Backed by
statsmodels.tsa.statespace.sarimax.SARIMAX with a fixed, moderate default
non-seasonal order — this project does not perform auto-ARIMA order search
(that would require an additional dependency such as pmdarima); (1,1,1) is
a common, reasonable default for business data.

Seasonal order is now frequency-aware (Task 1/4): the seasonal period
(the "12" in the old fixed (1,1,1,12)) is looked up from
forecast.utils.SEASONAL_PERIOD for the series' detected frequency instead
of always assuming monthly data. Frequencies with no meaningful sub-annual
cycle (yearly data) fall back to a non-seasonal ARIMA fit — SARIMA reduces
to plain ARIMA rather than being skipped outright, since trend-only
differencing is still meaningful for yearly data.

Automatic order search (production-grade pass, Task 5): the non-seasonal
(p, q) order is no longer a fixed (1, 1) — suggest_hyperparameters()
grid-searches a small (p, q) space and picks whichever combination has
the lowest AIC, once per forecast run (see base.py's
suggest_hyperparameters() docstring for exactly when this runs and why
only once). The differencing order d stays fixed at 1 rather than also
being searched — AIC is only directly comparable across models fit on
the SAME differenced series, so standard practice is to fix d first
(commonly via a stationarity test) and search p/q by AIC afterward,
rather than comparing AIC across different d values directly; d=1 is a
common, reasonable default for trending business revenue data. The
seasonal order is not searched, to keep this one-time search fast.

Depends on the optional `statsmodels` package. Degrades gracefully via
is_available() if it is not installed.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import seasonal_period_for_freq

# Non-seasonal AR/MA order search space for suggest_hyperparameters().
# (0, 0) is skipped — no AR or MA terms at all is a degenerate ARIMA
# model, not a meaningful SARIMA candidate. 3x3 - 1 = 8 SARIMAX fits per
# tuning call, small enough to run once per forecast request without a
# noticeable delay.
_P_Q_GRID = (0, 1, 2)
_D_FIXED = 1


def _try_import_sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        return SARIMAX
    except ImportError:
        return None


class SARIMAModel(BaseForecastModel):
    name = "SARIMA"

    ORDER = (1, 1, 1)  # fallback default if tuning is skipped/fails
    MIN_OBS_NON_SEASONAL = 10   # small fallback floor when no seasonal cycle applies (e.g. yearly data)

    @classmethod
    def is_available(cls) -> bool:
        return _try_import_sarimax() is not None

    def _resolve_seasonal_order(self, frequency: str, series_len: int) -> Tuple[Tuple[int, int, int, int], int]:
        """Returns (seasonal_order, min_obs_required) for the given
        (canonical) frequency. Falls back to a non-seasonal order,
        (0, 0, 0, 0), when the frequency has no meaningful sub-annual
        seasonal cycle (yearly) — SARIMA still fits, just without a
        seasonal component, rather than being blocked entirely."""
        seasonal_period = seasonal_period_for_freq(frequency)
        if seasonal_period and seasonal_period > 1:
            return (1, 1, 1, seasonal_period), seasonal_period * 2
        return (0, 0, 0, 0), self.MIN_OBS_NON_SEASONAL

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        _, min_obs = self._resolve_seasonal_order(frequency, len(series))
        if len(series) < min_obs:
            seasonal_period = seasonal_period_for_freq(frequency)
            if seasonal_period and seasonal_period > 1:
                reason = f"needs at least {min_obs} {frequency} observations (2 full seasonal cycles)"
            else:
                reason = f"needs at least {min_obs} observations to fit reliably"
            return False, f"SARIMA {reason} — only {len(series)} available — skipped."
        return True, None

    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """
        Task 5 — grid-search the non-seasonal (p, q) order, picking the
        lowest-AIC combination. See module docstring for why d is fixed
        and the seasonal order is not searched. Any combination that
        fails to converge is simply skipped, not treated as an error —
        SARIMAX's optimizer can legitimately fail on some orders for some
        data without that indicating anything is wrong overall.
        """
        SARIMAX = _try_import_sarimax()
        if SARIMAX is None:
            return {}
        seasonal_order, min_obs = self._resolve_seasonal_order(frequency, len(series))
        if len(series) < min_obs:
            return {}  # too little data to tune (check_requirements will block this model anyway)

        values = series.values.astype(float)
        best_order, best_aic = self.ORDER, np.inf
        for p in _P_Q_GRID:
            for q in _P_Q_GRID:
                if p == 0 and q == 0:
                    continue
                order = (p, _D_FIXED, q)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = SARIMAX(
                            values, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                        ).fit(disp=False, maxiter=50)
                    if np.isfinite(result.aic) and result.aic < best_aic:
                        best_aic, best_order = result.aic, order
                except Exception:
                    continue

        return {"order": best_order}

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "SARIMAModel":
        SARIMAX = _try_import_sarimax()
        if SARIMAX is None:
            raise ImportError("statsmodels is not installed — SARIMA is unavailable.")

        seasonal_order, min_obs = self._resolve_seasonal_order(frequency, len(series))
        if len(series) < min_obs:
            raise ValueError(
                f"SARIMA needs at least {min_obs} {frequency} observations — only {len(series)} available."
            )

        order = hyperparams.get("order") or self.ORDER

        self.series_ = series
        self.frequency_ = frequency
        self.seasonal_order_ = seasonal_order
        self.order_ = order
        values = series.values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels convergence/frequency warnings
            model = SARIMAX(
                values, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SARIMA must be fit() before predict().")
        return np.asarray(self._result.forecast(steps), dtype=float)
