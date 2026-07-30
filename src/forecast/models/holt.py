"""
holt.py
-------
Holt Linear Trend (double exponential smoothing): tracks both a level and
a trend, so the forecast is a straight line continuing the most recently
observed slope. Implemented directly in numpy (no statsmodels dependency)
so this baseline is always available. Recommended by Phase 1's
model_selector for "Trending" series.

alpha (level smoothing) and beta (trend smoothing) are chosen by a small
grid search minimising in-sample squared error, same rationale as SES.
"""

import numpy as np
import pandas as pd

from .base import BaseForecastModel

_ALPHA_GRID = [0.2, 0.4, 0.6, 0.8]
_BETA_GRID  = [0.05, 0.2, 0.4]


def _holt_fit(values: np.ndarray, alpha: float, beta: float):
    n = len(values)
    levels = np.empty(n)
    trends = np.empty(n)
    levels[0] = values[0]
    trends[0] = values[1] - values[0]
    for t in range(1, n):
        prev_level, prev_trend = levels[t - 1], trends[t - 1]
        levels[t] = alpha * values[t] + (1 - alpha) * (prev_level + prev_trend)
        trends[t] = beta * (levels[t] - prev_level) + (1 - beta) * prev_trend
    return levels, trends


class HoltLinearModel(BaseForecastModel):
    name = "Holt Linear"

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "HoltLinearModel":
        # Already self-tunes alpha/beta via its own grid search below on
        # every fit() call — accepts **hyperparams for a uniform call
        # signature, but has nothing external left to tune.
        if len(series) == 0:
            raise ValueError("Holt Linear requires at least 1 observation.")
        self.series_ = series
        self.frequency_ = frequency
        values = series.values.astype(float)

        if len(values) < 2:
            # Not enough points to estimate a trend — degrade to a flat
            # level, same behaviour as Naive, rather than raising.
            self.level_, self.trend_ = float(values[0]), 0.0
            self.fitted = True
            return self

        best_params, best_sse = (_ALPHA_GRID[0], _BETA_GRID[0]), np.inf
        for alpha in _ALPHA_GRID:
            for beta in _BETA_GRID:
                levels, trends = _holt_fit(values, alpha, beta)
                one_step = levels[:-1] + trends[:-1]
                sse = float(np.sum((values[1:] - one_step) ** 2))
                if sse < best_sse:
                    best_sse, best_params = sse, (alpha, beta)

        levels, trends = _holt_fit(values, *best_params)
        self.level_, self.trend_ = float(levels[-1]), float(trends[-1])
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Holt Linear must be fit() before predict().")
        horizons = np.arange(1, steps + 1)
        return self.level_ + horizons * self.trend_
