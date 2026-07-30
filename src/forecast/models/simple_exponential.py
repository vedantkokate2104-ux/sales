"""
simple_exponential.py
----------------------
Simple Exponential Smoothing (SES): a weighted average of past
observations where more recent points count more. Implemented directly in
numpy (no statsmodels dependency) so this baseline is always available,
matching the project's existing convention of a rule-based/manual fallback
staying active when an optional ML dependency is missing (see
embedding_mapper.py's fallback to the rule engine in Phase 1).

The smoothing constant alpha is chosen by a small grid search minimising
in-sample squared error — cheap for the short series this project deals
with, and avoids a hardcoded alpha that might suit one dataset poorly.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import trend_strength

_ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

# SES has no trend component at all — above this trend-strength score
# (see utils.trend_strength: 0 = no linear trend, 1 = perfectly linear)
# it will systematically lag a series that is clearly trending up or down.
# This is a soft caution, not a hard block: SES is a GUARANTEED_CANDIDATES
# baseline in trainer.py and must remain trainable even on trending data —
# the warning just explains why it may rank poorly against Holt Linear /
# Holt-Winters on such a series.
STRONG_TREND_THRESHOLD = 0.7


def _ses_fit(values: np.ndarray, alpha: float) -> np.ndarray:
    """Return the level at every time step for a given alpha."""
    levels = np.empty(len(values))
    levels[0] = values[0]
    for t in range(1, len(values)):
        levels[t] = alpha * values[t] + (1 - alpha) * levels[t - 1]
    return levels


class SimpleExponentialSmoothingModel(BaseForecastModel):
    name = "Simple Exponential Smoothing"

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        """Soft caution only (see STRONG_TREND_THRESHOLD above) — always
        returns ok=True so this guaranteed baseline never gets skipped."""
        if len(series) < 3:
            return True, None
        strength = trend_strength(series.values.astype(float))
        if strength >= STRONG_TREND_THRESHOLD:
            return True, (
                "Simple Exponential Smoothing does not model trend and may lag behind "
                "this series' strong upward/downward movement — Holt Linear or "
                "Holt-Winters are usually a better fit for trending data."
            )
        return True, None

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "SimpleExponentialSmoothingModel":
        # Already self-tunes alpha via its own grid search below on every
        # fit() call — accepts **hyperparams for a uniform call
        # signature, but has nothing external left to tune.
        if len(series) == 0:
            raise ValueError("Simple Exponential Smoothing requires at least 1 observation.")
        self.series_ = series
        self.frequency_ = frequency
        values = series.values.astype(float)

        if len(values) == 1:
            self.level_ = float(values[0])
            self.fitted = True
            return self

        best_alpha, best_sse = _ALPHA_GRID[0], np.inf
        for alpha in _ALPHA_GRID:
            levels = _ses_fit(values, alpha)
            # one-step-ahead in-sample error, standard SES fit criterion
            sse = float(np.sum((values[1:] - levels[:-1]) ** 2))
            if sse < best_sse:
                best_sse, best_alpha = sse, alpha

        self.alpha_ = best_alpha
        self.level_ = float(_ses_fit(values, best_alpha)[-1])
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Simple Exponential Smoothing must be fit() before predict().")
        return np.full(steps, self.level_, dtype=float)
