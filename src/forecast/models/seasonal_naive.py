"""
seasonal_naive.py
------------------
Seasonal Naive: every future period repeats the value observed one full
seasonal cycle earlier.

    monthly    forecast[t] = actual[t - 12]
    weekly     forecast[t] = actual[t - 52]
    quarterly  forecast[t] = actual[t -  4]
    daily      forecast[t] = actual[t -  7]

This is the mandatory benchmark for seasonal business data, and its
absence was a real gap: the library's three guaranteed baselines (Naive,
Moving Average, Simple Exponential Smoothing) are all FLAT forecasters,
so nothing established whether a series' seasonal structure was
exploitable at all. A sophisticated seasonal model that cannot beat
"repeat last year's same month" has not demonstrated anything, and until
now there was no way to notice that.

Horizons longer than one cycle wrap around and repeat the cycle again,
which is the standard definition — at that distance the model is
asserting nothing more than "the last observed cycle recurs".

No external dependency, so this is always available. It is only offered
when the series actually covers at least two full cycles: with less than
that there is no repeated cycle to copy, and check_requirements() blocks
it rather than letting it degrade into a lagged Naive.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..utils import seasonal_period_for_freq

# Two full cycles: one to learn the shape, one to confirm it repeats.
MIN_CYCLES = 2


class SeasonalNaiveModel(BaseForecastModel):
    name = "Seasonal Naive"

    def _period(self, frequency: str) -> Optional[int]:
        period = seasonal_period_for_freq(frequency)
        return period if period and period > 1 else None

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        period = self._period(frequency)
        if period is None:
            return False, (
                f"Seasonal Naive needs a repeating seasonal cycle, but '{frequency}' data has no "
                f"sub-annual cycle to repeat — skipped."
            )
        needed = period * MIN_CYCLES
        if len(series) < needed:
            return False, (
                f"Seasonal Naive needs at least {needed} {frequency} observations "
                f"({MIN_CYCLES} full seasonal cycles of {period}) — only {len(series)} available — skipped."
            )
        return True, None

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "SeasonalNaiveModel":
        # Nothing to estimate — the "fit" is simply retaining the most
        # recent complete cycle. **hyperparams is accepted only so the
        # trainer can call every model uniformly.
        period = self._period(frequency)
        if period is None:
            raise ValueError(f"Seasonal Naive needs a sub-annual seasonal cycle — '{frequency}' data has none.")
        needed = period * MIN_CYCLES
        if len(series) < needed:
            raise ValueError(
                f"Seasonal Naive needs at least {needed} {frequency} observations "
                f"({MIN_CYCLES} full seasonal cycles) — only {len(series)} available."
            )

        self.series_ = series
        self.frequency_ = frequency
        self.seasonal_period_ = period
        self.last_cycle_ = series.iloc[-period:].to_numpy(dtype=float)
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Seasonal Naive must be fit() before predict().")
        # Repeat the most recent complete cycle, wrapping for horizons
        # longer than the cycle itself.
        idx = np.arange(steps) % self.seasonal_period_
        return self.last_cycle_[idx].astype(float)
