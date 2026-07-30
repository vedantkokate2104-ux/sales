"""
naive.py
--------
Naive Forecast: the simplest possible baseline. Every future period is
predicted to equal the last observed value. No external dependency —
always available, and used as one of the trainer's guaranteed fallback
candidates so there is always at least one model to fall back on.
"""

import numpy as np
import pandas as pd

from .base import BaseForecastModel


class NaiveModel(BaseForecastModel):
    name = "Naive"

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "NaiveModel":
        # No hyperparameters to tune — accepts **hyperparams only so
        # forecast.trainer can call every model uniformly with whatever
        # suggest_hyperparameters() returned (here, always {}).
        if len(series) == 0:
            raise ValueError("Naive requires at least 1 observation.")
        self.series_ = series
        self.frequency_ = frequency
        self.last_value_ = float(series.iloc[-1])
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Naive must be fit() before predict().")
        return np.full(steps, self.last_value_, dtype=float)
