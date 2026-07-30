"""
moving_average.py
------------------
Moving Average: forecast is the mean of the last `window` observations,
held constant for every future period. No external dependency — always
available. Recommended by Phase 1's model_selector for Short History and
New Product series, and used as a guaranteed fallback candidate.
"""

import numpy as np
import pandas as pd

from .base import BaseForecastModel


class MovingAverageModel(BaseForecastModel):
    name = "Moving Average"

    def __init__(self, window: int = 3):
        super().__init__()
        self.window = window

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "MovingAverageModel":
        # No hyperparameters to tune (window is a fixed constructor
        # argument) — accepts **hyperparams only for a uniform fit()
        # call signature across every model in the registry.
        if len(series) == 0:
            raise ValueError("Moving Average requires at least 1 observation.")
        self.series_ = series
        self.frequency_ = frequency
        effective_window = min(self.window, len(series))
        self.level_ = float(series.iloc[-effective_window:].mean())
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Moving Average must be fit() before predict().")
        return np.full(steps, self.level_, dtype=float)
