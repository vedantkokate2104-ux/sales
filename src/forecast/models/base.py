"""
base.py
-------
Common interface for every model in the Forecast Model Library.

Every subclass implements fit() and predict(); evaluate() and forecast()
are provided here once so metric computation and future-period labelling
are never duplicated across the seven model implementations.

Frequency preservation (Phase E): forecast.trainer now aggregates the raw
data at whatever frequency was actually detected (daily / weekly /
monthly / quarterly / yearly — see forecast/utils.py), not always monthly.
forecast() below needed NO code change for this: it labels each future
period as `self.series_.index[-1] + (i + 1)`, and pandas' PeriodIndex
arithmetic is frequency-agnostic — adding an integer to a Period always
advances by one unit of whatever freq that PeriodIndex already has,
whether that's "D", "W", "M", "Q", or "Y". Only this docstring's previous
"always monthly" assumption needed updating.

fit() now takes an optional `frequency` argument (Task 1/4) so a model can
size its seasonal component, or judge its own suitability, relative to the
data's actual cadence instead of an assumed monthly one. It defaults to
"monthly" for backward compatibility with any caller that does not pass it.

Automatic hyperparameter tuning (production-grade pass, Task 5): fit() now
also accepts arbitrary `**hyperparams`, and a new suggest_hyperparameters()
hook lets a model search its own hyperparameter space ONCE per forecast
run (see that method's docstring for exactly when/how it is called and
why that keeps tuning both cheap and leakage-free). Models with nothing
worth tuning simply inherit the base no-op implementation and ignore any
hyperparams they're not expecting.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from forecast.evaluator import compute_metrics


class BaseForecastModel:
    """Abstract base — subclasses must set `name` and implement fit/predict."""

    name: str = "Base"

    def __init__(self):
        self.fitted = False
        self.series_: Optional[pd.Series] = None
        self.frequency_: str = "monthly"

    # ── dependency check (overridden by models with an optional import) ──
    @classmethod
    def is_available(cls) -> bool:
        """True if this model's dependencies are installed. Pure-numpy
        models (Moving Average, Naive, SES, Holt Linear) are always
        available; statsmodels/Prophet-backed models override this."""
        return True

    # ── suitability check (Task 4) ──────────────────────────────────────────
    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        """
        Lightweight, fast pre-flight check of whether this model is a
        reasonable fit for `series` at the given detected `frequency` —
        distinct from is_available(), which only checks whether the
        model's dependency is installed.

        Returns (ok, message):
          - ok=False, message=<business-friendly reason>  — hard block;
            the trainer skips this model entirely (used where fitting is
            genuinely not statistically possible, e.g. SARIMA without
            enough seasonal cycles).
          - ok=True,  message=<business-friendly caution>  — soft warning;
            the trainer still trains this model but surfaces the caution
            (used for heuristic guidance, e.g. Simple Exponential
            Smoothing on a strongly trending series).
          - ok=True,  message=None                          — no concerns.

        Base implementation always allows training; models with genuine
        prerequisites override this.
        """
        return True, None

    # ── hyperparameter tuning hook (Task 5) ─────────────────────────────────
    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """
        One-time, lightweight hyperparameter search performed ONCE per
        candidate model per forecast run — NOT once per walk-forward
        fold. forecast.trainer calls this a single time, on the same
        chronological prefix its first walk-forward fold would use, and
        then reuses the returned hyperparameters UNCHANGED for every
        fold and the final full-series fit. This is a deliberate,
        documented trade-off: true nested per-fold tuning would be more
        rigorous but multiplies training cost by (grid size x fold
        count) for every candidate model on every forecast request,
        which is not practical for an interactive dashboard. Tuning once
        on a chronological prefix, then holding the choice fixed, is
        standard practice for lightweight production pipelines (as
        opposed to research benchmarking) and never lets a later fold's
        hyperparameters be chosen using data that fold itself would not
        yet have.

        Returns a dict of keyword arguments to pass into fit(). Base
        implementation returns {} (nothing to tune). Models with real
        hyperparameters (SARIMA, Holt-Winters, Prophet, XGBoost) override
        this; models that already self-tune internally on every fit()
        call (SES, Holt Linear — see their own alpha/beta grid search)
        or have no meaningful hyperparameter at all (Naive, Moving
        Average) correctly inherit this no-op.
        """
        return {}

    # ── required overrides ────────────────────────────────────────────────
    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, steps: int) -> np.ndarray:
        raise NotImplementedError

    # ── shared behaviour (do not override) ────────────────────────────────
    def evaluate(self, y_true, y_pred) -> dict:
        """Compute MAE / RMSE / MAPE / SMAPE via the shared evaluator so
        every model reports metrics the same way."""
        return compute_metrics(y_true, y_pred)

    def forecast(self, horizon: int) -> list:
        """
        Convenience wrapper: predict() the next `horizon` steps and label
        each with the next period (at whatever frequency this model was
        fitted on) after the series this model was fitted on. Returns a
        list of (period_label: str, value: float).
        """
        if not self.fitted or self.series_ is None:
            raise RuntimeError(f"{self.name} must be fit() before forecast().")
        values = self.predict(horizon)
        last_period = self.series_.index[-1]
        return [
            (str(last_period + (i + 1)), float(max(0.0, values[i])))
            for i in range(horizon)
        ]
