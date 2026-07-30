"""
prophet.py
----------
Prophet: an additive trend + seasonality model from Meta, robust to
outliers and missing timestamps via automatic changepoint detection.
Recommended by Phase 1's model_selector for "Trend + Seasonal" and
"Highly Volatile" series.

Frequency-aware (Task 1): yearly_seasonality is decided from whether the
series spans at least 2 full years at its detected frequency (via
forecast.utils.seasonal_period_for_freq), and the future dataframe used
for prediction is built with the series' actual frequency alias
(forecast.utils.timestamp_freq_code).

Automatic tuning (production-grade pass, Task 5): suggest_hyperparameters()
uses a small internal chronological train/validation split (Prophet has no
AIC-like criterion — it is MAP-optimised, not maximum-likelihood over a
fixed parameter count), scoring (changepoint_prior_scale, seasonality_mode)
candidates by RMSE on the internal validation tail.

Stability pass (runtime fault-tolerance redesign): every code path that
actually touches the `prophet`/`cmdstanpy` packages now runs inside an
ISOLATED CHILD PROCESS via forecast.process_isolation.run_isolated(),
never directly in the caller's process. This is a direct response to a
confirmed runtime failure: on some Prophet 1.1.5 + cmdstanpy + Python 3.12
combinations, `Prophet(...)` construction can raise
`AttributeError: 'Prophet' object has no attribute 'stan_backend'`
immediately followed by a SEGMENTATION FAULT inside the native Stan/
cmdstanpy code — a hard process-level crash that no amount of Python
try/except can catch or prevent, because it terminates the OS process
itself rather than raising a catchable exception. See
process_isolation.py's module docstring for the full explanation and why
"spawn" (not "fork") is used.

Practical effect: fit() now performs a real isolated fit (+ a 1-step
smoke-test predict) rather than a cheap in-process call, and predict()
performs its own fresh isolated fit+predict every time it is invoked
(walk-forward validation calls this once per fold; the final forecast
calls it once more) — a live Prophet/Stan model object is never carried
across the process boundary, since such objects are not reliably
picklable and reintroducing one into the caller's process for a later
predict() call would simply move the crash risk, not remove it. This is
slower than the previous in-process implementation (each call now also
pays a fresh interpreter-startup + re-import cost on top of Prophet's own
already-substantial fit time), which is a deliberate trade: this pass's
stated priority is "never crash the app," not "minimise Prophet's own
wall-clock cost" — Prophet was already the slowest candidate model by a
wide margin before this change.

A confirmed NATIVE crash (segfault/abort/timeout — as opposed to an
ordinary caught Python exception, e.g. "not enough data") sets a
process-wide flag (_mark_confirmed_broken()) so is_available() reports
False immediately afterward, without another subprocess spawn attempt,
for the rest of this server process' lifetime. An ordinary per-series
exception does NOT set this flag — those are legitimate, potentially
transient, per-request failures and Prophet is retried normally for the
next series/request.
"""

import logging
import threading
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..process_isolation import run_isolated

logger = logging.getLogger(__name__)

# changepoint_prior_scale: Prophet's own default (0.05, conservative) vs.
# a much more flexible trend (0.5) — useful for volatile business series
# with real regime shifts. Kept to two values so the internal tuning
# split only needs up to 2 (changepoint) x 2 (seasonality mode) = 4
# Prophet fits, bounding tuning cost to roughly the same order as one
# walk-forward fold.
_CHANGEPOINT_PRIOR_GRID = (0.05, 0.5)


def _try_import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except ImportError:
        try:
            # older installs sometimes still expose the pre-rename package name
            from fbprophet import Prophet
            return Prophet
        except ImportError:
            return None
        except Exception:
            # Some broken installs raise something other than ImportError
            # while merely being imported (e.g. a native extension failing
            # to load) — treated the same as "not available", never
            # allowed to propagate out of an availability check.
            return None
    except Exception:
        return None


# ── process-wide "is Prophet actually usable here" cache ───────────────────
# Set once, the first time an isolated Prophet call fails in a way that
# looks like a native crash rather than an ordinary exception (see
# _looks_like_native_failure below). Protected by a lock since trainer.py
# trains candidates concurrently across threads.
_state_lock = threading.Lock()
_confirmed_broken = False


def _mark_confirmed_broken(reason: str) -> None:
    global _confirmed_broken
    with _state_lock:
        if not _confirmed_broken:
            logger.warning(
                "Prophet appears to be broken in this environment (%s) — marking it unavailable "
                "for the remainder of this run instead of retrying it on every series.", reason,
            )
        _confirmed_broken = True


def _is_confirmed_broken() -> bool:
    with _state_lock:
        return _confirmed_broken


def _looks_like_native_failure(message: str) -> bool:
    """True for failure messages produced BY process_isolation.run_isolated()
    itself (abnormal process exit / timeout / isolation-machinery failure)
    — i.e. something more serious than an ordinary Python exception raised
    from inside Prophet's own code (which is reported as that exception's
    own text, e.g. "AttributeError: ...", and does NOT indicate the
    install itself is broken)."""
    lowered = (message or "").lower()
    return (
        "terminated unexpectedly" in lowered
        or "timed out" in lowered
        or "process isolation failed" in lowered
    )


def _yearly_seasonality_flag(series_len: int, frequency: str) -> bool:
    from ..utils import seasonal_period_for_freq
    periods_per_year = seasonal_period_for_freq(frequency)
    if not periods_per_year or periods_per_year <= 1:
        return False
    return series_len >= periods_per_year * 2


# ── isolated worker functions ───────────────────────────────────────────────
# Module-level (not methods/closures) so they can be pickled and sent to
# the spawned child process. Every actual `import prophet` / `Prophet(...)`
# / `.fit()` / `.predict()` call in this file happens inside one of these
# two functions, and ONLY when running inside run_isolated()'s child
# process — never directly in the caller's process.

def _prophet_fit_predict(series: pd.Series, frequency: str, steps: int, hyperparams: dict):
    """Fit a fresh Prophet model on `series` and predict `steps` steps
    ahead, entirely inside an isolated child process. Returns a plain
    list of floats (never a live Prophet/Stan object — those are not
    reliably picklable back across the process boundary, and doing so
    would just relocate the crash risk rather than remove it)."""
    from prophet import Prophet
    from ..utils import timestamp_freq_code

    changepoint_prior_scale = hyperparams.get("changepoint_prior_scale", 0.05)
    seasonality_mode = hyperparams.get("seasonality_mode", "additive")
    values = series.values.astype(float)
    if seasonality_mode == "multiplicative" and not np.all(values > 0):
        # Defensive: multiplicative seasonality is undefined/unstable with
        # zero-or-negative values — never let a stale tuned hyperparameter
        # (tuned against different data) push this model into an invalid
        # configuration.
        seasonality_mode = "additive"

    df = pd.DataFrame({"ds": series.index.to_timestamp(), "y": values})
    model = Prophet(
        yearly_seasonality=_yearly_seasonality_flag(len(series), frequency),
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    model.fit(df)

    freq_alias = timestamp_freq_code(frequency)
    future = model.make_future_dataframe(periods=steps, freq=freq_alias, include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].values.astype(float).tolist()


def _prophet_tune(series: pd.Series, frequency: str):
    """Full internal train/validation-split tuning grid search (see module
    docstring), entirely inside an isolated child process. Returns the
    best {changepoint_prior_scale, seasonality_mode} dict, or {} if
    nothing scored. A crash on even the FIRST candidate can never reach
    the caller — the whole search either returns a plain dict or the
    isolation layer reports it as an ordinary failure."""
    from prophet import Prophet
    from ..utils import timestamp_freq_code

    n = len(series)
    val_size = max(1, min(3, n // 5))
    train_part, val_part = series.iloc[:-val_size], series.iloc[-val_size:]
    values = series.values.astype(float)
    freq_alias = timestamp_freq_code(frequency)

    candidates = [(cp, "additive") for cp in _CHANGEPOINT_PRIOR_GRID]
    if np.all(values > 0):
        candidates.append((_CHANGEPOINT_PRIOR_GRID[0], "multiplicative"))

    best_cfg, best_rmse = {}, np.inf
    for changepoint_prior, seasonality_mode in candidates:
        try:
            df = pd.DataFrame({
                "ds": train_part.index.to_timestamp(),
                "y": train_part.values.astype(float),
            })
            model = Prophet(
                yearly_seasonality=_yearly_seasonality_flag(len(train_part), frequency),
                changepoint_prior_scale=changepoint_prior,
                seasonality_mode=seasonality_mode,
            )
            model.fit(df)
            future = model.make_future_dataframe(periods=len(val_part), freq=freq_alias, include_history=False)
            preds = model.predict(future)["yhat"].values.astype(float)
            rmse = float(np.sqrt(np.mean((val_part.values.astype(float) - preds) ** 2)))
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_cfg = {"changepoint_prior_scale": changepoint_prior, "seasonality_mode": seasonality_mode}
        except Exception:
            # One candidate failing (an ordinary exception) does not mean
            # the whole tuning search should abort — the remaining
            # candidates are still worth trying.
            continue

    return best_cfg


class ProphetModel(BaseForecastModel):
    name = "Prophet"

    MIN_OBS = 4

    @classmethod
    def is_available(cls) -> bool:
        if _is_confirmed_broken():
            return False
        return _try_import_prophet() is not None

    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        if len(series) < self.MIN_OBS:
            return False, (
                f"Prophet needs at least {self.MIN_OBS} observations — only {len(series)} available — skipped."
            )
        return True, None

    def suggest_hyperparameters(self, series: pd.Series, frequency: str = "monthly") -> dict:
        """Task 5 — see module docstring for the internal-split tuning
        approach. Runs entirely in an isolated child process (see
        _prophet_tune); any failure — ordinary or a native crash — simply
        results in {} (tuning skipped, fit() falls back to Prophet's own
        defaults), never an exception raised out of this method."""
        if _is_confirmed_broken():
            return {}
        if _try_import_prophet() is None or len(series) < self.MIN_OBS + 2:
            return {}

        ok, result = run_isolated(_prophet_tune, args=(series, frequency))
        if not ok:
            if _looks_like_native_failure(result):
                _mark_confirmed_broken(result)
            else:
                logger.debug("Prophet hyperparameter tuning failed, using defaults (%s)", result)
            return {}
        return result or {}

    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "ProphetModel":
        if len(series) < self.MIN_OBS:
            raise ValueError(f"Prophet needs at least {self.MIN_OBS} observations.")
        if _is_confirmed_broken():
            raise RuntimeError(
                "Prophet is unavailable in this environment — a prior attempt crashed its Stan backend."
            )
        if _try_import_prophet() is None:
            raise ImportError("prophet is not installed — Prophet is unavailable.")

        # Isolated smoke-test: a real fit + a single 1-step prediction, so
        # a fundamentally broken Prophet install (including one that would
        # segfault) is caught right here — exactly where trainer.py's
        # existing per-candidate try/except already treats a fit()
        # failure as "this candidate failed to train" — rather than only
        # surfacing later, when a forecast is actually requested from a
        # model that was incorrectly marked as successfully trained.
        ok, result = run_isolated(_prophet_fit_predict, args=(series, frequency, 1, dict(hyperparams)))
        if not ok:
            if _looks_like_native_failure(result):
                _mark_confirmed_broken(result)
            raise RuntimeError(f"Prophet failed to train: {result}")

        self.series_ = series
        self.frequency_ = frequency
        self.hyperparameters_ = dict(hyperparams)
        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Prophet must be fit() before predict().")
        if _is_confirmed_broken():
            raise RuntimeError(
                "Prophet is unavailable in this environment — a prior attempt crashed its Stan backend."
            )

        ok, result = run_isolated(
            _prophet_fit_predict, args=(self.series_, self.frequency_, steps, self.hyperparameters_),
        )
        if not ok:
            if _looks_like_native_failure(result):
                _mark_confirmed_broken(result)
            raise RuntimeError(f"Prophet failed to predict: {result}")
        return np.asarray(result, dtype=float)
