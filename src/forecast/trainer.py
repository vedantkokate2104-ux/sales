"""
trainer.py
----------
Training Pipeline (Phase 2, Step 2): builds the historical series at its
DETECTED frequency, validates it with expanding-window walk-forward
validation WITHOUT shuffling (time series order is always preserved), and
trains every candidate model recommended by Phase 1's model_selector.py
against it.

Phase E changes (scientific-correctness pass):
  - build_monthly_series() (always-monthly) was generalised into
    build_series(dates, values, frequency), which aggregates at whatever
    frequency the data actually has (daily/weekly/monthly/quarterly/
    yearly), using the frequency vocabulary centralised in forecast/utils.py.
    build_monthly_series() is kept as a thin monthly-only wrapper around
    build_series() so nothing that already imports it breaks.
  - time_series_split() (single hold-out tail) was replaced by
    walk_forward_validate() (expanding-window, one-step-ahead, repeated
    across the whole recent history) — see its docstring for the full
    rationale.
  - Each candidate now gets a pre-flight check_requirements() call (Task 4)
    before training is attempted, and raised exceptions are translated to
    business-friendly text before being surfaced in `warnings`, while the
    real exception always still goes to the server log via
    logger.exception() (Task 6).

Does not modify Phase 1's analyzer.py — analyzer.py already computes the
dataset's frequency (forecast.types.SeriesStats.frequency, via
utils.infer_frequency); forecast_engine.py passes that same value in here
so both the "Dataset Analysis" panel and the trainer always agree on what
frequency the data is at, instead of each recomputing it independently.

Production-quality pass #1 (ranking/confidence/ensemble review):
walk_forward_validate() also returns a per-model fold-to-fold
"stability" score, and every ModelEvalResult built below is tagged with
validation_folds / training_observations / a short machine-readable
`status` string. This is what lets selector.py rank models on more than
MAPE and compute a confidence score from more than one signal.

Production-quality pass #2 (preprocessing + auto-tuning):
  - Raw (date, revenue) rows now go through forecast.preprocessing
    .clean_transactions() (Task 2) before aggregation, and the
    aggregated series goes through .cap_outliers() +
    .interpolate_short_gaps() (Tasks 2, 11) before any model sees it.
    Forecasting never runs on raw, uncorrected data any more.
  - Each candidate model gets ONE chance to tune its own hyperparameters
    (Task 5), via its suggest_hyperparameters() hook, on a shared
    chronological prefix (the same one walk-forward's first fold would
    use) computed once per run — not once per fold, which would be far
    more expensive for negligible extra rigor at this dataset scale. The
    resulting hyperparameters are then reused, unchanged, for every
    walk-forward fold AND the final full-series fit, so the accuracy
    metrics reported always describe the exact configuration that goes
    on to produce the forecast.

train_candidates()'s return signature grew a 5th value, `preprocessing_info`
(counts of what preprocessing corrected, for the confidence calculation's
data-quality signal) — its only caller, forecast_engine.py, was updated
to receive it. Every other existing field, function, and control-flow
branch in this file is unchanged.

Production-quality pass #3 (Task 16 — performance): each candidate's
tune -> walk-forward-validate -> final-fit pipeline (previously one long
sequential loop body) is now factored into `_train_one_candidate()` and
run for every pre-checked candidate concurrently via a small
ThreadPoolExecutor, since candidates share no mutable state at the PYTHON
level (each builds its own model instances; `series`/`tuning_prefix` are
read-only inputs). Results are merged back into the original candidate
order afterward, so output is unchanged in content and order.

Runtime stability audit: a live segmentation fault was reported with
worker threads inside `concurrent.futures.thread._worker` on the stack,
plus `resource_tracker: leaked semaphore objects` warnings at shutdown, on
macOS (Apple Silicon). A project-wide search confirmed this file's
ThreadPoolExecutor above is the ONLY thread/process pool in the entire
codebase — every SARIMA/Holt-Winters (statsmodels -> scipy -> LAPACK/BLAS)
and XGBoost (its own internal OpenMP pool, never given an explicit
thread-count limit) fit can therefore run CONCURRENTLY across several
threads, entering native, C-level thread-pool/BLAS code from multiple
Python threads at once — a documented crash-prone pattern on macOS (see
CHANGELOG for the full library-by-library reasoning). Rather than removing
parallel execution, candidate training now runs behind a module-level
`ENABLE_PARALLEL_FORECASTING` flag: sequential (`_train_sequential()`,
calling the exact same `_train_one_candidate()` one candidate at a time —
identical logic and output, just no concurrent native-library invocation)
is now the default while stability is confirmed in the field; the
ThreadPoolExecutor path is fully preserved, selectable by setting the flag
to `True`. Separately, `process_isolation.py` (used by Prophet) now
explicitly closes every `multiprocessing.Queue`/`Process` it creates
instead of leaving them for garbage collection — the direct fix for the
observed leaked-semaphore warning.

Forecasting-quality pass (this pass — multi-step-aware validation):
  - walk_forward_validate() previously ALWAYS evaluated exactly one step
    ahead per fold (`probe.predict(1)`), regardless of how the resulting
    model would actually be used later. But forecast_engine.py's
    run_auto_forecast() defaults to, and is frequently called with, a
    multi-step horizon (horizon=3 by default, and callers can request
    more) — a model's 1-step-ahead accuracy and its 3-step-ahead accuracy
    are NOT the same thing: a recursive multi-step forecaster (this
    project's XGBoost) accumulates error across steps in a way a 1-step
    evaluation never exercises, and even a model with a native multi-step
    API (SARIMA/Holt-Winters/Prophet/Holt/SES) can have its trend or
    seasonal-phase estimate drift further from actuals the further out it
    forecasts. Selecting the "best" model using only 1-step error was
    therefore evaluating something subtly different from what deployment
    actually asks the model to do.
  - Each walk-forward fold now evaluates `_resolve_eval_horizon(...)`
    steps ahead (a small, fixed cap of 3 — the same default horizon
    run_auto_forecast() itself uses — capped further so at least a
    couple of folds remain possible) instead of always exactly 1,
    calling `probe.predict(eval_horizon)` once per fold and comparing
    against that many real future points, still never touching anything
    at or after the fold's own training cutoff (no leakage — see that
    function's docstring). For short series where the data cannot
    support this without starving the fold count, `_resolve_eval_horizon`
    degrades to exactly 1 — IDENTICAL to this function's prior behaviour
    — so short/thin datasets are not penalised with fewer, noisier folds
    just to exercise a longer horizon they cannot support.
  - Every existing guarantee is unchanged: still expanding-window
    (`train = series.iloc[:train_end]` grows by one point per fold), still
    leakage-free (each fold's prediction never sees series.iloc[train_end]
    or anything after it, regardless of how many steps ahead it predicts),
    still pools every fold's (actual, predicted) pairs before computing
    metrics once (now across `eval_horizon` points per fold instead of
    one), and `stability` still means the same thing — fold-to-fold
    consistency of a SINGLE per-fold error number, now that per-fold
    number is the fold's own mean absolute error across its evaluation
    window rather than a single point's absolute error, so it still
    answers "does this model perform consistently across different
    forecast origins" rather than conflating that with within-fold,
    horizon-step-to-horizon-step degradation (a different question).
    walk_forward_validate()'s signature and return type
    (Tuple[Optional[dict], int, Optional[float]]) are completely
    unchanged, as is train_candidates()'s signature and every other
    function in this file — every existing caller keeps working exactly
    as before, just against a validation signal that better reflects how
    the resulting model is actually used.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import _clean_numeric
from .backtest import (
    BacktestResult,
    EnsembleBacktest,
    FoldSpec,
    ValidationPlan,
    backtest_model,
    evaluate_ensemble,
    plan_validation,
    prepare_training_window,
)
from .evaluator import compute_metrics
from .grain import build_forecast_series
from .models import MODEL_REGISTRY
from .preprocessing import cap_outliers, clean_transactions, interpolate_short_gaps, parse_dates_deterministic
from .utils import effective_training_frequency, infer_frequency, period_freq_code, seasonal_period_for_freq

logger = logging.getLogger(__name__)

# Always evaluated in addition to whatever Phase 1 recommends, so there is
# always at least one (in fact three) trainable baseline — this is what
# keeps series types like "Intermittent Demand" or "Highly Volatile"
# (whose Phase 1 top picks, Croston/XGBoost, are not yet implemented in
# this library) from ending up with zero candidates to compare.
#
# Phase 2: "Seasonal Naive" joins the list. The other three are all FLAT
# forecasters, so before this there was no benchmark answering "is this
# series' seasonal structure exploitable at all" — a seasonal model that
# cannot beat "repeat the same month last year" has demonstrated nothing.
# Seasonal Naive blocks itself via check_requirements() when the series
# does not cover two full cycles, so adding it here is safe for short
# series.
GUARANTEED_CANDIDATES = ["Naive", "Seasonal Naive", "Moving Average", "Simple Exponential Smoothing"]

MIN_VALIDATION_POINTS = 1
MAX_VALIDATION_POINTS = 6

# Walk-forward validation needs at least this many successful folds before
# its pooled metrics are considered meaningful enough to report — below
# this, a model is still trained and forecast on the full series, just
# without a reported accuracy score (same "fit without validation metrics"
# degradation the old hold-out split used for very short series).
MIN_WALKFORWARD_FOLDS = 1

# Below this many successful folds, fold-to-fold variability is too noisy
# to summarise into a single stability score (variance of 0-1 samples is
# not meaningful) — stability comes back as None rather than a misleading
# number computed from almost nothing.
MIN_FOLDS_FOR_STABILITY = 2

_STABILITY_EPS = 1e-6

# Forecasting-quality pass — how many steps ahead each walk-forward fold
# evaluates, at most. Matches run_auto_forecast()'s own default `horizon`
# (see forecast_engine.py) so validation is scored against the SAME depth
# of forecast the engine typically actually produces, rather than only
# ever the easiest possible case (one step ahead). Deliberately a small,
# fixed internal constant rather than the caller's actual requested
# horizon: train_candidates()/walk_forward_validate() are called before
# forecast_engine.py knows (or needs to know) what horizon the eventual
# forecast() call will use, and this file's public functions keep their
# exact existing signatures either way.
_MULTISTEP_EVAL_HORIZON = 3


def _resolve_eval_horizon(n: int, min_train_size: int) -> int:
    """
    DEPRECATED (Phase 2) — retained only so any existing caller keeps
    importing successfully. The validation horizon is now decided by
    forecast.backtest.plan_validation(), which holds it equal to the
    PRODUCTION horizon and reduces it only as a last resort, reporting
    the mismatch via `ValidationPlan.degraded` instead of silently
    dropping to 1-step evaluation as this function did.
    """
    available = n - min_train_size
    if available < 2:
        return 1
    max_supportable = max(1, available // 2)
    return max(1, min(_MULTISTEP_EVAL_HORIZON, max_supportable))


# Task 16 — how many candidate models' tuning+validation+fit run
# concurrently. Each candidate is independent (its own model instance, its
# own hyperparameter search, its own walk-forward loop — none of them
# mutate `series`/`tuning_prefix`, which are only ever read), so this is
# safe as plain threads without any locking. Kept modest (rather than one
# worker per candidate) because the heaviest candidates (Prophet's Stan
# backend, SARIMA/Holt-Winters' statsmodels optimisers) are also the most
# CPU/memory-hungry per fit — running all of them fully concurrently on a
# typical interactive-dashboard host would trade wall-clock time for
# memory pressure and CPU contention rather than a clean speedup.
MAX_PARALLEL_CANDIDATES = 4

# Runtime stability audit — configuration flag, not a removal of parallel
# support. See CHANGELOG for the full investigation: a live segfault was
# observed with worker threads inside concurrent.futures.thread._worker
# on the stack, plus `resource_tracker: leaked semaphore objects` warnings
# at shutdown, on macOS (Apple Silicon). This project's ONLY
# ThreadPoolExecutor is the one below — every SARIMA/Holt-Winters
# (statsmodels -> scipy -> LAPACK/BLAS) and XGBoost (its own internal
# OpenMP thread pool, never given an explicit nthread limit) fit
# potentially runs CONCURRENTLY, across up to MAX_PARALLEL_CANDIDATES
# threads, entering native, C-level thread-pool/BLAS code from multiple
# Python threads at once. macOS's Accelerate BLAS backend and bundled
# libomp (used by XGBoost's macOS wheels) both have documented
# thread-safety edge cases under exactly this kind of concurrent,
# multi-threaded native invocation — see CHANGELOG Step 2/6 for the full
# reasoning. Defaulting this to False takes the ThreadPoolExecutor
# entirely out of the picture (every candidate trains sequentially, one
# native call at a time, in the SAME function — _train_one_candidate() —
# either way, so behaviour/output is identical) while stability is being
# confirmed in the field. Flip to True to re-enable the concurrent path;
# it is fully preserved, not removed.
ENABLE_PARALLEL_FORECASTING = False


def build_series(dates: pd.Series, values: pd.Series, frequency: Optional[str] = None) -> pd.Series:
    """
    Aggregate (date, revenue) observations into a total-per-period series
    indexed by a gapless PeriodIndex, at the given forecast grain.

    Phase 1, Step 5 — this now delegates to
    forecast.grain.build_forecast_series() rather than doing its own
    `reindex(full_range, fill_value=0.0)`. That old one-liner asserted
    that a period with no invoice had zero demand, which conflates three
    distinguishable things (an observed zero total, an unobserved period,
    and a recording gap). grain.build_forecast_series() keeps them apart,
    tracks which periods were actually observed, and chooses how to fill
    the rest from the series' own density instead of assuming zero
    everywhere — see that module's docstring for the full rationale.

    This function keeps its exact original signature and return type (a
    plain per-period pd.Series), so every existing caller is unaffected.
    Callers that need the observed/missing breakdown should call
    grain.build_forecast_series() directly and read the SeriesConstruction.

    `frequency` should be the grain chosen by grain.select_forecast_grain()
    (forecast_engine.py passes it in). When omitted it falls back to
    utils.infer_frequency() — transaction arrival cadence — purely so this
    function remains usable standalone; that is explicitly NOT how the
    engine picks a grain any more.
    """
    if frequency is None:
        parsed = parse_dates_deterministic(dates)
        frequency = infer_frequency(parsed[parsed.notna()])

    return build_forecast_series(dates, values, frequency).series


def build_monthly_series(dates: pd.Series, values: pd.Series) -> pd.Series:
    """Backward-compatible monthly-only wrapper around build_series(), kept
    for any existing caller that explicitly wants monthly aggregation
    regardless of the data's detected frequency."""
    return build_series(dates, values, frequency="monthly")


def _default_min_train_size(n: int) -> int:
    """
    Shared sizing rule for "how much of the series is the first
    walk-forward fold's training window" — used both by
    walk_forward_validate() itself and by train_candidates() to carve out
    the SAME prefix for one-time hyperparameter tuning (Task 5), so
    tuning never sees more data than the earliest fold would have had.
    Extracted into its own function so the two call sites can never
    silently drift apart (Task 15 — no duplicated logic).
    """
    val_span = max(MIN_VALIDATION_POINTS, min(MAX_VALIDATION_POINTS, n // 5))
    return max(2, n - val_span)


def walk_forward_validate(
    series: pd.Series,
    model_cls,
    frequency: str,
    min_train_size: Optional[int] = None,
    hyperparams: Optional[dict] = None,
) -> Tuple[Optional[dict], int, Optional[float]]:
    """
    Expanding-window walk-forward validation for a single candidate model
    class — simulates how the model is actually used in production,
    unlike a single fixed hold-out split:

        Train on series[:i]  ->  predict the next `eval_horizon` steps
        ->  compare against series[i:i+eval_horizon]  ->  expand i by 1
        ->  repeat until the series is exhausted.

    Order is never shuffled; each fold's training window is a strict
    chronological prefix of the one before it, and every prediction comes
    from a FRESH model instance fit only on that fold's own prefix (a
    fold's fit never reuses another fold's fitted state, and never sees
    series.iloc[train_end] or anything after it, no matter how many steps
    ahead it is asked to predict) — this is what makes the walk-forward
    loop leakage-free by construction.

    `eval_horizon` (see _resolve_eval_horizon()) is usually more than one
    step: forecast_engine.py's actual deployment forecasts are multi-step
    (run_auto_forecast() defaults to horizon=3), and a model's one-step
    accuracy is not necessarily representative of its multi-step accuracy
    — a recursive forecaster (this project's XGBoost) compounds error
    across steps in a way a single-step evaluation never exercises, and
    even models with a native multi-step API can drift further from
    actuals the further out they forecast. Evaluating multiple steps per
    fold, when the series has enough data to support it, scores models on
    something much closer to what they are actually asked to do in
    production. Short series automatically fall back to exactly one step
    per fold (see _resolve_eval_horizon()) — identical to this function's
    original behaviour — so this never costs a thin dataset any folds it
    would otherwise have had.

    All folds' (actual, predicted) pairs — now up to `eval_horizon` pairs
    per fold — are pooled and every metric (MAE/RMSE/MAPE/SMAPE/WAPE) is
    computed ONCE over the pooled arrays, rather than computing a metric
    per fold and then averaging the per-fold metrics. For a single-step
    fold, a per-fold RMSE degenerates to the same value as a per-fold MAE
    (sqrt of one squared term is just the absolute value), so averaging
    per-fold RMSEs would silently produce MAE's number under RMSE's name.
    Pooling first and computing once preserves the actual meaning of each
    metric while still reflecting performance averaged across every
    validation fold (and, now, every step within every fold).

    A fold whose model fails to fit, or fails to produce `eval_horizon`
    predictions, on that (early, shorter) training window — e.g. SARIMA
    cannot fit until 2 full seasonal cycles exist — is simply skipped
    rather than aborting the whole validation; the caller decides whether
    too few folds succeeded to trust the result.

    A third return value, `stability`, is a 0-1 score describing how
    consistent this model's per-fold error was across folds: 1 -
    min(1, coefficient_of_variation_of_fold_errors). Each fold's own error
    is now the MEAN absolute error across that fold's evaluation window
    (rather than a single point's absolute error) — this keeps `stability`
    answering the same question it always did, "does this model perform
    consistently across different training-window/forecast-origin
    choices," rather than conflating that with a different question
    (whether error grows within a single fold's own multi-step horizon,
    which is instead reflected in the pooled accuracy metrics themselves).
    A model whose error swings wildly fold to fold scores low even if its
    pooled average looks fine; a model that errs by a similar amount every
    fold scores high. None if fewer than MIN_FOLDS_FOR_STABILITY folds
    succeeded (too little data to say anything about variability).

    `hyperparams`, if given, is forwarded into every fold's fit() call
    unchanged (Task 5) — see train_candidates() for where it comes from
    and why it is computed once per model rather than once per fold.

    Returns (metrics_dict_or_None, n_successful_folds, stability_or_None).
    Signature and return type are unchanged from this function's original
    version — every existing caller keeps working exactly as before.
    """
    n = len(series)
    if min_train_size is None:
        min_train_size = _default_min_train_size(n)
    hyperparams = hyperparams or {}

    if n - min_train_size < 1:
        return None, 0, None

    # Phase 2: delegate to the backtest engine rather than keeping a second
    # copy of the walk-forward loop here. One implementation means the
    # leakage-safe per-fold preprocessing and the multi-step scoring can
    # never drift apart from what this function reports. Every origin from
    # `min_train_size` onward is used (step 1), preserving this function's
    # original fold layout for callers that pass an explicit
    # min_train_size.
    horizon = _resolve_eval_horizon(n, min_train_size)
    folds = [
        FoldSpec(index=i + 1, train_end=origin, val_start=origin, val_end=origin + horizon)
        for i, origin in enumerate(range(min_train_size, n - horizon + 1))
    ]
    if not folds:
        return None, 0, None

    plan = ValidationPlan(
        horizon=horizon, requested_horizon=horizon, min_train_size=min_train_size,
        step=1, folds=folds,
        rule="explicit min_train_size supplied by caller (legacy walk_forward_validate path)",
    )
    observed = pd.Series(True, index=series.index)
    result = backtest_model(
        series, observed, model_cls, plan, frequency,
        hyperparams=hyperparams, model_name=getattr(model_cls, "name", model_cls.__name__),
    )

    if result.n_folds < MIN_WALKFORWARD_FOLDS or result.pooled_metrics is None:
        return None, result.n_folds, None
    stability = result.stability if result.n_folds >= MIN_FOLDS_FOR_STABILITY else None
    return result.pooled_metrics, result.n_folds, stability


def _dedupe_preserve_order(names: List[str]) -> List[str]:
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _business_friendly_training_error(name: str, exc: Exception) -> str:
    """
    Translate a raised exception into a short, non-technical message that
    is safe to display directly in the Forecast page's warning banner.
    The real exception — full type and message, e.g. the kind of internal
    attribute error a library can raise — is always still sent to the
    server log via logger.exception() at the call site; this function only
    controls what the end user sees.
    """
    text = str(exc)
    lowered = text.lower()

    if isinstance(exc, ImportError) or "not installed" in lowered:
        return f"{name} is unavailable because a required package is not installed."

    # Models raise their own already business-friendly ValueError messages
    # for known, expected causes (see each model's fit()/check_requirements())
    # — surface those as-is instead of masking them with a generic message.
    friendly_markers = ("observation", "cycle", "requires at least", "needs a", "needs at least")
    if isinstance(exc, ValueError) and any(marker in lowered for marker in friendly_markers):
        return f"{name} could not be trained: {text}"

    return (
        f"{name} could not be trained on this dataset — it may not be well suited "
        f"to this data's pattern or size."
    )


def _train_one_candidate(
    name: str,
    model_cls,
    series: pd.Series,
    training_frequency: str,
    min_train_size: int,
    tuning_prefix: pd.Series,
    n_obs_total: int,
    raw_series: Optional[pd.Series] = None,
    observed_mask: Optional[pd.Series] = None,
    plan: Optional[ValidationPlan] = None,
):
    """
    Task 16 — the full per-candidate pipeline (tune -> validate -> final
    fit) for ONE model, factored out so it can be run either sequentially
    or submitted to a thread pool without any behavioural difference.
    Reads only its own arguments (`series`/`tuning_prefix` are shared but
    never mutated by any model's fit()/predict() — every model instance
    below is a fresh, private object) and returns everything the caller
    needs (a ModelEvalResult, the fitted model or None, and this
    candidate's own warning strings) rather than appending to any shared
    list itself — so nothing here touches shared mutable state, which is
    what makes it safe to call from multiple threads concurrently.

    Identical in outcome to the pre-Task-16 inline loop body; only the
    "where do the results go" mechanism changed (return value instead of
    closure-captured list.append()).
    """
    local_warnings: List[str] = []
    from .results import ModelEvalResult  # local import avoids a cycle with results.py

    # Task 5: one-time hyperparameter tuning, before validation/fitting.
    hyperparams: dict = {}
    try:
        probe = model_cls()
        hyperparams = probe.suggest_hyperparameters(tuning_prefix, training_frequency) or {}
    except Exception as exc:
        logger.debug("%s: hyperparameter tuning failed, using defaults (%s)", name, exc)
        hyperparams = {}

    try:
        # ── Phase 2: honest walk-forward backtest ──────────────────────────
        # Every fold refits from scratch on its own training prefix, with
        # preprocessing fitted INSIDE the fold (see backtest.py), and is
        # scored at the production horizon against raw actuals.
        backtest: Optional[BacktestResult] = None
        metrics, n_folds, stability = None, 0, None
        fold_metrics: List[dict] = []
        fold_stats: Optional[dict] = None

        if plan is not None and plan.n_folds > 0 and raw_series is not None and observed_mask is not None:
            backtest = backtest_model(
                raw_series, observed_mask, model_cls, plan, training_frequency,
                hyperparams=hyperparams, model_name=name,
            )
            metrics = backtest.pooled_metrics
            n_folds = backtest.n_folds
            stability = backtest.stability
            fold_metrics = [f.to_dict() for f in backtest.folds]
            fold_stats = backtest.fold_stats
            if metrics is None:
                logger.debug("%s: no walk-forward folds could be validated", name)
                local_warnings.append(
                    f"{name}: could not complete any of the {plan.n_folds} walk-forward validation "
                    f"folds — trained on all available data, but no independent accuracy estimate "
                    f"is available."
                )
        elif plan is not None and plan.n_folds == 0:
            local_warnings.append(
                f"{name}: series too short for walk-forward validation — trained on all available "
                f"data without an independent accuracy estimate."
            )

        final_model = model_cls()
        # Refit on full history with the SAME tuned hyperparameters every
        # fold used, so the reported accuracy describes the configuration
        # that actually produces the forecast.
        final_model.fit(series, frequency=training_frequency, **hyperparams)

        status = "trained_validated" if metrics is not None else "trained_no_validation"
        result = ModelEvalResult(
            model=name, available=True, trained=True, metrics=metrics,
            validation_folds=n_folds, training_observations=n_obs_total,
            stability=stability, status=status,
            hyperparameters=(hyperparams or None),
            fold_metrics=fold_metrics or None,
            fold_stats=fold_stats,
            runtime_seconds=round(backtest.runtime_seconds, 3) if backtest else None,
        )
        return result, final_model, local_warnings, backtest

    except Exception as exc:
        logger.exception("%s failed to train on the full series", name)
        friendly = _business_friendly_training_error(name, exc)
        local_warnings.append(friendly)
        result = ModelEvalResult(
            model=name, available=True, trained=False, error=friendly,
            training_observations=n_obs_total, status="failed",
        )
        return result, None, local_warnings, None


def _train_sequential(
    pending: List[Tuple[str, type]],
    series: pd.Series,
    training_frequency: str,
    min_train_size: int,
    tuning_prefix: pd.Series,
    n_obs_total: int,
    raw_series: Optional[pd.Series] = None,
    observed_mask: Optional[pd.Series] = None,
    plan: Optional[ValidationPlan] = None,
) -> Dict[str, tuple]:
    """
    Run _train_one_candidate() for every pending candidate one at a time,
    in a plain for-loop — no ThreadPoolExecutor, no concurrent native
    library invocation. Used both when ENABLE_PARALLEL_FORECASTING is
    False, and as the defensive fallback if the thread pool itself cannot
    be used even when the flag is True. Each candidate is individually
    guarded (Task 4/8 — one candidate's bug can never stop the remaining
    candidates from being attempted), the same guarantee the parallel path
    provides, so switching between the two modes changes nothing about
    fault tolerance — only whether candidates share the process's threads
    while running.
    """
    from .results import ModelEvalResult  # local import avoids a cycle with results.py

    outcomes: Dict[str, tuple] = {}
    for name, model_cls in pending:
        try:
            outcomes[name] = _train_one_candidate(
                name, model_cls, series, training_frequency, min_train_size, tuning_prefix, n_obs_total,
                raw_series=raw_series, observed_mask=observed_mask, plan=plan,
            )
        except Exception as exc:
            logger.exception("%s: failed during sequential training", name)
            outcomes[name] = (
                ModelEvalResult(
                    model=name, available=True, trained=False,
                    error=f"Unexpected internal error while training: {exc}",
                    training_observations=n_obs_total, status="failed",
                ),
                None,
                [f"{name}: unexpected internal error during training — skipped."],
                None,
            )
    return outcomes


@dataclass
class TrainingOutcome:
    """
    Everything train_candidates() produces. Phase 2 replaced the previous
    5-tuple return with this so the fold plan, the per-model backtests and
    the evaluated ensemble travel alongside the existing fields instead of
    the tuple growing further. forecast_engine.py is the only caller.
    """
    results: List["ModelEvalResult"]
    trained_models: Dict[str, object]
    warnings: List[str]
    series: pd.Series
    preprocessing_info: dict
    plan: ValidationPlan
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    ensemble: Optional[EnsembleBacktest] = None


def train_candidates(
    dates: pd.Series,
    values: pd.Series,
    candidate_names: List[str],
    frequency: Optional[str] = None,
    horizon: int = 3,
) -> "TrainingOutcome":
    """
    Build the historical series at the selected forecast grain, backtest
    every candidate model against one shared fold plan, and refit each on
    the full history for production forecasting.

    Phase 2 changes:
      - Validation is planned once (forecast.backtest.plan_validation) and
        shared by every candidate, so all models are scored on identical
        origins at the PRODUCTION horizon and their rankings are directly
        comparable.
      - Outlier capping and gap bridging are no longer applied to the whole
        series here. They are fitted INSIDE each fold on that fold's
        training window only (see backtest.prepare_training_window), which
        removes the leak where IQR bounds were derived from periods the
        model was about to be scored against. The same preprocessing is
        applied once to the full series for the final production fit.
      - Per-fold metrics, fold-level summary statistics and a real
        ensemble backtest are returned rather than only pooled numbers.

    Args:
        frequency — the forecast grain chosen in Phase 1
            (SeriesStats.frequency). Inferred locally if omitted so this
            function remains usable standalone.
        horizon   — the production forecast horizon, used as the validation
            horizon so the two cannot drift apart.

    Returns a TrainingOutcome (see above).
    """
    from .results import ModelEvalResult  # local import avoids a cycle with results.py

    # ── Tasks 2 & 11: clean the raw rows BEFORE aggregation ────────────────
    # Forecasting must never run on raw, uncorrected data — see
    # preprocessing.py for exactly what each step does and does not change.
    #
    # Stability pass (Task 9): every preprocessing step below is now wrapped
    # in its own try/except with a safe fallback, so a bug in the
    # preprocessing/cleaning logic itself can never prevent forecasting from
    # proceeding — at worst, that ONE correction is skipped (and a warning
    # says so) while the rest of the pipeline continues on the
    # best-available data, exactly the same "degrade, never crash"
    # philosophy applied to individual models elsewhere in this file.
    warnings: List[str] = []
    try:
        dates, values, clean_notes, clean_counts = clean_transactions(dates, values)
        warnings.extend(clean_notes)
    except Exception:
        logger.exception("clean_transactions() failed — falling back to minimal inline cleaning")
        # Deliberately the simplest possible parse here, not
        # parse_dates_deterministic() — this branch only runs if
        # clean_transactions() itself (which already calls
        # parse_dates_deterministic() internally) raised, so the fallback
        # of a fallback stays as minimal as possible rather than risking
        # a second failure in the same class of code.
        dates = pd.to_datetime(dates, errors="coerce")
        values = _clean_numeric(values)
        valid = dates.notna() & values.notna()
        dates, values = dates[valid], values[valid]
        clean_counts = {
            "n_start": int(valid.shape[0]), "n_dropped_invalid": int((~valid).sum()),
            "n_implausible_dates": 0, "n_exact_duplicates": 0, "n_negative": 0,
        }
        warnings.append(
            "Automated data cleaning hit an unexpected error and was skipped for this run — "
            "basic date/number validation was still applied so forecasting can proceed."
        )

    if frequency is None:
        try:
            frequency = infer_frequency(dates)
        except Exception:
            logger.exception("infer_frequency() failed — falling back to monthly")
            frequency = "monthly"
            warnings.append("Could not automatically detect this dataset's frequency — assumed monthly.")

    def _empty_outcome(series_obj: pd.Series) -> "TrainingOutcome":
        return TrainingOutcome(
            results=[], trained_models={}, warnings=warnings, series=series_obj,
            preprocessing_info={**clean_counts, "n_capped": 0, "n_series_points": 0,
                                "missing_timestamp_ratio": 0.0},
            plan=plan_validation(0, horizon, None),
        )

    try:
        # build_forecast_series() (not build_series()) so the observed /
        # missing distinction survives into validation — a period nobody
        # recorded must not be scored as if it were a measured zero.
        construction = build_forecast_series(dates, values, frequency)
        series = construction.series
    except Exception:
        logger.exception("build_series() failed — cannot aggregate this dataset into a time series")
        warnings.append("This dataset could not be aggregated into a time series — cannot train any model.")
        return _empty_outcome(pd.Series(dtype=float))

    # Canonical grain the series above was actually built at — identical to
    # `frequency` for daily/weekly/monthly/quarterly/yearly, but resolved to
    # the same "monthly" fallback build_series() used internally whenever
    # `frequency` is "irregular"/"unknown". Every model-facing call below
    # uses THIS value, so a model never receives a frequency label that
    # doesn't match its input series' real PeriodIndex grain.
    training_frequency = effective_training_frequency(frequency)

    if len(series) == 0:
        warnings.append("No valid (date, revenue) observations available — cannot train any model.")
        return _empty_outcome(series)

    # ── Phase 2, Step 5: period-level cleanup is NO LONGER done here ───────
    # `cap_outliers()` and `interpolate_short_gaps()` used to run on the
    # complete series at this point, before any train/validation split.
    # That leaked twice: the IQR bounds used to cap training values were
    # derived partly from the periods later folds would be scored against,
    # and a gap could be bridged using a value from the future. Both
    # operations now happen inside each fold, fitted on that fold's own
    # training window (see backtest.prepare_training_window), and once on
    # the full series below purely to produce the final production fit.
    #
    # `raw_series` keeps the untransformed period totals so validation can
    # score forecasts against what actually happened rather than against a
    # capped version of it.
    raw_series = series
    observed_mask = construction.observed_mask

    clean_values, full_preprocessor = prepare_training_window(
        raw_series.to_numpy(dtype=float), observed_mask.to_numpy(dtype=bool),
    )
    series = pd.Series(clean_values, index=raw_series.index)
    n_capped = full_preprocessor.n_capped
    if n_capped:
        warnings.append(
            f"{n_capped} period(s) had a revenue spike or drop far outside this series' normal range "
            f"and were capped before the final model fit. During validation the same capping was "
            f"re-fitted inside each fold, using only that fold's training data."
        )

    # Missing-timestamp ratio (Task 8's data-quality signal): what fraction
    # of the gapless period range build_series() produced had NO underlying
    # raw transaction row at all (as opposed to a period that genuinely
    # summed to zero because every row in it happened to net to zero). Uses
    # the same frequency code build_series() itself resolved to, so this is
    # counted against the identical period grid the series was built on.
    try:
        code = period_freq_code(frequency)
        n_periods_with_data = dates.dt.to_period(code).nunique() if len(dates) > 0 else 0
        missing_timestamp_ratio = round(max(0.0, 1 - (n_periods_with_data / max(1, len(series)))), 4)
    except Exception:
        logger.exception("missing_timestamp_ratio computation failed — using a neutral default")
        missing_timestamp_ratio = 0.0

    preprocessing_info = {
        **clean_counts,
        "n_capped": n_capped,
        "n_series_points": len(series),
        "missing_timestamp_ratio": missing_timestamp_ratio,
    }

    all_candidates = _dedupe_preserve_order(list(candidate_names) + GUARANTEED_CANDIDATES)

    n_obs_total = len(series)

    # ── Phase 2, Steps 2-4: one fold plan, shared by every candidate ───────
    # Computed once from the series length, the PRODUCTION horizon and the
    # seasonal cycle, so every model is scored on identical origins and the
    # rankings below compare like with like.
    plan = plan_validation(
        n_obs_total, requested_horizon=horizon,
        seasonal_period=seasonal_period_for_freq(training_frequency),
    )
    if plan.degraded and plan.degradation_reason:
        warnings.append(plan.degradation_reason)
    if plan.n_folds == 0:
        warnings.append(
            f"Only {n_obs_total} {frequency} observation(s) available — too little history for "
            f"walk-forward validation. Models were fit on all available data without validation metrics."
        )

    results_by_name: Dict[str, ModelEvalResult] = {}
    warnings_by_name: Dict[str, List[str]] = {}
    trained_models: Dict[str, object] = {}
    backtests: Dict[str, BacktestResult] = {}

    # Task 5 — hyperparameter tuning uses the SAME chronological prefix the
    # first validation fold trains on, so tuning never sees data that fold
    # would not yet have had.
    min_train_size = plan.min_train_size if plan.n_folds else _default_min_train_size(n_obs_total)
    tuning_prefix = series.iloc[:min_train_size] if 0 < min_train_size < n_obs_total else series

    # ── Pass 1 (sequential, fast): registry lookup / dependency / suitability
    # checks for every candidate. Kept sequential because it is cheap
    # (no model fitting happens here) and determines the warnings/skipped
    # entries in a fixed, easy-to-reason-about order before any concurrent
    # work starts.
    pending: List[Tuple[str, type]] = []  # candidates that passed all pre-checks and are ready to train
    for name in all_candidates:
        model_cls = MODEL_REGISTRY.get(name)

        if model_cls is None:
            results_by_name[name] = ModelEvalResult(
                model=name, available=False, trained=False,
                error="Not yet implemented in the Forecast Model Library.",
                training_observations=n_obs_total, status="not_implemented",
            )
            warnings.append(f"{name} recommended by Phase 1 but not implemented — skipped.")
            continue

        if not model_cls.is_available():
            results_by_name[name] = ModelEvalResult(
                model=name, available=False, trained=False,
                error="Required dependency is not installed.",
                training_observations=n_obs_total, status="skipped_unavailable",
            )
            warnings.append(f"{name} unavailable: required dependency is not installed.")
            continue

        # ── Task 4: per-model suitability check, before any fitting is attempted ──
        probe = model_cls()
        ok, requirement_message = probe.check_requirements(series, training_frequency)
        if not ok:
            results_by_name[name] = ModelEvalResult(
                model=name, available=True, trained=False, error=requirement_message,
                training_observations=n_obs_total, status="skipped_unsuitable",
            )
            if requirement_message:
                warnings.append(requirement_message)
            continue
        if requirement_message:
            # Soft caution (e.g. SES on a strong-trend series) — the model
            # still trains; the user just gets a heads-up about why it may
            # underperform.
            warnings.append(requirement_message)

        pending.append((name, model_cls))

    # ── Pass 2: tune + validate + fit every candidate that passed
    # pre-checks. Two interchangeable execution modes, controlled by
    # ENABLE_PARALLEL_FORECASTING (runtime stability audit — see that
    # flag's docstring above):
    #
    #   - Parallel (Task 16, flag True): each candidate's independent
    #     tuning/validation/fit pipeline (`series`/`tuning_prefix` are only
    #     ever read, never mutated — no shared mutable state between
    #     candidates at the PYTHON level) runs concurrently across a small
    #     ThreadPoolExecutor. Every worker returns its own outcome rather
    #     than mutating a shared list, and results are merged back in by
    #     name afterward.
    #   - Sequential (flag False, the current default): the exact same
    #     per-candidate function, _train_one_candidate(), called one at a
    #     time via _train_sequential() — identical logic, identical output,
    #     just no concurrent native-library invocation.
    #
    # Both paths funnel into the same `outcomes` dict shape, so the
    # assembly step below (which restores the exact original
    # `all_candidates` order) is identical either way.
    if pending:
        if ENABLE_PARALLEL_FORECASTING:
            try:
                max_workers = min(MAX_PARALLEL_CANDIDATES, len(pending))
                outcomes = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _train_one_candidate, name, model_cls, series,
                            training_frequency, min_train_size, tuning_prefix, n_obs_total,
                            raw_series, observed_mask, plan,
                        ): name
                        for name, model_cls in pending
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        # Task 8 — defensive second layer: _train_one_candidate()
                        # already catches every exception internally and returns
                        # it as data (result, model, warnings) instead of
                        # raising, so future.result() should never actually
                        # raise here. This try/except exists purely so that
                        # even a future bug that lets something escape that
                        # function can never propagate out of the thread pool,
                        # take down the whole training run, and (via Flask's
                        # own request-level exception handling) surface as an
                        # opaque 500 instead of "this one model failed."
                        try:
                            outcomes[name] = future.result()
                        except Exception as exc:
                            logger.exception("%s: unexpected exception escaped its worker thread", name)
                            outcomes[name] = (
                                ModelEvalResult(
                                    model=name, available=True, trained=False,
                                    error=f"Unexpected internal error while training: {exc}",
                                    training_observations=n_obs_total, status="failed",
                                ),
                                None,
                                [f"{name}: unexpected internal error during training — skipped."],
                                None,
                            )
            except Exception:
                # Defensive fallback: if the thread pool itself cannot be
                # used for any reason (e.g. a restricted execution
                # environment), fall back to the same sequential path used
                # when the flag is off, rather than losing the forecast
                # entirely.
                logger.exception("Parallel candidate training failed — falling back to sequential training.")
                outcomes = _train_sequential(
                    pending, series, training_frequency, min_train_size, tuning_prefix, n_obs_total,
                    raw_series=raw_series, observed_mask=observed_mask, plan=plan,
                )
        else:
            outcomes = _train_sequential(
                pending, series, training_frequency, min_train_size, tuning_prefix, n_obs_total,
                raw_series=raw_series, observed_mask=observed_mask, plan=plan,
            )

        for name, _model_cls in pending:
            result, trained_model, local_warnings, backtest = outcomes[name]
            results_by_name[name] = result
            warnings_by_name[name] = local_warnings
            if trained_model is not None:
                trained_models[name] = trained_model
            if backtest is not None and backtest.folds:
                backtests[name] = backtest

    # ── Final assembly: reconstruct `results` (and each candidate's own
    # warnings) in the ORIGINAL all_candidates order, identical in shape and
    # ordering to the pre-Task-16 sequential loop — parallelising pass 2
    # only changes wall-clock time, never the output.
    results: List[ModelEvalResult] = [results_by_name[name] for name in all_candidates]
    for name in all_candidates:
        warnings.extend(warnings_by_name.get(name, []))

    return TrainingOutcome(
        results=results,
        trained_models=trained_models,
        warnings=warnings,
        series=series,
        preprocessing_info=preprocessing_info,
        plan=plan,
        backtests=backtests,
    )
