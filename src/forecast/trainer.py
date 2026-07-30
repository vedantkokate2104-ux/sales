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

Production-quality pass #2 (this pass — preprocessing + auto-tuning):
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

Runtime stability audit (this pass): a live segmentation fault was
reported with worker threads inside `concurrent.futures.thread._worker`
on the stack, plus `resource_tracker: leaked semaphore objects` warnings
at shutdown, on macOS (Apple Silicon). A project-wide search confirmed
this file's ThreadPoolExecutor above is the ONLY thread/process pool in
the entire codebase — every SARIMA/Holt-Winters (statsmodels -> scipy ->
LAPACK/BLAS) and XGBoost (its own internal OpenMP pool, never given an
explicit thread-count limit) fit can therefore run CONCURRENTLY across
several threads, entering native, C-level thread-pool/BLAS code from
multiple Python threads at once — a documented crash-prone pattern on
macOS (see CHANGELOG for the full library-by-library reasoning). Rather
than removing parallel execution, candidate training now runs behind a
module-level `ENABLE_PARALLEL_FORECASTING` flag: sequential
(`_train_sequential()`, calling the exact same `_train_one_candidate()`
one candidate at a time — identical logic and output, just no concurrent
native-library invocation) is now the default while stability is
confirmed in the field; the ThreadPoolExecutor path is fully preserved,
selectable by setting the flag to `True`. Separately, `process_isolation
.py` (used by Prophet) now explicitly closes every `multiprocessing.Queue`
/`Process` it creates instead of leaving them for garbage collection —
the direct fix for the observed leaked-semaphore warning.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import _clean_numeric
from .evaluator import compute_metrics
from .models import MODEL_REGISTRY
from .preprocessing import cap_outliers, clean_transactions, interpolate_short_gaps, parse_dates_deterministic
from .utils import effective_training_frequency, infer_frequency, period_freq_code

logger = logging.getLogger(__name__)

# Always evaluated in addition to whatever Phase 1 recommends, so there is
# always at least one (in fact three) trainable baseline — this is what
# keeps series types like "Intermittent Demand" or "Highly Volatile"
# (whose Phase 1 top picks, Croston/XGBoost, are not yet implemented in
# this library) from ending up with zero candidates to compare.
GUARANTEED_CANDIDATES = ["Naive", "Moving Average", "Simple Exponential Smoothing"]

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
    indexed by a gapless PeriodIndex (missing periods filled with 0), at
    whatever frequency the data was actually detected to have — daily,
    weekly, monthly, quarterly, or yearly. Irregular/unknown cadences fall
    back to a monthly grain purely so training remains possible; that
    fallback is reported to the user as a dataset-validation warning (see
    validation.py), not silently applied as if it were the "real" cadence.

    This replaces the previous behaviour of always aggregating to monthly
    regardless of the data's actual cadence.

    Date parsing (Task 10) uses parse_dates_deterministic() rather than a
    bare pd.to_datetime(): see preprocessing.py's module docstring for why
    pandas' own per-row format inference (triggered whenever a column's
    format can't be inferred from the first few rows — as with this
    project's own sample dataset's day-first, 2-digit-year dates) is a
    genuine silent-corruption risk on ambiguous dates, not just a noisy
    warning.
    """
    dates = parse_dates_deterministic(dates)
    valid = dates.notna()
    dates, values = dates[valid], _clean_numeric(values[valid])

    if len(dates) == 0:
        return pd.Series(dtype=float)

    if frequency is None:
        frequency = infer_frequency(dates)

    code = period_freq_code(frequency)  # falls back to "M" for irregular/unknown
    periods = dates.dt.to_period(code)
    series = values.groupby(periods).sum().sort_index()

    full_range = pd.period_range(series.index.min(), series.index.max(), freq=code)
    return series.reindex(full_range, fill_value=0.0)


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
    Expanding-window, one-step-ahead walk-forward validation for a single
    candidate model class — simulates how the model is actually used in
    production, unlike a single fixed hold-out split:

        Train on series[:i]  ->  predict series[i]  ->  compare
        ->  expand i by 1     ->  repeat until the series is exhausted.

    Order is never shuffled; each fold's training window is a strict
    chronological prefix of the one before it, and every prediction comes
    from a FRESH model instance fit only on that fold's own prefix (a
    fold's fit never reuses another fold's fitted state, and never sees
    series.iloc[train_end] or anything after it) — this is what makes the
    walk-forward loop leakage-free by construction.

    All folds' (actual, predicted) pairs are pooled and every metric
    (MAE/RMSE/MAPE/SMAPE/WAPE) is computed ONCE over the pooled arrays,
    rather than computing a metric per fold and then averaging the
    per-fold metrics. For a single-step fold, a per-fold RMSE degenerates
    to the same value as a per-fold MAE (sqrt of one squared term is just
    the absolute value), so averaging per-fold RMSEs would silently
    produce MAE's number under RMSE's name. Pooling first and computing
    once preserves the actual meaning of each metric while still
    reflecting performance averaged across every validation fold.

    A fold whose model fails to fit on that (early, shorter) training
    window — e.g. SARIMA cannot fit until 2 full seasonal cycles exist —
    is simply skipped rather than aborting the whole validation; the
    caller decides whether too few folds succeeded to trust the result.

    A third return value, `stability`, is a 0-1 score describing how
    consistent this model's per-fold absolute error was across folds:
    1 - min(1, coefficient_of_variation_of_fold_errors). A model whose
    error swings wildly fold to fold scores low even if its pooled
    average looks fine; a model that errs by a similar amount every fold
    scores high. None if fewer than MIN_FOLDS_FOR_STABILITY folds
    succeeded (too little data to say anything about variability).

    `hyperparams`, if given, is forwarded into every fold's fit() call
    unchanged (Task 5) — see train_candidates() for where it comes from
    and why it is computed once per model rather than once per fold.

    Returns (metrics_dict_or_None, n_successful_folds, stability_or_None).
    """
    n = len(series)
    if min_train_size is None:
        min_train_size = _default_min_train_size(n)
    hyperparams = hyperparams or {}

    if n - min_train_size < 1:
        return None, 0, None

    actuals: List[float] = []
    preds: List[float] = []
    fold_abs_errors: List[float] = []

    for train_end in range(min_train_size, n):
        train = series.iloc[:train_end]
        actual = series.iloc[train_end]
        try:
            probe = model_cls()
            probe.fit(train, frequency=frequency, **hyperparams)
            pred = probe.predict(1)[0]
        except Exception as exc:
            logger.debug(
                "%s skipped walk-forward fold ending at %d (%d obs): %s",
                getattr(model_cls, "name", model_cls.__name__), train_end, len(train), exc,
            )
            continue
        actuals.append(float(actual))
        preds.append(float(pred))
        fold_abs_errors.append(abs(float(actual) - float(pred)))

    n_folds = len(actuals)
    if n_folds < MIN_WALKFORWARD_FOLDS:
        return None, n_folds, None

    metrics = compute_metrics(np.array(actuals), np.array(preds))

    stability = None
    if n_folds >= MIN_FOLDS_FOR_STABILITY:
        mean_err = float(np.mean(fold_abs_errors))
        std_err = float(np.std(fold_abs_errors))
        if mean_err <= _STABILITY_EPS:
            # every fold's error was ~0 -- perfectly consistent (and
            # perfectly accurate), not "undefined"
            cv = 0.0
        else:
            cv = std_err / (mean_err + _STABILITY_EPS)
        stability = round(max(0.0, min(1.0, 1 - min(1.0, cv))), 4)

    return metrics, n_folds, stability


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
        metrics, n_folds, stability = None, 0, None
        if len(series) >= 4:
            metrics, n_folds, stability = walk_forward_validate(
                series, model_cls, training_frequency, min_train_size=min_train_size, hyperparams=hyperparams
            )
            if metrics is None and n_folds == 0:
                logger.debug("%s: no walk-forward folds could be validated", name)
                local_warnings.append(
                    f"{name}: not enough history for repeated walk-forward validation folds — "
                    f"trained on all available data, but no independent accuracy estimate is "
                    f"available yet."
                )

        final_model = model_cls()
        # refit on full history, with the SAME tuned hyperparameters every
        # walk-forward fold above just used, for the actual forecast
        final_model.fit(series, frequency=training_frequency, **hyperparams)

        status = "trained_validated" if metrics is not None else "trained_no_validation"
        result = ModelEvalResult(
            model=name, available=True, trained=True, metrics=metrics,
            validation_folds=n_folds, training_observations=n_obs_total,
            stability=stability, status=status,
            hyperparameters=(hyperparams or None),
        )
        return result, final_model, local_warnings

    except Exception as exc:
        logger.exception("%s failed to train on the full series", name)
        friendly = _business_friendly_training_error(name, exc)
        local_warnings.append(friendly)
        result = ModelEvalResult(
            model=name, available=True, trained=False, error=friendly,
            training_observations=n_obs_total, status="failed",
        )
        return result, None, local_warnings


def _train_sequential(
    pending: List[Tuple[str, type]],
    series: pd.Series,
    training_frequency: str,
    min_train_size: int,
    tuning_prefix: pd.Series,
    n_obs_total: int,
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
            )
    return outcomes


def train_candidates(
    dates: pd.Series,
    values: pd.Series,
    candidate_names: List[str],
    frequency: Optional[str] = None,
):
    """
    Build the historical series at its detected frequency, then validate
    (walk-forward) and train every candidate model (Phase 1's
    recommendations plus the guaranteed baselines).

    Args:
        frequency — the dataset's detected frequency (from Phase 1's
            SeriesStats.frequency). If not supplied, it is inferred locally
            from `dates` so this function remains usable standalone.

    Returns:
        results          — list[ModelEvalResult] (imported lazily to avoid
                            a circular import with results.py)
        trained_models    — dict[model_name -> fitted model instance ready
                            to forecast()], refit on the FULL series so the
                            final forecast uses all available history
        warnings          — list[str] describing any skipped/failed model
                            OR data correction, written in business-friendly
                            language
        series            — the full series (at the detected frequency),
                            AFTER preprocessing (Tasks 2, 11), used for training
        preprocessing_info — dict with the raw counts behind the
                            preprocessing notes above (n_start,
                            n_dropped_invalid, n_implausible_dates,
                            n_exact_duplicates, n_negative, n_capped,
                            n_series_points) — consumed by
                            forecast_engine.py to compute a data-quality
                            score for the confidence calculation (Task 8)
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

    try:
        series = build_series(dates, values, frequency=frequency)
    except Exception:
        logger.exception("build_series() failed — cannot aggregate this dataset into a time series")
        warnings.append("This dataset could not be aggregated into a time series — cannot train any model.")
        preprocessing_info = {**clean_counts, "n_capped": 0, "n_series_points": 0, "missing_timestamp_ratio": 0.0}
        return [], {}, warnings, pd.Series(dtype=float), preprocessing_info

    # Canonical grain the series above was actually built at — identical to
    # `frequency` for daily/weekly/monthly/quarterly/yearly, but resolved to
    # the same "monthly" fallback build_series() used internally whenever
    # `frequency` is "irregular"/"unknown". Every model-facing call below
    # uses THIS value, so a model never receives a frequency label that
    # doesn't match its input series' real PeriodIndex grain.
    training_frequency = effective_training_frequency(frequency)

    if len(series) == 0:
        warnings.append("No valid (date, revenue) observations available — cannot train any model.")
        preprocessing_info = {**clean_counts, "n_capped": 0, "n_series_points": 0, "missing_timestamp_ratio": 0.0}
        return [], {}, warnings, series, preprocessing_info

    # ── Tasks 2 & 11: period-level cleanup, AFTER aggregation ──────────────
    # "An outlier" and "a gap" are only meaningful once the data has a
    # fixed periodicity, so these run on the aggregated series, not the
    # raw rows. Each step falls back to leaving `series` exactly as it was
    # going in if it raises — a skipped correction, never a stopped pipeline.
    n_capped = 0
    try:
        series, cap_notes, n_capped = cap_outliers(series)
        warnings.extend(cap_notes)
    except Exception:
        logger.exception("cap_outliers() failed — proceeding without outlier capping for this run")
        warnings.append("Automated outlier capping hit an unexpected error and was skipped for this run.")

    try:
        series, gap_notes = interpolate_short_gaps(series)
        warnings.extend(gap_notes)
    except Exception:
        logger.exception("interpolate_short_gaps() failed — proceeding without gap interpolation for this run")
        warnings.append("Automated gap interpolation hit an unexpected error and was skipped for this run.")

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

    if len(series) < 4:
        warnings.append(
            f"Only {len(series)} {frequency} observation(s) available — too little history for "
            f"reliable validation. Models were fit on all available data without validation metrics."
        )

    results_by_name: Dict[str, ModelEvalResult] = {}
    warnings_by_name: Dict[str, List[str]] = {}
    trained_models: Dict[str, object] = {}

    n_obs_total = len(series)
    # Task 5 — the SAME chronological prefix walk-forward's first fold
    # would use, shared across every candidate so hyperparameter tuning is
    # done once per model (not once per fold) and never sees data beyond
    # what the earliest fold itself would have had.
    min_train_size = _default_min_train_size(n_obs_total)
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
                )
        else:
            outcomes = _train_sequential(
                pending, series, training_frequency, min_train_size, tuning_prefix, n_obs_total,
            )

        for name, _model_cls in pending:
            result, trained_model, local_warnings = outcomes[name]
            results_by_name[name] = result
            warnings_by_name[name] = local_warnings
            if trained_model is not None:
                trained_models[name] = trained_model

    # ── Final assembly: reconstruct `results` (and each candidate's own
    # warnings) in the ORIGINAL all_candidates order, identical in shape and
    # ordering to the pre-Task-16 sequential loop — parallelising pass 2
    # only changes wall-clock time, never the output.
    results: List[ModelEvalResult] = [results_by_name[name] for name in all_candidates]
    for name in all_candidates:
        warnings.extend(warnings_by_name.get(name, []))

    return results, trained_models, warnings, series, preprocessing_info
