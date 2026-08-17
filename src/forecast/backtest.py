"""
backtest.py
------------
Honest time-series backtesting (Phase 2).

WHAT WAS WRONG
---------------
`trainer.walk_forward_validate()` was already expanding-window and
chronological, but three things made its output untrustworthy:

  1. VALIDATION SIZE WAS CAPPED AT SIX PERIODS.
     `val_span = max(1, min(6, n // 5))` meant folds saturated at 4 and
     validation points at 12 no matter how long the series was — 12 of
     209 weekly periods (5.7%), 12 of 48 monthly periods (25%). Every
     accuracy number and the entire model choice rested on the final
     handful of periods.

  2. PREPROCESSING WAS FITTED ON THE WHOLE SERIES BEFORE SPLITTING.
     `cap_outliers()` derived IQR bounds from all 48 periods, including
     the ones later folds were scored against, and
     `interpolate_short_gaps()` could bridge a gap using a value from the
     future. Both leak.

  3. THE VALIDATION HORIZON DRIFTED FROM THE PRODUCTION HORIZON.
     `_resolve_eval_horizon()` silently fell back to 1- or 2-step
     evaluation on shorter series while production forecasts 3 steps.

THE PROTOCOL IMPLEMENTED HERE
------------------------------
One fold plan is computed per series (see `plan_validation`) and shared
by every candidate, so all models are scored on identical origins and
their rankings are directly comparable. For each fold:

    train = raw[:origin]                  strictly before the origin
      -> fit preprocessing on train ONLY  (IQR bounds, gap bridging)
      -> transform train
      -> fit model on transformed train
      -> predict `horizon` steps
      -> score against RAW actuals at [origin : origin+horizon]

Nothing at or after `origin` participates in fitting the model or the
preprocessing. Validation actuals are never transformed — a forecast is
judged against what actually happened, not against a capped version of
it. Periods inside the validation window that were never observed (see
forecast.grain) are excluded from scoring rather than compared against a
filled placeholder.

Per-fold predictions are retained, which is what makes it possible to
evaluate an ENSEMBLE properly (see `evaluate_ensemble`): the ensemble's
metrics are computed from actual blended predictions on the same folds,
not inherited from its best component.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluator import compute_metrics
from .utils import seasonal_period_for_freq

logger = logging.getLogger(__name__)

# ── Fold planning constants ────────────────────────────────────────────────

# Absolute floor on a training window. Below this nothing can be fit
# meaningfully, however desperate the fold budget.
MIN_ABSOLUTE_TRAIN = 8

# A model needs at least this many folds before its pooled metrics are
# treated as an estimate of anything. Below it, fold-to-fold variation is
# indistinguishable from noise.
MIN_FOLDS = 3

# Target number of validation origins. Chosen to cover a meaningful
# stretch of history without refitting every model dozens of times: at
# 48 monthly periods this yields ~11 origins spanning the second half of
# the series. When more origins are available than this, the step between
# them is widened rather than the count being truncated, so the folds stay
# spread across history instead of clustering at one end.
TARGET_FOLDS = 12

# Fraction of the series reserved as training history before the FIRST
# validation origin. Half is a deliberate compromise: large enough that
# fold 1 is not fit on an unrealistically short window, small enough to
# leave room for enough origins to see varied conditions.
MIN_TRAIN_FRACTION = 0.5

# Outlier capping multiplier, applied to bounds derived from the TRAINING
# window of each fold only. Matches forecast.preprocessing's constant so
# behaviour is unchanged apart from where the bounds come from.
OUTLIER_IQR_MULTIPLIER = 3.0

# Runs of missing periods longer than this are not bridged (matches
# forecast.grain.MAX_INTERPOLATED_RUN).
MAX_INTERPOLATED_RUN = 2


@dataclass
class FoldSpec:
    """One validation origin. `train_end` is exclusive and equals
    `val_start`, so training data always strictly precedes validation
    data by construction — there is no arithmetic anywhere that could
    make them overlap."""
    index: int
    train_end: int
    val_start: int
    val_end: int

    @property
    def train_size(self) -> int:
        return self.train_end

    def to_dict(self) -> dict:
        return {
            "fold": self.index, "train_end": self.train_end,
            "val_start": self.val_start, "val_end": self.val_end,
            "train_size": self.train_size,
        }


@dataclass
class ValidationPlan:
    """The fold layout every candidate model is scored against."""
    horizon: int
    requested_horizon: int
    min_train_size: int
    step: int
    folds: List[FoldSpec]
    rule: str
    degraded: bool = False
    degradation_reason: Optional[str] = None

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def n_validation_points(self) -> int:
        return sum(f.val_end - f.val_start for f in self.folds)

    def to_dict(self) -> dict:
        return {
            "horizon":             self.horizon,
            "requested_horizon":   self.requested_horizon,
            "min_train_size":      self.min_train_size,
            "step":                self.step,
            "n_folds":             self.n_folds,
            "n_validation_points": self.n_validation_points,
            "rule":                self.rule,
            "degraded":            self.degraded,
            "degradation_reason":  self.degradation_reason,
            "folds":               [f.to_dict() for f in self.folds],
        }


@dataclass
class FoldOutcome:
    """One model's result on one fold, including the raw predictions —
    retained so ensembles can be blended and scored on the same folds."""
    fold: int
    train_size: int
    origin_period: str
    periods: List[str]
    actuals: List[float]
    predictions: List[float]
    metrics: Dict[str, Optional[float]]

    def to_dict(self) -> dict:
        return {
            "fold": self.fold, "train_size": self.train_size,
            "origin_period": self.origin_period, "periods": self.periods,
            "actuals": [round(a, 2) for a in self.actuals],
            "predictions": [round(p, 2) for p in self.predictions],
            "metrics": self.metrics,
        }


@dataclass
class BacktestResult:
    """Everything measured about one model across the whole fold plan."""
    model: str
    folds: List[FoldOutcome] = field(default_factory=list)
    pooled_metrics: Optional[Dict[str, Optional[float]]] = None
    fold_stats: Optional[Dict[str, Optional[float]]] = None
    stability: Optional[float] = None
    runtime_seconds: float = 0.0
    n_folds_attempted: int = 0
    error: Optional[str] = None

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def n_points(self) -> int:
        return sum(len(f.actuals) for f in self.folds)

    def fold_metric_series(self, key: str = "wape") -> List[float]:
        out = []
        for f in self.folds:
            v = f.metrics.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
        return out

    def to_dict(self) -> dict:
        return {
            "model":             self.model,
            "n_folds":           self.n_folds,
            "n_folds_attempted": self.n_folds_attempted,
            "n_points":          self.n_points,
            "pooled_metrics":    self.pooled_metrics,
            "fold_stats":        self.fold_stats,
            "stability":         self.stability,
            "runtime_seconds":   round(self.runtime_seconds, 3),
            "error":             self.error,
            "folds":             [f.to_dict() for f in self.folds],
        }


# ── Fold planning (Step 4) ─────────────────────────────────────────────────

def plan_validation(
    n: int,
    requested_horizon: int = 3,
    seasonal_period: Optional[int] = None,
) -> ValidationPlan:
    """
    Derive the fold layout for a series of `n` periods.

    THE RULE, exactly:

      1. The validation horizon equals the production horizon
         (`requested_horizon`). It is only ever reduced as a last resort
         (step 4 below), and when that happens `degraded` is set so the
         mismatch is reported rather than hidden.

      2. The first training window is
             min_train = max(MIN_ABSOLUTE_TRAIN,
                             2 x seasonal_period,
                             MIN_TRAIN_FRACTION x n)
         Two full seasonal cycles is the floor at which seasonal models
         (Holt-Winters, SARIMA) can estimate a cycle at all, and half the
         series keeps fold 1 from being fit on an unrealistically short
         window.

      3. If that leaves fewer than MIN_FOLDS origins, the seasonal
         requirement is relaxed one step at a time — two cycles, then one
         cycle, then none — before anything else gives way. Each
         relaxation is recorded in `rule`.

      4. If there are still too few origins, the horizon is reduced
         (3 -> 2 -> 1) and `degraded` is set with a reason.

      5. Origins run from `min_train` to `n - horizon`, spaced by
             step = ceil(available_origins / TARGET_FOLDS)
         so folds stay spread across history instead of clustering, and
         a long series does not force one refit per period.

    Returns a plan with zero folds when the series cannot support
    validation at all; callers treat that as "trained, not validated"
    rather than as an error.
    """
    notes: List[str] = []

    def _origins(min_train: int, horizon: int) -> Tuple[List[FoldSpec], int]:
        last_origin = n - horizon
        if min_train > last_origin:
            return [], 0
        available = last_origin - min_train + 1
        step = max(1, int(np.ceil(available / TARGET_FOLDS)))
        specs = []
        for i, origin in enumerate(range(min_train, last_origin + 1, step)):
            specs.append(FoldSpec(index=i + 1, train_end=origin,
                                  val_start=origin, val_end=origin + horizon))
        return specs, step

    seasonal_min_candidates = []
    if seasonal_period and seasonal_period > 1:
        seasonal_min_candidates = [
            (2 * seasonal_period, f"two full seasonal cycles ({2 * seasonal_period} periods)"),
            (seasonal_period, f"one full seasonal cycle ({seasonal_period} periods)"),
        ]
    seasonal_min_candidates.append((0, "no seasonal-cycle requirement"))

    fraction_floor = int(np.ceil(MIN_TRAIN_FRACTION * n))
    horizon = max(1, int(requested_horizon))

    for h in range(horizon, 0, -1):
        for seasonal_floor, seasonal_label in seasonal_min_candidates:
            min_train = max(MIN_ABSOLUTE_TRAIN, seasonal_floor, fraction_floor)
            folds, step = _origins(min_train, h)
            if len(folds) >= MIN_FOLDS:
                rule = (
                    f"min_train = max(absolute floor {MIN_ABSOLUTE_TRAIN}, {seasonal_label}, "
                    f"{int(MIN_TRAIN_FRACTION * 100)}% of {n} periods) = {min_train}; "
                    f"origins {min_train}..{n - h} stepped by {step} to target ~{TARGET_FOLDS} folds; "
                    f"horizon {h} evaluated at every origin"
                )
                return ValidationPlan(
                    horizon=h, requested_horizon=requested_horizon,
                    min_train_size=min_train, step=step, folds=folds, rule=rule,
                    degraded=(h != requested_horizon),
                    degradation_reason=(
                        f"Only {n} periods available — the validation horizon was reduced from "
                        f"{requested_horizon} to {h} to obtain at least {MIN_FOLDS} folds. "
                        f"Validation therefore measures {h}-step accuracy while production "
                        f"forecasts {requested_horizon} steps."
                    ) if h != requested_horizon else None,
                )
        notes.append(f"horizon {h} could not yield {MIN_FOLDS} folds")

    # Nothing worked: relax the training floor to the absolute minimum and
    # take whatever folds exist, even if fewer than MIN_FOLDS.
    for h in range(horizon, 0, -1):
        folds, step = _origins(MIN_ABSOLUTE_TRAIN, h)
        if folds:
            return ValidationPlan(
                horizon=h, requested_horizon=requested_horizon,
                min_train_size=MIN_ABSOLUTE_TRAIN, step=step, folds=folds,
                rule=(
                    f"Series too short for the standard rule ({'; '.join(notes)}). Fell back to the "
                    f"absolute minimum training window of {MIN_ABSOLUTE_TRAIN} periods at horizon {h}."
                ),
                degraded=True,
                degradation_reason=(
                    f"Only {len(folds)} validation fold(s) at horizon {h} were possible from {n} "
                    f"periods — fewer than the {MIN_FOLDS} needed for a reliable estimate. Treat "
                    f"these metrics as indicative only."
                ),
            )

    return ValidationPlan(
        horizon=horizon, requested_horizon=requested_horizon,
        min_train_size=MIN_ABSOLUTE_TRAIN, step=1, folds=[],
        rule=f"{n} periods is too short to hold out any validation window at horizon {horizon}.",
        degraded=True,
        degradation_reason=(
            f"No walk-forward validation was possible: {n} periods cannot support a training "
            f"window of at least {MIN_ABSOLUTE_TRAIN} plus a {horizon}-period holdout."
        ),
    )


# ── Per-fold preprocessing (Step 5) ────────────────────────────────────────

@dataclass
class FoldPreprocessor:
    """
    Preprocessing whose parameters are FITTED on one training window and
    then applied to it. Deliberately has no `transform` for validation
    data: validation actuals are compared raw, because the question being
    asked is "how close was the forecast to what really happened", not
    "how close was it to a cleaned version of what happened".
    """
    lower: Optional[float] = None
    upper: Optional[float] = None
    n_capped: int = 0

    @classmethod
    def fit(cls, train_values: np.ndarray, multiplier: float = OUTLIER_IQR_MULTIPLIER) -> "FoldPreprocessor":
        finite = train_values[np.isfinite(train_values)]
        if len(finite) < 4:
            return cls()
        q1, q3 = np.percentile(finite, 25), np.percentile(finite, 75)
        iqr = q3 - q1
        if iqr <= 0:
            return cls()
        return cls(lower=max(0.0, q1 - multiplier * iqr), upper=q3 + multiplier * iqr)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.lower is None or self.upper is None:
            return values
        clipped = np.clip(values, self.lower, self.upper)
        self.n_capped = int(np.sum(clipped != values))
        return clipped


def _bridge_missing(values: np.ndarray, observed: np.ndarray, max_run: int = MAX_INTERPOLATED_RUN) -> np.ndarray:
    """
    Fill unobserved periods inside a TRAINING window only. Short runs
    bounded by observed activity are interpolated; everything else falls
    back to zero. Because this is called per fold on `values[:origin]`,
    it can only ever look backwards from the fold's own cutoff — unlike
    interpolating the full series up front, which could bridge a gap
    using a value the fold is about to be scored against.
    """
    result = np.where(observed, values, 0.0).astype(float)
    n = len(result)
    i = 0
    while i < n:
        if observed[i]:
            i += 1
            continue
        j = i
        while j < n and not observed[j]:
            j += 1
        run = j - i
        if run <= max_run and i > 0 and observed[i - 1] and j < n and observed[j]:
            left, right = result[i - 1], result[j]
            for k in range(run):
                result[i + k] = left + (k + 1) / (run + 1) * (right - left)
        i = j
    return result


def prepare_training_window(
    raw_values: np.ndarray, observed: np.ndarray
) -> Tuple[np.ndarray, FoldPreprocessor]:
    """Bridge missing periods, then cap outliers using bounds derived from
    this window alone. Returns (clean_values, fitted_preprocessor)."""
    bridged = _bridge_missing(raw_values, observed)
    pre = FoldPreprocessor.fit(bridged)
    return pre.transform(bridged), pre


# ── Backtesting one model ──────────────────────────────────────────────────

def backtest_model(
    raw_series: pd.Series,
    observed_mask: pd.Series,
    model_cls,
    plan: ValidationPlan,
    frequency: str,
    hyperparams: Optional[dict] = None,
    model_name: Optional[str] = None,
) -> BacktestResult:
    """
    Run `model_cls` across every fold in `plan`, refitting from scratch on
    each fold's own training prefix.

    `raw_series` carries the untouched period totals (with whatever the
    grain layer filled for unobserved periods) and `observed_mask` says
    which periods were genuinely measured. Scoring uses raw actuals at
    observed periods only.

    A fold whose fit or prediction fails is skipped, not fatal — an early,
    short training window legitimately defeats seasonal models. The
    surviving fold count is reported so the caller can tell a model
    validated on 11 folds from one that scraped through on 3.
    """
    name = model_name or getattr(model_cls, "name", model_cls.__name__)
    result = BacktestResult(model=name, n_folds_attempted=plan.n_folds)
    hyperparams = hyperparams or {}
    started = time.time()

    values = raw_series.to_numpy(dtype=float)
    observed = observed_mask.to_numpy(dtype=bool)
    index = raw_series.index

    for spec in plan.folds:
        train_values = values[:spec.train_end]
        train_observed = observed[:spec.train_end]
        if len(train_values) < MIN_ABSOLUTE_TRAIN:
            continue

        # Preprocessing is fitted here, inside the fold, on training data
        # only — the whole point of Step 5.
        clean_train, _pre = prepare_training_window(train_values, train_observed)
        train_series = pd.Series(clean_train, index=index[:spec.train_end])

        val_slice = slice(spec.val_start, spec.val_end)
        actual_values = values[val_slice]
        actual_observed = observed[val_slice]
        if not actual_observed.any():
            continue  # nothing real to score against in this window

        try:
            model = model_cls()
            model.fit(train_series, frequency=frequency, **hyperparams)
            predicted = np.asarray(model.predict(plan.horizon), dtype=float)
            if len(predicted) < len(actual_values):
                raise ValueError(
                    f"predict({plan.horizon}) returned {len(predicted)} value(s)"
                )
        except Exception as exc:
            logger.debug("%s skipped fold %d (train=%d): %s", name, spec.index, spec.train_end, exc)
            continue

        predicted = predicted[:len(actual_values)]
        keep = actual_observed & np.isfinite(actual_values) & np.isfinite(predicted)
        if not keep.any():
            continue

        fold_actuals = actual_values[keep].tolist()
        fold_preds = predicted[keep].tolist()
        periods = [str(p) for p, k in zip(index[val_slice], keep) if k]

        result.folds.append(FoldOutcome(
            fold=spec.index,
            train_size=spec.train_size,
            origin_period=str(index[spec.train_end - 1]),
            periods=periods,
            actuals=fold_actuals,
            predictions=fold_preds,
            metrics=compute_metrics(np.array(fold_actuals), np.array(fold_preds)),
        ))

    result.runtime_seconds = time.time() - started

    if not result.folds:
        result.error = "No validation fold could be completed for this model."
        return result

    result.pooled_metrics = pool_metrics(result.folds)
    result.fold_stats, result.stability = summarise_folds(result.fold_metric_series("wape"))
    return result


def pool_metrics(folds: Sequence[FoldOutcome]) -> Dict[str, Optional[float]]:
    """
    Pool every fold's (actual, predicted) pairs and compute each metric
    ONCE over the pooled arrays, rather than averaging per-fold metrics.
    Averaging per-fold RMSEs would not be an RMSE; pooling keeps every
    metric's definition intact while still reflecting performance across
    all origins.
    """
    actuals, preds = [], []
    for f in folds:
        actuals.extend(f.actuals)
        preds.extend(f.predictions)
    return compute_metrics(np.array(actuals), np.array(preds))


def summarise_folds(fold_values: Sequence[float]) -> Tuple[Dict[str, Optional[float]], Optional[float]]:
    """
    Step 8 — distribution of a model's per-fold error, plus a single
    fold-to-fold stability score.

    A model scoring 30/31/32/90 is less trustworthy than one scoring
    38/39/40/41 even though their means are close, so the spread is
    reported explicitly (std, min, max, median) instead of being hidden
    behind an average. `stability` is 1 - min(1, std/mean): 1.0 means
    identical error every fold, 0.0 means the spread is as large as the
    mean itself. None when fewer than two folds — variability across one
    fold is not a measurable quantity.
    """
    values = [float(v) for v in fold_values if np.isfinite(v)]
    if not values:
        return {}, None

    arr = np.array(values, dtype=float)
    stats: Dict[str, Optional[float]] = {
        "n_folds": len(values),
        "mean":    round(float(np.mean(arr)), 4),
        "median":  round(float(np.median(arr)), 4),
        "std":     round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "min":     round(float(np.min(arr)), 4),
        "max":     round(float(np.max(arr)), 4),
    }

    if len(arr) < 2:
        return stats, None

    mean = float(np.mean(arr))
    if mean <= 1e-9:
        stability = 1.0
    else:
        stability = max(0.0, 1.0 - min(1.0, float(np.std(arr, ddof=1)) / mean))
    return stats, round(stability, 4)


# ── Ensemble evaluation (Steps 10 & 11) ────────────────────────────────────

@dataclass
class EnsembleBacktest:
    """A candidate blend, scored on the same folds as its components."""
    components: List[str]
    weights: Dict[str, float]
    weight_basis: str
    folds: List[FoldOutcome] = field(default_factory=list)
    pooled_metrics: Optional[Dict[str, Optional[float]]] = None
    fold_stats: Optional[Dict[str, Optional[float]]] = None
    stability: Optional[float] = None

    @property
    def name(self) -> str:
        return f"Weighted Ensemble ({' + '.join(self.components)})"

    def to_dict(self) -> dict:
        return {
            "components":     self.components,
            "weights":        self.weights,
            "weight_basis":   self.weight_basis,
            "pooled_metrics": self.pooled_metrics,
            "fold_stats":     self.fold_stats,
            "stability":      self.stability,
            "n_folds":        len(self.folds),
            "folds":          [f.to_dict() for f in self.folds],
        }


def _fold_key(outcome: FoldOutcome) -> Tuple[int, Tuple[str, ...]]:
    return outcome.fold, tuple(outcome.periods)


def evaluate_ensemble(
    component_results: Sequence[BacktestResult],
    metric: str = "wape",
) -> Optional[EnsembleBacktest]:
    """
    Steps 10 & 11 — build the ensemble's predictions on every fold and
    measure what it ACTUALLY achieved, instead of reporting its best
    component's numbers.

    Weights are leakage-free by construction: the blend used at fold k is
    weighted by the components' inverse pooled error over folds STRICTLY
    BEFORE k. Fold 1 has no history to learn from, so it uses equal
    weights. No fold is ever weighted using its own outcome, and the
    final reported weights (those a production forecast would use) come
    from all folds — which is legitimate, because they are applied to
    future periods, not to the folds that produced them.

    DOCUMENTED LIMITATION: this is expanding-origin weighting, not fully
    nested inner cross-validation. A properly nested scheme would run a
    second walk-forward inside each fold's training window to fit weights,
    multiplying training cost by the inner fold count. Given the series
    lengths this project handles (tens of periods), that would consume
    more history than it could pay back. The scheme here is leakage-free
    with respect to the outer folds, which is the property that matters
    for the reported metrics; it is simply less sample-efficient than
    nesting would be.

    Returns None when fewer than two components share enough folds to
    blend.
    """
    usable = [r for r in component_results if r.folds and r.pooled_metrics]
    if len(usable) < 2:
        return None

    # Only folds every component completed can be blended.
    common = set(_fold_key(f) for f in usable[0].folds)
    for r in usable[1:]:
        common &= set(_fold_key(f) for f in r.folds)
    if not common:
        return None

    by_model: Dict[str, Dict[Tuple[int, Tuple[str, ...]], FoldOutcome]] = {
        r.model: {_fold_key(f): f for f in r.folds if _fold_key(f) in common}
        for r in usable
    }
    ordered_keys = sorted(common, key=lambda k: k[0])
    names = [r.model for r in usable]

    blended_folds: List[FoldOutcome] = []
    for position, key in enumerate(ordered_keys):
        if position == 0:
            weights = {n: 1.0 / len(names) for n in names}
        else:
            # Inverse pooled error over the PRECEDING folds only.
            prior_keys = ordered_keys[:position]
            errors = {}
            for n in names:
                prior = [by_model[n][k] for k in prior_keys]
                pooled = pool_metrics(prior)
                v = pooled.get(metric)
                errors[n] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) and v > 0 else None
            if any(e is None for e in errors.values()):
                weights = {n: 1.0 / len(names) for n in names}
            else:
                inv = {n: 1.0 / errors[n] for n in names}
                total = sum(inv.values())
                weights = {n: inv[n] / total for n in names}

        reference = by_model[names[0]][key]
        stacked = np.array([
            np.array(by_model[n][key].predictions, dtype=float) * weights[n] for n in names
        ])
        blended = stacked.sum(axis=0)
        actuals = np.array(reference.actuals, dtype=float)

        blended_folds.append(FoldOutcome(
            fold=reference.fold,
            train_size=reference.train_size,
            origin_period=reference.origin_period,
            periods=list(reference.periods),
            actuals=actuals.tolist(),
            predictions=blended.tolist(),
            metrics=compute_metrics(actuals, blended),
        ))

    # Weights a production forecast would use: inverse pooled error across
    # every fold. Applied to future periods, never to these folds.
    final_errors = {}
    for r in usable:
        v = (r.pooled_metrics or {}).get(metric)
        final_errors[r.model] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) and v > 0 else None
    if any(e is None for e in final_errors.values()):
        final_weights = {n: round(1.0 / len(names), 4) for n in names}
        basis = "equal weights (at least one component had no usable pooled error)"
    else:
        inv = {n: 1.0 / final_errors[n] for n in names}
        total = sum(inv.values())
        final_weights = {n: round(inv[n] / total, 4) for n in names}
        basis = (
            f"inverse pooled {metric.upper()} across all folds; during backtesting each fold used "
            f"weights derived only from the folds before it (fold 1: equal weights)"
        )

    ensemble = EnsembleBacktest(
        components=names, weights=final_weights, weight_basis=basis, folds=blended_folds,
    )
    ensemble.pooled_metrics = pool_metrics(blended_folds)
    fold_wapes = [
        f.metrics[metric] for f in blended_folds
        if isinstance(f.metrics.get(metric), (int, float)) and np.isfinite(f.metrics[metric])
    ]
    ensemble.fold_stats, ensemble.stability = summarise_folds(fold_wapes)
    return ensemble
