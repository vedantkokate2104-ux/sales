"""
experiments.py
---------------
Controlled feature and model ablations (Phase 3, Steps 8-13).

Every experiment runs through the UNCHANGED Phase 2 protocol: the same
`plan_validation()` fold layout, the same `backtest_model()` loop, the
same per-fold preprocessing, the same metrics. Nothing here adjusts a
validation setting, and nothing here can — the plan is built once from
the series and handed to every candidate. That is the only way the
resulting table means anything: if the protocol moved between rows, a
lower WAPE would prove nothing.

The output is deliberately blunt about failure. An experiment that makes
accuracy worse is reported with its number, marked REJECTED, and kept in
the table. A feature family only earns its place by lowering
out-of-sample error against the SES baseline, and the report says so
either way.

OVERFITTING (Step 11)
----------------------
For every ML candidate the runner records in-sample training WAPE
alongside validation WAPE, plus the gap between them. A configuration
whose training error is far below its validation error is memorising its
training window; `overfit_ratio` makes that visible instead of leaving it
to be inferred from a suspiciously good fit.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import (
    BacktestResult,
    ValidationPlan,
    backtest_model,
    evaluate_ensemble,
    plan_validation,
    prepare_training_window,
)
from .evaluator import wape
from .features import FeatureConfig
from .grain import SeriesConstruction
from .models import MODEL_REGISTRY
from .models.tabular import ESTIMATOR_FACTORIES, make_tabular_model
from .utils import seasonal_period_for_freq

logger = logging.getLogger(__name__)

# A candidate whose validation WAPE exceeds its training WAPE by more
# than this multiple is flagged as overfitting. Not a hard rejection on
# its own — a model can overfit and still be the most accurate thing
# available — but it is reported, and it is the tie-breaker when two
# candidates validate similarly.
OVERFIT_RATIO_THRESHOLD = 2.0

# A candidate must beat the baseline by at least this many percentage
# points of WAPE to be treated as a genuine improvement rather than fold
# noise. With ~11 folds on 48 periods, smaller differences are not
# separable from which months happened to land in which fold.
MIN_MEANINGFUL_IMPROVEMENT_PP = 1.0


@dataclass
class ExperimentResult:
    """One row of the Phase 3 benchmark table."""
    experiment: str
    model: str
    features: str
    wape: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    smape: Optional[float] = None
    mape: Optional[float] = None
    fold_std: Optional[float] = None
    fold_mean: Optional[float] = None
    fold_min: Optional[float] = None
    fold_max: Optional[float] = None
    stability: Optional[float] = None
    training_error: Optional[float] = None
    validation_error: Optional[float] = None
    overfit_ratio: Optional[float] = None
    n_folds: int = 0
    n_points: int = 0
    n_features: Optional[int] = None
    runtime_seconds: float = 0.0
    status: str = "CANDIDATE"
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ExperimentReport:
    frequency: str
    n_periods: int
    plan: ValidationPlan
    baseline_model: str
    baseline_wape: Optional[float] = None
    results: List[ExperimentResult] = field(default_factory=list)
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    ensemble_note: Optional[str] = None
    total_runtime_seconds: float = 0.0

    @property
    def best(self) -> Optional[ExperimentResult]:
        scored = [r for r in self.results if r.wape is not None]
        return min(scored, key=lambda r: r.wape) if scored else None

    def to_dict(self) -> dict:
        return {
            "frequency":      self.frequency,
            "n_periods":      self.n_periods,
            "validation":     self.plan.to_dict(),
            "baseline_model": self.baseline_model,
            "baseline_wape":  self.baseline_wape,
            "results":        [r.to_dict() for r in self.results],
            "ensemble_note":  self.ensemble_note,
            "best_model":     self.best.model if self.best else None,
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
        }


def _training_error_for(model_cls, series: pd.Series, observed: pd.Series,
                        plan: ValidationPlan, frequency: str) -> Optional[float]:
    """
    In-sample WAPE on the LAST fold's training window (Step 11).

    Uses the same `prepare_training_window()` the backtest uses, so the
    training data measured here is exactly what that fold's model saw.
    Models exposing `training_error_` (the tabular family) report their
    own; classical models are re-fit and scored on one-step-ahead
    in-sample predictions instead.
    """
    if not plan.folds:
        return None
    spec = plan.folds[-1]
    values = series.to_numpy(dtype=float)[:spec.train_end]
    mask = observed.to_numpy(dtype=bool)[:spec.train_end]
    clean, _pre = prepare_training_window(values, mask)
    train = pd.Series(clean, index=series.index[:spec.train_end])

    try:
        model = model_cls()
        model.fit(train, frequency=frequency)
    except Exception:
        return None

    reported = getattr(model, "training_error_", None)
    if reported is not None:
        return round(float(reported), 2)

    # Classical models: walk one step at a time through the training
    # window, refitting on the prefix. Expensive but only done once per
    # candidate, and it is the like-for-like counterpart of the in-sample
    # error the tabular models report.
    actuals, preds = [], []
    start = max(2, len(train) - 12)
    for cut in range(start, len(train)):
        try:
            probe = model_cls()
            probe.fit(train.iloc[:cut], frequency=frequency)
            preds.append(float(probe.predict(1)[0]))
            actuals.append(float(train.iloc[cut]))
        except Exception:
            continue
    if not actuals:
        return None
    value = wape(np.array(actuals), np.array(preds))
    return round(float(value), 2) if value is not None else None


def _result_from_backtest(
    experiment: str, features: str, result: BacktestResult,
    training_error: Optional[float], n_features: Optional[int],
) -> ExperimentResult:
    pooled = result.pooled_metrics or {}
    stats = result.fold_stats or {}
    validation_error = pooled.get("wape")
    overfit_ratio = None
    if training_error and validation_error and training_error > 0:
        overfit_ratio = round(validation_error / training_error, 2)

    return ExperimentResult(
        experiment=experiment,
        model=result.model,
        features=features,
        wape=validation_error,
        mae=pooled.get("mae"),
        rmse=pooled.get("rmse"),
        smape=pooled.get("smape"),
        mape=pooled.get("mape"),
        fold_std=stats.get("std"),
        fold_mean=stats.get("mean"),
        fold_min=stats.get("min"),
        fold_max=stats.get("max"),
        stability=result.stability,
        training_error=training_error,
        validation_error=validation_error,
        overfit_ratio=overfit_ratio,
        n_folds=result.n_folds,
        n_points=result.n_points,
        n_features=n_features,
        runtime_seconds=round(result.runtime_seconds, 2),
    )


def _feature_count(model_cls, series: pd.Series, frequency: str) -> Optional[int]:
    try:
        probe = model_cls()
        probe.fit(series, frequency=frequency)
        return len(getattr(probe, "feature_names_", []) or []) or None
    except Exception:
        return None


def run_experiments(
    construction: SeriesConstruction,
    horizon: int = 3,
    baseline_model: str = "Simple Exponential Smoothing",
    include_classical: bool = True,
    tabular_specs: Optional[Sequence[dict]] = None,
    measure_training_error: bool = True,
) -> ExperimentReport:
    """
    Run the Step 8 ablations plus the Step 9 model comparison.

    `tabular_specs` is a list of {experiment, name, estimator, config,
    hyperparams} dicts. Defaults to the A-F ladder described in Phase 3
    when omitted.
    """
    frequency = construction.frequency
    series = construction.series
    observed = construction.observed_mask
    n = len(series)

    plan = plan_validation(n, horizon, seasonal_period_for_freq(frequency))
    report = ExperimentReport(
        frequency=frequency, n_periods=n, plan=plan, baseline_model=baseline_model,
    )
    started = time.time()

    # ── Experiment A and the classical comparison set ─────────────────────
    if include_classical:
        for name in ("Simple Exponential Smoothing", "Moving Average", "Holt Linear",
                     "Naive", "Seasonal Naive", "Holt-Winters", "SARIMA"):
            model_cls = MODEL_REGISTRY.get(name)
            if model_cls is None or not model_cls.is_available():
                continue
            try:
                ok, _msg = model_cls().check_requirements(series, frequency)
            except Exception:
                ok = False
            if not ok:
                continue

            result = backtest_model(series, observed, model_cls, plan, frequency, model_name=name)
            if not result.folds:
                continue
            report.backtests[name] = result
            training_error = (
                _training_error_for(model_cls, series, observed, plan, frequency)
                if measure_training_error else None
            )
            row = _result_from_backtest(
                "A: univariate baseline", "none (univariate)", result, training_error, None,
            )
            if name == baseline_model:
                row.status = "BASELINE"
                report.baseline_wape = row.wape
            report.results.append(row)

    # ── Experiments B-F: feature families x estimators ────────────────────
    if tabular_specs is None:
        tabular_specs = default_tabular_specs()

    for spec in tabular_specs:
        config: FeatureConfig = spec["config"]
        estimator_name = spec["estimator"]
        factory = ESTIMATOR_FACTORIES.get(estimator_name)
        if factory is None:
            continue

        label = spec["name"]
        model_cls = make_tabular_model(label, factory, config)

        try:
            ok, message = model_cls().check_requirements(series, frequency)
        except Exception as exc:
            ok, message = False, str(exc)
        if not ok:
            report.results.append(ExperimentResult(
                experiment=spec["experiment"], model=label, features=config.describe(),
                status="SKIPPED", note=message,
            ))
            continue

        result = backtest_model(
            series, observed, model_cls, plan, frequency,
            hyperparams=spec.get("hyperparams") or {}, model_name=label,
        )
        if not result.folds:
            report.results.append(ExperimentResult(
                experiment=spec["experiment"], model=label, features=config.describe(),
                status="FAILED", note=result.error,
            ))
            continue

        report.backtests[label] = result
        training_error = (
            _training_error_for(model_cls, series, observed, plan, frequency)
            if measure_training_error else None
        )
        report.results.append(_result_from_backtest(
            spec["experiment"], config.describe(), result, training_error,
            _feature_count(model_cls, series, frequency),
        ))

    # ── Verdicts against the baseline (Steps 13 & 16) ─────────────────────
    baseline = report.baseline_wape
    for row in report.results:
        if row.status in ("BASELINE", "SKIPPED", "FAILED") or row.wape is None:
            continue
        if baseline is None:
            row.status = "CANDIDATE"
            continue
        improvement = baseline - row.wape
        if improvement >= MIN_MEANINGFUL_IMPROVEMENT_PP:
            row.status = "ACCEPTED"
            row.note = f"beats baseline by {improvement:.2f}pp"
        elif improvement > 0:
            row.status = "REJECTED"
            row.note = (
                f"only {improvement:.2f}pp better than baseline — below the "
                f"{MIN_MEANINGFUL_IMPROVEMENT_PP}pp threshold for a separable improvement"
            )
        else:
            row.status = "REJECTED"
            row.note = f"{abs(improvement):.2f}pp worse than baseline"
        if row.overfit_ratio and row.overfit_ratio >= OVERFIT_RATIO_THRESHOLD:
            row.note = (row.note or "") + (
                f"; validation error is {row.overfit_ratio}x training error — overfitting"
            )

    report.results.sort(key=lambda r: (r.wape if r.wape is not None else float("inf")))
    report.total_runtime_seconds = time.time() - started
    return report


def default_tabular_specs() -> List[dict]:
    """
    The Step 8 ablation ladder. Each experiment adds ONE feature family to
    an otherwise identical setup, so any change in validation error is
    attributable to that family rather than to a bundle of simultaneous
    edits.

    Ridge is carried alongside the tree models throughout: with ~12 to 36
    training rows a linear model with shrinkage is the more appropriate
    prior, and including it keeps the comparison from being a contest
    between two flavours of the same overfitting risk.
    """
    lags = FeatureConfig(use_lags=True)
    rolling = FeatureConfig(use_rolling=True)
    trend = FeatureConfig(use_trend=True)
    calendar = FeatureConfig(use_calendar=True)
    combined = FeatureConfig(use_lags=True, use_rolling=True, use_trend=True, use_calendar=True)
    lean = FeatureConfig(use_lags=True, use_rolling=True)

    specs: List[dict] = []
    for experiment, config, tag in (
        ("B: + lag features", lags, "lags"),
        ("C: + rolling features", rolling, "rolling"),
        ("D: + trend features", trend, "trend"),
        ("E: + calendar features", calendar, "calendar"),
        ("F: ML + selected features", lean, "lags+rolling"),
        ("F: ML + all families", combined, "all"),
    ):
        for estimator in ("Ridge", "XGBoost"):
            specs.append({
                "experiment": experiment,
                "name": f"{estimator} ({tag})",
                "estimator": estimator,
                "config": config,
            })

    # Step 9's optional estimators, on the leanest feature set that the
    # ablations above are likely to favour.
    for estimator in ("ElasticNet", "RandomForest", "GradientBoosting"):
        specs.append({
            "experiment": "F: ML + selected features",
            "name": f"{estimator} (lags+rolling)",
            "estimator": estimator,
            "config": lean,
        })

    return specs


def format_report(report: ExperimentReport) -> str:
    """Render the Step 15 table as plain text."""
    p = report.plan
    lines = [
        f"Frequency: {report.frequency} | periods: {report.n_periods} | folds: {p.n_folds} | "
        f"validation points: {p.n_validation_points} | horizon: {p.horizon} | "
        f"first training window: {p.min_train_size}",
        f"Baseline: {report.baseline_model} at "
        f"{report.baseline_wape if report.baseline_wape is not None else 'n/a'}% WAPE",
        "",
    ]
    header = (f"{'Model':<30}{'Features':<14}{'WAPE%':>8}{'MAE':>11}{'RMSE':>11}{'SMAPE%':>8}"
              f"{'FoldStd':>8}{'Train%':>8}{'Val%':>8}{'O/F':>6}{'Feat':>5}{'Sec':>7}  Status")
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.results:
        lines.append(
            f"{r.model:<30}{r.features:<14}{_f(r.wape,2):>8}{_f(r.mae,0):>11}{_f(r.rmse,0):>11}"
            f"{_f(r.smape,2):>8}{_f(r.fold_std,2):>8}{_f(r.training_error,2):>8}"
            f"{_f(r.validation_error,2):>8}{_f(r.overfit_ratio,2):>6}"
            f"{(r.n_features if r.n_features else '-'):>5}{r.runtime_seconds:>7.1f}  {r.status}"
        )
    notes = [r for r in report.results if r.note]
    if notes:
        lines.append("")
        lines.append("Notes:")
        for r in notes:
            lines.append(f"  {r.model}: {r.note}")
    if report.ensemble_note:
        lines.append("")
        lines.append(report.ensemble_note)
    return "\n".join(lines)


def _f(value, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"
