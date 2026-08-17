"""
benchmark.py
-------------
Model benchmarking against the honest validation protocol (Phase 2,
Steps 13 & 14).

Distinct from `trainer.train_candidates()`, which only evaluates the
models Phase 1 RECOMMENDS for a series' type plus the guaranteed
baselines. This runs EVERY model in the registry that can run on the
given series, against the same shared fold plan, so a like-for-like
comparison table can be produced — including models the recommender
would not have shortlisted. That matters for auditing the recommender
itself: if Holt-Winters beats every shortlisted candidate on a series
type that never nominates it, that is a finding, not an accident.

A model that cannot run (missing optional dependency, or not enough
history for its own requirements) is reported separately with the
reason rather than failing the whole benchmark.

Nothing here feeds the production forecast. It is a measurement tool.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .backtest import (
    BacktestResult,
    EnsembleBacktest,
    ValidationPlan,
    backtest_model,
    evaluate_ensemble,
    plan_validation,
)
from .grain import SeriesConstruction, build_forecast_series, select_forecast_grain
from .models import MODEL_REGISTRY
from .utils import effective_training_frequency, seasonal_period_for_freq

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRow:
    """One model's line in the benchmark table."""
    model: str
    wape: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    fold_std: Optional[float] = None
    fold_mean: Optional[float] = None
    fold_median: Optional[float] = None
    fold_min: Optional[float] = None
    fold_max: Optional[float] = None
    stability: Optional[float] = None
    n_folds: int = 0
    n_points: int = 0
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BenchmarkReport:
    frequency: str
    n_periods: int
    plan: ValidationPlan
    rows: List[BenchmarkRow] = field(default_factory=list)
    unavailable: Dict[str, str] = field(default_factory=dict)
    ensemble: Optional[EnsembleBacktest] = None
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    total_runtime_seconds: float = 0.0

    @property
    def best(self) -> Optional[BenchmarkRow]:
        ranked = [r for r in self.rows if r.wape is not None]
        return min(ranked, key=lambda r: r.wape) if ranked else None

    @property
    def seasonal_naive(self) -> Optional[BenchmarkRow]:
        return next((r for r in self.rows if r.model == "Seasonal Naive"), None)

    def to_dict(self) -> dict:
        return {
            "frequency":             self.frequency,
            "n_periods":             self.n_periods,
            "validation":            self.plan.to_dict(),
            "rows":                  [r.to_dict() for r in self.rows],
            "unavailable":           self.unavailable,
            "ensemble":              self.ensemble.to_dict() if self.ensemble else None,
            "best_model":            self.best.model if self.best else None,
            "seasonal_naive_wape":   self.seasonal_naive.wape if self.seasonal_naive else None,
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
        }


def _row_from_backtest(result: BacktestResult) -> BenchmarkRow:
    pooled = result.pooled_metrics or {}
    stats = result.fold_stats or {}
    return BenchmarkRow(
        model=result.model,
        wape=pooled.get("wape"),
        mae=pooled.get("mae"),
        rmse=pooled.get("rmse"),
        mape=pooled.get("mape"),
        smape=pooled.get("smape"),
        fold_std=stats.get("std"),
        fold_mean=stats.get("mean"),
        fold_median=stats.get("median"),
        fold_min=stats.get("min"),
        fold_max=stats.get("max"),
        stability=result.stability,
        n_folds=result.n_folds,
        n_points=result.n_points,
        runtime_seconds=result.runtime_seconds,
    )


def benchmark_series(
    construction: SeriesConstruction,
    horizon: int = 3,
    model_names: Optional[List[str]] = None,
    tune: bool = False,
) -> BenchmarkReport:
    """
    Backtest every requested model against one shared fold plan.

    `tune=False` (the default) runs each model on its own defaults. Phase 2
    is about making validation trustworthy, not about squeezing out
    hyperparameters — tuning is deliberately left off so the comparison
    reflects each model's out-of-the-box behaviour and stays cheap enough
    to run repeatedly. Set tune=True to let each model's
    suggest_hyperparameters() run once on the first fold's training prefix.
    """
    frequency = effective_training_frequency(construction.frequency)
    raw_series = construction.series
    observed = construction.observed_mask
    n = len(raw_series)

    plan = plan_validation(n, horizon, seasonal_period_for_freq(frequency))
    report = BenchmarkReport(frequency=frequency, n_periods=n, plan=plan)

    names = model_names if model_names is not None else list(MODEL_REGISTRY.keys())
    started = time.time()

    for name in names:
        model_cls = MODEL_REGISTRY.get(name)
        if model_cls is None:
            report.unavailable[name] = "Not implemented in the Forecast Model Library."
            continue
        if not model_cls.is_available():
            report.unavailable[name] = "Required optional dependency is not installed."
            continue

        try:
            ok, message = model_cls().check_requirements(raw_series, frequency)
        except Exception as exc:
            report.unavailable[name] = f"Suitability check failed: {exc}"
            continue
        if not ok:
            report.unavailable[name] = message or "Not suitable for this series."
            continue

        hyperparams = {}
        if tune and plan.n_folds:
            try:
                prefix = raw_series.iloc[:plan.min_train_size]
                hyperparams = model_cls().suggest_hyperparameters(prefix, frequency) or {}
            except Exception as exc:
                logger.debug("%s: tuning failed, using defaults (%s)", name, exc)

        try:
            result = backtest_model(
                raw_series, observed, model_cls, plan, frequency,
                hyperparams=hyperparams, model_name=name,
            )
        except Exception as exc:
            logger.exception("%s: benchmark run failed", name)
            report.unavailable[name] = f"Backtest failed: {exc}"
            continue

        if not result.folds:
            report.unavailable[name] = result.error or "No validation fold could be completed."
            continue

        report.backtests[name] = result
        report.rows.append(_row_from_backtest(result))

    report.rows.sort(key=lambda r: (r.wape if r.wape is not None else float("inf")))

    # Ensemble of the leading models, measured on the same folds — never
    # assumed to be better (see backtest.evaluate_ensemble).
    top = [report.backtests[r.model] for r in report.rows[:3] if r.model in report.backtests]
    if len(top) >= 2:
        report.ensemble = evaluate_ensemble(top)

    report.total_runtime_seconds = time.time() - started
    return report


def benchmark_dataframe(
    df: pd.DataFrame, mappings: list, horizon: int = 3, tune: bool = False,
) -> BenchmarkReport:
    """Convenience wrapper: resolve columns, select the forecast grain the
    production pipeline would use, and benchmark every model on it."""
    from analytics import get_col

    date_col = get_col(df, mappings, "invoice date")
    rev_col = get_col(df, mappings, "net sales")
    if not date_col or not rev_col:
        raise ValueError("Benchmarking requires a resolved invoice date and net sales column.")

    selection = select_forecast_grain(df[date_col], df[rev_col], horizon=horizon)
    construction = build_forecast_series(df[date_col], df[rev_col], selection.selected_frequency)
    return benchmark_series(construction, horizon=horizon, tune=tune)


def format_report(report: BenchmarkReport) -> str:
    """Render the Step 14 table as plain text."""
    lines: List[str] = []
    p = report.plan
    lines.append(
        f"Frequency: {report.frequency} | modelled periods: {report.n_periods} | "
        f"folds: {p.n_folds} | validation points: {p.n_validation_points} | "
        f"validation horizon: {p.horizon} | first training window: {p.min_train_size}"
    )
    lines.append(f"Fold rule: {p.rule}")
    if p.degraded and p.degradation_reason:
        lines.append(f"DEGRADED: {p.degradation_reason}")
    lines.append("")

    header = (f"{'Model':<30}{'WAPE%':>9}{'MAE':>13}{'RMSE':>13}{'SMAPE%':>9}"
              f"{'FoldStd':>9}{'Stab':>7}{'Folds':>7}{'Runtime':>9}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.rows:
        lines.append(
            f"{r.model:<30}{_f(r.wape, 2):>9}{_f(r.mae, 0):>13}{_f(r.rmse, 0):>13}"
            f"{_f(r.smape, 2):>9}{_f(r.fold_std, 2):>9}{_f(r.stability, 3):>7}"
            f"{r.n_folds:>7}{r.runtime_seconds:>8.1f}s"
        )

    if report.unavailable:
        lines.append("")
        lines.append("Not benchmarked:")
        for name, reason in report.unavailable.items():
            lines.append(f"  {name}: {reason}")

    if report.ensemble and report.ensemble.pooled_metrics:
        em = report.ensemble.pooled_metrics
        lines.append("")
        lines.append(f"Ensemble ({' + '.join(report.ensemble.components)}):")
        lines.append(
            f"  WAPE {em.get('wape')}%  MAE {em.get('mae')}  RMSE {em.get('rmse')}  "
            f"SMAPE {em.get('smape')}%  over {len(report.ensemble.folds)} folds"
        )
        lines.append(f"  weights: {report.ensemble.weights}")
        best = report.best
        if best and em.get("wape") is not None and best.wape is not None:
            verdict = "BETTER than" if em["wape"] < best.wape else "WORSE than"
            lines.append(f"  {verdict} the best single model ({best.model}, {best.wape}%)")

    return "\n".join(lines)


def _f(value, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"
