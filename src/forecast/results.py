"""
results.py
----------
Shared data structures for the Auto Forecasting Engine (Phase 2). Plain
dataclasses, mirroring the style of forecast/types.py from Phase 1, so
everything serialises cleanly to JSON for the Flask API.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ModelEvalResult:
    """Outcome of attempting to train and validate one candidate model."""
    model: str
    available: bool             # dependency present?
    trained: bool                # did fit()/predict() succeed?
    error: Optional[str] = None  # populated if available/trained is False
    metrics: Optional[Dict[str, float]] = None   # MAE / RMSE / MAPE / SMAPE / WAPE (evaluator.compute_metrics)
    rank: Optional[int] = None   # filled in by selector.py after ranking
    # Additive fields (scientific-correctness pass): every existing key
    # above is unchanged, so any consumer reading only those keeps working.
    validation_folds: int = 0          # successful walk-forward folds this model was scored on (0 = trained but not validated)
    training_observations: int = 0     # length of the series (at its detected frequency) this model saw
    stability: Optional[float] = None  # 0-1 walk-forward fold-to-fold consistency score; None if too few folds to estimate
    status: str = ""                   # short machine-readable outcome, e.g. "trained_validated" / "trained_no_validation" / "skipped_unsuitable" / "skipped_unavailable" / "not_implemented" / "failed"
    hyperparameters: Optional[dict] = None  # chosen by this model's own suggest_hyperparameters() (Task 5); None if the model has nothing tunable
    # ── Additive (Phase 2, Steps 7 & 8). Fold-level performance is no
    # longer hidden behind the pooled average: `fold_metrics` carries every
    # fold's own MAE/RMSE/MAPE/SMAPE/WAPE together with its actuals and
    # predictions, and `fold_stats` summarises the spread of per-fold WAPE
    # (mean, median, std, min, max) so a model scoring 30/31/32/90 can be
    # told apart from one scoring 38/39/40/41. ──
    fold_metrics: Optional[List[dict]] = None
    fold_stats: Optional[dict] = None
    runtime_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model":                  self.model,
            "available":              self.available,
            "trained":                self.trained,
            "error":                  self.error,
            "metrics":                self.metrics,
            "rank":                   self.rank,
            "validation_folds":       self.validation_folds,
            "training_observations":  self.training_observations,
            "stability":              self.stability,
            "status":                 self.status,
            "hyperparameters":        self.hyperparameters,
            "fold_metrics":           self.fold_metrics,
            "fold_stats":             self.fold_stats,
            "runtime_seconds":        self.runtime_seconds,
        }


@dataclass
class SelectionResult:
    """Outcome of forecast.selector picking the best-performing model."""
    selected_model: str
    reason: List[str]
    metrics: Optional[Dict[str, float]]
    confidence: float
    ranking: List[ModelEvalResult] = field(default_factory=list)
    # Additive (Task 8 — Weighted Ensemble): populated only when the top two
    # models had statistically similar validation performance and were
    # blended instead of one being picked arbitrarily. Each entry is
    # {"model": <name>, "weight": <0-1 float>}. Internal to forecast_engine
    # (used to build the blended forecast) — not required on the JSON
    # contract, since `selected_model`/`reason` already describe the
    # ensemble in human-readable form.
    ensemble_components: Optional[List[dict]] = None
    # Additive (Task 8 redesign — 8-signal confidence): the same confidence
    # expressed on a 0-100 scale, computed directly (not just confidence*100
    # rounded) so its own component weights can be reasoned about in whole
    # points. `confidence` (0-1) is unchanged and remains the field the
    # existing frontend reads (Task 17).
    confidence_pct: Optional[int] = None
    # Additive (Phase 2, Step 10): the ensemble's OWN backtest — its
    # measured per-fold and pooled metrics, its weights and the basis they
    # were derived from. Populated only when a measured ensemble actually
    # beat the best single model and was therefore selected. `metrics`
    # above now carries the ensemble's real performance in that case,
    # rather than its best component's.
    ensemble_evaluation: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "selected_model":      self.selected_model,
            "reason":              self.reason,
            "metrics":             self.metrics,
            "confidence":          self.confidence,
            "ranking":             [r.to_dict() for r in self.ranking],
            "ensemble_components": self.ensemble_components,
            "confidence_pct":      self.confidence_pct,
            "ensemble_evaluation": self.ensemble_evaluation,
        }


@dataclass
class ForecastPoint:
    period: str
    value: float
    # ── Additive (Phase 4: prediction intervals) ──────────────────────────
    # `period` and `value` are unchanged in name, position and meaning, so
    # every existing consumer — including the chart and month cards in
    # ui/static/js/app.js, which read only those two keys — keeps working
    # untouched. Bounds are None when there is too little out-of-sample
    # error history to estimate an interval (see forecast.intervals), which
    # consumers must render as "unavailable" rather than as zero.
    lower_80: Optional[float] = None
    upper_80: Optional[float] = None
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None
    interval_width_80: Optional[float] = None
    interval_width_95: Optional[float] = None

    def to_dict(self) -> dict:
        payload = {"period": self.period, "value": self.value}
        if self.lower_80 is not None:
            payload.update({
                "lower_80": self.lower_80,
                "upper_80": self.upper_80,
                "interval_width_80": self.interval_width_80,
                "prediction_interval_80": [self.lower_80, self.upper_80],
            })
        if self.lower_95 is not None:
            payload.update({
                "lower_95": self.lower_95,
                "upper_95": self.upper_95,
                "interval_width_95": self.interval_width_95,
                "prediction_interval_95": [self.lower_95, self.upper_95],
            })
        return payload


@dataclass
class AutoForecastResult:
    """Top-level result returned by forecast_engine.run_auto_forecast() —
    everything the Forecast page needs to render Phase 2's UI in one
    payload."""
    horizon: int
    selected_model: str
    selection_reason: List[str]
    evaluation_metrics: Optional[Dict[str, float]]
    forecast_confidence: float
    model_ranking: List[dict]
    forecast: List[dict]
    alternative_models: List[str]
    explainability: Dict[str, object]
    warnings: List[str] = field(default_factory=list)
    # Additive field (Phase E, Task 1): the dataset's detected frequency
    # (daily/weekly/monthly/quarterly/yearly/irregular), so a consumer can
    # tell what cadence `horizon` counts in (e.g. horizon=3 at "weekly"
    # frequency means 3 weeks, not 3 months). Placed last and given a
    # default so any existing caller building this dataclass positionally
    # without it keeps working unchanged.
    frequency: str = "monthly"
    # Additive (Task 8 redesign): forecast_confidence expressed on a 0-100
    # scale directly from selector.py's 8-signal breakdown. Optional /
    # defaulted so any existing caller building this dataclass without it
    # keeps working unchanged (Task 17).
    forecast_confidence_pct: Optional[int] = None
    # Additive (Task 2/11 redesign): 0-1 summary of how much correction the
    # raw input data needed before training (see forecast.preprocessing).
    data_quality_score: Optional[float] = None
    # ── Additive (Phase 1: forecast grain selection). Every field above is
    # unchanged in name, type and meaning, so the existing frontend and any
    # other API consumer keep working untouched. `frequency` in particular
    # is retained and always equals `selected_frequency`. ──
    selected_frequency: Optional[str] = None      # grain the series was modelled at
    frequency_reason: Optional[str] = None        # why that grain won over the alternatives
    series_density: Optional[float] = None        # 0-1: share of modelled periods containing real transactions
    zero_period_ratio: Optional[float] = None     # 0-1: share of modelled periods totalling zero
    candidate_frequencies: Optional[Dict[str, dict]] = None  # per-grain diagnostics (see forecast.grain)
    n_modelled_periods: Optional[int] = None      # length of the series actually trained on
    n_transactions: Optional[int] = None          # raw ERP rows behind that series
    # ── Additive (Phase 2, Step 12: explicit horizon semantics) ───────────
    # `horizon` above is retained unchanged for existing consumers. These
    # spell out what it actually means, because the API parameter is named
    # "months" while periods follow the SELECTED grain — so a requested
    # horizon of 3 on a weekly series is 3 weeks, not 3 months. Rather
    # than renaming the public parameter, the resolved meaning is stated:
    #   requested_horizon = 3, forecast_periods = 3, forecast_unit = "month"
    # `validation_horizon` is how many steps ahead each walk-forward fold
    # was actually scored at; it equals forecast_periods except on series
    # too short to support that, where the mismatch is flagged rather than
    # hidden.
    requested_horizon: Optional[int] = None
    forecast_periods: Optional[int] = None
    forecast_unit: Optional[str] = None
    validation_horizon: Optional[int] = None
    horizon_note: Optional[str] = None
    # Additive (Phase 2, Steps 4/7/8/10): the fold plan every model was
    # scored against, and the ensemble's own measured backtest when one
    # was selected.
    validation: Optional[Dict[str, object]] = None
    ensemble_evaluation: Optional[dict] = None
    # ── Additive (Phase 4: prediction intervals) ──────────────────────────
    # `prediction_intervals` carries the full report from
    # forecast.intervals — method, per-level empirical coverage, sharpness,
    # calibration status and the caveats attached to them. The per-period
    # bounds also ride along inside each `forecast` entry so a consumer
    # that only reads the forecast list still gets them.
    #
    # These are a DIFFERENT QUANTITY from `forecast_confidence`, which
    # remains untouched: confidence is a 0-1 heuristic reliability score
    # built from eight signals about the model and the data, and it is not
    # a probability that the forecast is correct. A prediction interval is
    # a range with a stated nominal coverage level measured against actual
    # out-of-sample outcomes. Phase 4 deliberately does not let one move
    # the other.
    prediction_intervals: Optional[dict] = None
    interval_method: Optional[str] = None
    interval_calibration_status: Optional[str] = None
    coverage_80: Optional[float] = None
    coverage_95: Optional[float] = None
    coverage_sample_size: Optional[int] = None
    # ── Additive (Phase 5: grouped / segment-level forecasting) ───────────
    # All None on a company-total forecast, which is what every existing
    # client already receives — so omitting group_col/group_value from a
    # request produces byte-identical behaviour to before Phase 5.
    #
    # `group_status` is the important one: it distinguishes a real segment
    # forecast (OK) from an unknown column, unknown value or empty segment.
    # Before Phase 5 an unknown column silently returned the COMPANY
    # forecast, so a consumer had no way to tell it was being shown the
    # wrong series under the label it asked for.
    group_col: Optional[str] = None
    group_value: Optional[str] = None
    group_label: Optional[str] = None
    group_status: Optional[str] = None
    group_records: Optional[int] = None
    group_message: Optional[str] = None
    group_history_status: Optional[str] = None
    group_history_note: Optional[str] = None
    # ── Additive (Phase 6: reliability and explainability) ────────────────
    # `forecast_confidence` and `forecast_confidence_pct` above are
    # UNCHANGED and still populated by selector._compute_confidence(), so
    # any client reading them keeps working. They are, however, a weighted
    # heuristic that was never calibrated against outcomes, and the UI used
    # to render one as "69%" beside the word confidence — which reads as a
    # probability the system cannot support.
    #
    # `reliability` supersedes them for display: the same idea, but scored
    # from measured signals (including Phase 4 interval calibration and
    # Phase 1 series density, neither of which the confidence score sees),
    # carrying a named level, a per-factor breakdown, and an explicit
    # `is_probability: false`.
    reliability: Optional[dict] = None
    model_explanation: Optional[dict] = None
    forecast_insights: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "horizon":                  self.horizon,
            "selected_model":           self.selected_model,
            "selection_reason":         self.selection_reason,
            "evaluation_metrics":       self.evaluation_metrics,
            "forecast_confidence":      self.forecast_confidence,
            "model_ranking":            self.model_ranking,
            "forecast":                 self.forecast,
            "alternative_models":       self.alternative_models,
            "explainability":           self.explainability,
            "warnings":                 self.warnings,
            "frequency":                self.frequency,
            "forecast_confidence_pct":  self.forecast_confidence_pct,
            "data_quality_score":       self.data_quality_score,
            "selected_frequency":       self.selected_frequency,
            "frequency_reason":         self.frequency_reason,
            "series_density":           self.series_density,
            "zero_period_ratio":        self.zero_period_ratio,
            "candidate_frequencies":    self.candidate_frequencies,
            "n_modelled_periods":       self.n_modelled_periods,
            "n_transactions":           self.n_transactions,
            "requested_horizon":        self.requested_horizon,
            "forecast_periods":         self.forecast_periods,
            "forecast_unit":            self.forecast_unit,
            "validation_horizon":       self.validation_horizon,
            "horizon_note":             self.horizon_note,
            "validation":               self.validation,
            "ensemble_evaluation":      self.ensemble_evaluation,
            "prediction_intervals":         self.prediction_intervals,
            "interval_method":              self.interval_method,
            "interval_calibration_status":  self.interval_calibration_status,
            "coverage_80":                  self.coverage_80,
            "coverage_95":                  self.coverage_95,
            "coverage_sample_size":         self.coverage_sample_size,
            "group_col":                    self.group_col,
            "group_value":                  self.group_value,
            "group_label":                  self.group_label,
            "group_status":                 self.group_status,
            "group_records":                self.group_records,
            "group_message":                self.group_message,
            "group_history_status":         self.group_history_status,
            "group_history_note":           self.group_history_note,
            "reliability":                  self.reliability,
            "model_explanation":            self.model_explanation,
            "forecast_insights":            self.forecast_insights,
        }
