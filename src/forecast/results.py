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

    def to_dict(self) -> dict:
        return {
            "selected_model":      self.selected_model,
            "reason":              self.reason,
            "metrics":             self.metrics,
            "confidence":          self.confidence,
            "ranking":             [r.to_dict() for r in self.ranking],
            "ensemble_components": self.ensemble_components,
            "confidence_pct":      self.confidence_pct,
        }


@dataclass
class ForecastPoint:
    period: str
    value: float

    def to_dict(self) -> dict:
        return {"period": self.period, "value": self.value}


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
        }
