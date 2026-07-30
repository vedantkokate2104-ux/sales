"""
types.py
--------
Shared data structures for the Forecast Intelligence Engine.

Plain dataclasses (not pandas/ML objects) so results serialise cleanly to
JSON for the Flask API and stay easy to unit test in isolation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SeriesStats:
    """Structured result of the Forecast Analyzer for one series — a single
    product, a grouped series, or the whole dataset aggregated by date."""
    group: str
    n_observations: int
    start_date: str
    end_date: str
    frequency: str
    missing_timestamps: int
    missing_values: int
    trend_strength: float
    seasonality_strength: float
    variance: float
    coefficient_of_variation: float
    pct_zero_sales: float
    outlier_percentage: float
    history_length_days: int

    def to_dict(self) -> dict:
        return {
            "group":                 self.group,
            "n_observations":        self.n_observations,
            "start_date":            self.start_date,
            "end_date":              self.end_date,
            "frequency":             self.frequency,
            "missing_timestamps":    self.missing_timestamps,
            "missing_values":        self.missing_values,
            "trend_strength":        self.trend_strength,
            "seasonality_strength":  self.seasonality_strength,
            "variance":              self.variance,
            "coefficient_of_variation": self.coefficient_of_variation,
            "pct_zero_sales":        self.pct_zero_sales,
            "outlier_percentage":    self.outlier_percentage,
            "history_length_days":   self.history_length_days,
        }


@dataclass
class SeriesClassification:
    """Result of the Series Classifier: the series "type" plus the
    reasoning that produced it."""
    series_type: str
    confidence: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "series_type": self.series_type,
            "confidence":  self.confidence,
            "reasons":     self.reasons,
        }


@dataclass
class ModelRecommendation:
    """One ranked candidate model from the Model Selector. Advisory only —
    Phase 1 does not train or execute any of these models."""
    rank: int
    model: str
    reason: str

    def to_dict(self) -> dict:
        return {"rank": self.rank, "model": self.model, "reason": self.reason}
