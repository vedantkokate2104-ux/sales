"""
types.py
--------
Shared data structures for the Forecast Intelligence Engine.

Plain dataclasses (not pandas/ML objects) so results serialise cleanly to
JSON for the Flask API and stay easy to unit test in isolation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SeriesStats:
    """
    Structured result of the Forecast Analyzer for one series — a single
    product, a grouped series, or the whole dataset aggregated by date.

    Phase 1, Step 7: every statistic here now describes the MODELLED
    FORECASTING SERIES (periods at the selected forecast grain), not the
    raw transaction rows. Specifically:

      n_observations   — number of modelled PERIODS, not invoice count.
      pct_zero_sales   — share of modelled periods whose total is zero,
                          not share of individual invoices worth zero.
      coefficient_of_variation / variance / trend_strength /
      seasonality_strength / outlier_percentage
                       — all computed on the aggregated period series.

    Previously these were computed on individual transaction rows, so
    `n_observations` was an invoice count and `coefficient_of_variation`
    measured invoice-to-invoice spread — neither of which describes the
    thing a forecasting model is actually asked to predict, yet both were
    driving series classification, candidate filtering and confidence.

    The raw transaction count is still available as `n_transactions` so
    nothing is lost, it is just no longer mistaken for the series length.
    """
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
    # ── Additive fields (Phase 1). All defaulted, so any existing caller
    # constructing SeriesStats positionally without them keeps working. ──
    n_transactions: int = 0          # raw ERP rows behind this series
    observed_periods: int = 0        # periods containing >= 1 transaction
    empty_periods: int = 0           # periods with NO transaction recorded (missing, not zero)
    series_density: float = 1.0      # observed_periods / n_observations
    zero_period_ratio: float = 0.0   # 0-1 form of pct_zero_sales, for API consumers
    seasonal_period: Optional[int] = None   # cycle length seasonality was measured at; None if untestable
    fill_strategy: str = ""          # how missing periods were handled (see forecast.grain)

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
            "n_transactions":        self.n_transactions,
            "observed_periods":      self.observed_periods,
            "empty_periods":         self.empty_periods,
            "series_density":        self.series_density,
            "zero_period_ratio":     self.zero_period_ratio,
            "seasonal_period":       self.seasonal_period,
            "fill_strategy":         self.fill_strategy,
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
