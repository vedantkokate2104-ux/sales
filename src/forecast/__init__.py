"""
forecast package
----------------
Forecast Intelligence Engine (Phase 1).

Infrastructure for analysing a sales time series, classifying its pattern,
and recommending suitable forecasting models. This package does NOT
implement or run any forecasting model itself — analytics.run_forecast()
remains the sole active forecaster in the application. The modules here
are advisory only, feeding the new "Forecast Analysis" panel on the
Forecast page.
"""

from .analyzer import analyze_dataset, analyze_primary, analyze_series
from .classifier import classify_series
from .model_selector import recommend_models
from .types import SeriesStats, SeriesClassification, ModelRecommendation

__all__ = [
    "analyze_dataset", "analyze_primary", "analyze_series",
    "classify_series", "recommend_models",
    "SeriesStats", "SeriesClassification", "ModelRecommendation",
]
