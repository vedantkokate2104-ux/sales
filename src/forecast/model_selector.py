"""
model_selector.py
------------------
Candidate Model Selector — Step 3 of the Forecast Intelligence Engine.

Maps a series classification to a ranked list of forecasting models that
would be suitable for it. This module ONLY recommends models — it does
not train, run, or evaluate any of them. Execution is deliberately left
for a later phase (see the package docstring in forecast/__init__.py).
"""

from typing import List

from .types import SeriesClassification, SeriesStats, ModelRecommendation

# series_type -> ordered list of (model_name, reason_template).
# reason_template may reference {field_name} placeholders that are filled
# in from the SeriesStats object at recommendation time.
_MODEL_MAP = {
    "New Product": [
        ("Moving Average", "Minimal history available ({n_observations} observations) — "
                            "a simple average is the most reliable baseline until more data accumulates."),
    ],
    "Short History": [
        ("Moving Average", "Only {history_length_days} day(s) of history — insufficient for "
                            "models that need to learn a trend or seasonal cycle."),
    ],
    "Intermittent Demand": [
        ("Croston", "Purpose-built for series with a high proportion of zero-demand periods "
                     "({pct_zero_sales}% zero sales)."),
        ("Moving Average", "Simple fallback baseline if Croston's demand-interval assumptions do not hold."),
    ],
    "Highly Volatile": [
        ("XGBoost", "Volatility (coefficient of variation {coefficient_of_variation}) suggests "
                     "non-linear drivers that a gradient-boosted model can capture better than a "
                     "classical linear method."),
        ("Prophet", "Robust to outliers ({outlier_percentage}% detected) via automatic changepoint detection."),
        ("SARIMA", "Can model volatility if it is driven by autocorrelated noise rather than external factors."),
    ],
    "Trending": [
        ("Holt Linear", "Trend strength {trend_strength} with no material seasonality — Holt's "
                         "linear trend method models this directly."),
        ("Moving Average", "Simple baseline for comparison against the trend-based forecast."),
    ],
    "Seasonal": [
        ("Holt-Winters", "Seasonality strength {seasonality_strength} with no material trend — "
                          "Holt-Winters models a repeating cycle directly."),
        ("SARIMA", "Alternative seasonal model if the cycle is not well captured by simple exponential smoothing."),
    ],
    "Trend + Seasonal": [
        ("Prophet", "Trend strength {trend_strength} and seasonality strength {seasonality_strength} "
                     "together — Prophet decomposes both components and handles missing timestamps "
                     "({missing_timestamps}) gracefully."),
        ("SARIMA", "Classical alternative that jointly models trend and seasonality through differencing."),
        ("Holt-Winters", "Lighter-weight alternative if Prophet/SARIMA are unavailable or over-fit the "
                          "available history."),
    ],
    "Long History": [
        ("SARIMA", "{history_length_days} days of history is enough to reliably fit a seasonal ARIMA model."),
        ("Prophet", "Long history also supports Prophet's automatic seasonality and changepoint detection."),
    ],
    "Smooth": [
        ("Holt Linear", "Stable series (coefficient of variation {coefficient_of_variation}) with no "
                         "strong trend or seasonality — a simple exponential method is sufficient."),
        ("Moving Average", "Lightweight baseline appropriate for a low-variance, pattern-free series."),
    ],
}

_PHASE_NOTE = "Recommendation only — Phase 1 does not train or execute this model."


def recommend_models(classification: SeriesClassification, stats: SeriesStats) -> List[ModelRecommendation]:
    """
    Look up the ranked candidate model list for a series' classified type
    and fill in the reason templates with the actual computed statistics.
    Falls back to the "Smooth" mapping if an unrecognised series_type is
    ever passed in, so this never raises on a valid SeriesClassification.
    """
    entries = _MODEL_MAP.get(classification.series_type, _MODEL_MAP["Smooth"])
    stats_dict = stats.to_dict()

    recommendations = []
    for rank, (model_name, reason_template) in enumerate(entries, start=1):
        try:
            reason = reason_template.format(**stats_dict)
        except (KeyError, IndexError):
            reason = reason_template
        recommendations.append(
            ModelRecommendation(rank=rank, model=model_name, reason=f"{reason} {_PHASE_NOTE}")
        )
    return recommendations
