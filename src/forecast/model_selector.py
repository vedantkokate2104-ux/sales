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
from .utils import seasonal_period_for_freq

# Forecasting-quality pass (data-sufficiency-aware recommendations):
# _MODEL_MAP below picks candidates from a series' TYPE (trend/seasonal/
# volatility signature) but was previously blind to how much history is
# actually available to fit them on. That gap matters because "type" and
# "data volume" are independent: a monthly series can show a textbook
# trend+seasonal signature after as few as ~20 points, which is a strong
# enough SIGNAL for classification but not necessarily enough DATA for a
# multi-parameter model (SARIMA's differencing/AR/MA orders, Holt-Winters'
# initial seasonal indices) to estimate reliably rather than overfit noise.
# A benchmark across this project's own candidate models confirmed exactly
# this: on a 24-observation monthly series, simple statistical baselines
# consistently outperformed the more complex candidates Phase 1 was
# recommending for a "Trend + Seasonal" classification.
#
# _min_recommended_observations() adds that missing dimension as a
# post-filter, without touching classify_series() or _MODEL_MAP's existing
# type -> candidate associations (which already correctly capture trend/
# seasonality/volatility). It is deliberately a SOFTER threshold than each
# model's own hard check_requirements() floor in trainer.py (which only
# asks "can this mathematically fit at all", e.g. SARIMA's minimum of one
# seasonal cycle) — this asks the more conservative question "is fitting
# this model here likely to be a statistically sound idea, or is a simpler
# alternative more trustworthy at this sample size." Thresholds are
# expressed relative to the series' own seasonal cycle length
# (seasonal_period_for_freq) so they scale sensibly across frequencies
# (24 monthly points is two years of history; 24 daily points is three
# weeks) rather than using one fixed number for every cadence.
_DEFAULT_SEASONAL_PERIOD = 12  # fallback for yearly/irregular/unknown frequency

# Trend-only, low-parameter smoothing methods (level + trend, no seasonal
# component) — their data appetite doesn't scale with cycle length, so a
# flat floor is used instead of a period-relative one.
_FLAT_MIN_OBSERVATIONS = {
    "Holt Linear": 8,
    "Croston": 8,
}


def _min_recommended_observations(model_name: str, frequency: str) -> int:
    """
    Minimum observation count below which `model_name` is not recommended
    as a candidate for `frequency` data — see the module-level comment
    above for the rationale. Returns 0 (no floor) for any model not listed
    here, e.g. the always-safe simple baselines (Moving Average, Naive,
    Simple Exponential Smoothing) that this module already only ever
    recommends for the shortest-history series types.
    """
    if model_name in _FLAT_MIN_OBSERVATIONS:
        return _FLAT_MIN_OBSERVATIONS[model_name]

    period = seasonal_period_for_freq(frequency) or _DEFAULT_SEASONAL_PERIOD

    if model_name in ("Holt-Winters", "SARIMA"):
        # Classical seasonal models need to estimate initial seasonal
        # indices (Holt-Winters) or seasonal AR/MA/differencing orders
        # (SARIMA) on top of trend/level — two full cycles is the
        # commonly cited absolute floor for these to fit at all, and
        # noticeably more than two is what practitioners consider
        # reliable rather than a bare-minimum fit; 2.5 cycles is used as
        # the practical cutoff for recommending them by default.
        return int(round(period * 2.5))

    if model_name == "Prophet":
        # Prophet's regularised, Bayesian-changepoint decomposition
        # degrades more gracefully on thin data than a classical ARIMA/
        # exponential-smoothing seasonal fit (it can down-weight a poorly
        # supported seasonal term rather than needing it to be fully
        # estimable), so it is held to a lighter floor than Holt-Winters/
        # SARIMA — enough for one full cycle plus a margin.
        return int(round(period * 1.5))

    if model_name == "XGBoost":
        # Tree-based supervised learning needs enough rows, independent
        # of seasonal cycle length, to learn nonlinear splits without
        # overfitting once lag/rolling features consume some of the
        # earliest observations — 30 is used as an absolute floor
        # (matches this project's own benchmark finding that XGBoost
        # underperforms simple baselines well below this), scaled up
        # further for long-cycle frequencies (e.g. weekly data with a
        # 52-period yearly cycle) where useful seasonal features need
        # even more history to populate.
        return max(30, period + 18)

    return 0


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

    Data-sufficiency filter (forecasting-quality pass): candidates whose
    model_name does not meet _min_recommended_observations() for this
    series' n_observations/frequency are left out of the returned list —
    trainer.py trains every name this function returns (plus its own
    always-included simple baselines), so recommending a model here is
    what causes it to be trained and considered for selection at all.
    Dropping rather than merely reordering a data-inappropriate candidate
    is what actually keeps it from being selected or pulled into an
    ensemble downstream.

    This never leaves a series with zero usable candidates: trainer.py's
    GUARANTEED_CANDIDATES (Naive, Moving Average, Simple Exponential
    Smoothing — all always data-appropriate) are trained independently of
    whatever this function returns, so a very short history simply falls
    through to those simple, low-variance baselines, which is exactly the
    intended behaviour for that case. If every entry for a series_type is
    filtered out, this function returns an empty list rather than forcing
    a data-inappropriate model through, and the guaranteed baselines carry
    the forecast on their own.
    """
    entries = _MODEL_MAP.get(classification.series_type, _MODEL_MAP["Smooth"])
    stats_dict = stats.to_dict()

    recommendations = []
    rank = 0
    for model_name, reason_template in entries:
        try:
            reason = reason_template.format(**stats_dict)
        except (KeyError, IndexError):
            reason = reason_template

        min_obs = _min_recommended_observations(model_name, stats.frequency)
        if stats.n_observations < min_obs:
            continue

        rank += 1
        recommendations.append(
            ModelRecommendation(rank=rank, model=model_name, reason=f"{reason} {_PHASE_NOTE}")
        )
    return recommendations
