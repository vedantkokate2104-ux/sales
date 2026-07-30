"""
classifier.py
-------------
Series Classification — Step 2 of the Forecast Intelligence Engine.

Takes the descriptive statistics produced by forecast.analyzer and assigns
one of a fixed set of series "types", each with a confidence score and a
list of human-readable reasons. This is a rule-based classifier — no ML
model is trained or run here; thresholds are centralised at the top of the
file so they can be tuned without touching the decision logic.
"""

from .types import SeriesStats, SeriesClassification

SHORT_HISTORY_DAYS   = 90     # below this: not enough history for pattern detection
LONG_HISTORY_DAYS     = 730   # 2+ years: enough for complex seasonal models
NEW_PRODUCT_MIN_OBS   = 4     # fewer observations than this: effectively no history
INTERMITTENT_ZERO_PCT = 30.0  # % zero-sales periods that indicates sporadic demand
HIGH_CV               = 1.0   # coefficient of variation above this: highly volatile
TREND_THRESHOLD       = 0.4   # trend_strength at/above this counts as "trending"
SEASONALITY_THRESHOLD = 0.4   # seasonality_strength at/above this counts as "seasonal"


def classify_series(stats: SeriesStats) -> SeriesClassification:
    """
    Classify a series into one of: New Product, Short History, Intermittent
    Demand, Highly Volatile, Trend + Seasonal, Seasonal, Trending,
    Long History, Smooth. Checks run in priority order — the first
    condition that matches wins, since these classes are not mutually
    exclusive in the underlying statistics (e.g. a series can technically
    be both "trending" and "long history").
    """
    reasons = []

    if stats.n_observations < NEW_PRODUCT_MIN_OBS:
        reasons.append(
            f"Only {stats.n_observations} observation(s) available — too little "
            f"history to identify any pattern."
        )
        return SeriesClassification("New Product", 0.9, reasons)

    if stats.history_length_days < SHORT_HISTORY_DAYS:
        reasons.append(
            f"History spans only {stats.history_length_days} day(s), below the "
            f"{SHORT_HISTORY_DAYS}-day threshold used for reliable pattern detection."
        )
        return SeriesClassification("Short History", 0.85, reasons)

    if stats.pct_zero_sales >= INTERMITTENT_ZERO_PCT:
        reasons.append(
            f"{stats.pct_zero_sales}% of periods have zero sales, indicating "
            f"sporadic rather than continuous demand."
        )
        confidence = round(min(0.95, 0.55 + stats.pct_zero_sales / 200), 2)
        return SeriesClassification("Intermittent Demand", confidence, reasons)

    if stats.coefficient_of_variation >= HIGH_CV:
        reasons.append(
            f"Coefficient of variation is {stats.coefficient_of_variation}, "
            f"indicating volatility far above the series mean."
        )
        if stats.outlier_percentage > 0:
            reasons.append(f"{stats.outlier_percentage}% of values are statistical outliers.")
        confidence = round(min(0.95, 0.5 + stats.coefficient_of_variation / 4), 2)
        return SeriesClassification("Highly Volatile", confidence, reasons)

    has_trend = stats.trend_strength >= TREND_THRESHOLD
    has_season = stats.seasonality_strength >= SEASONALITY_THRESHOLD

    if has_trend and has_season:
        reasons.append(f"Trend strength {stats.trend_strength} indicates a consistent upward/downward movement.")
        reasons.append(f"Seasonality strength {stats.seasonality_strength} indicates a recurring cyclical pattern.")
        reasons.append(f"{stats.n_observations} observations spanning {stats.history_length_days} days.")
        confidence = round(min(0.95, (stats.trend_strength + stats.seasonality_strength) / 2 + 0.25), 2)
        return SeriesClassification("Trend + Seasonal", confidence, reasons)

    if has_season:
        reasons.append(
            f"Seasonality strength {stats.seasonality_strength} indicates a recurring "
            f"cyclical pattern with no strong overall trend."
        )
        confidence = round(min(0.95, stats.seasonality_strength + 0.3), 2)
        return SeriesClassification("Seasonal", confidence, reasons)

    if has_trend:
        reasons.append(
            f"Trend strength {stats.trend_strength} indicates a consistent upward/downward "
            f"movement with no strong seasonal cycle."
        )
        confidence = round(min(0.95, stats.trend_strength + 0.3), 2)
        return SeriesClassification("Trending", confidence, reasons)

    if stats.history_length_days >= LONG_HISTORY_DAYS:
        reasons.append(
            f"History spans {stats.history_length_days} days (2+ years) with no strong "
            f"trend or seasonal signal detected."
        )
        return SeriesClassification("Long History", 0.6, reasons)

    reasons.append(
        f"Low variance (coefficient of variation {stats.coefficient_of_variation}) and no "
        f"strong trend or seasonal signal — series is stable and predictable."
    )
    return SeriesClassification("Smooth", 0.7, reasons)
