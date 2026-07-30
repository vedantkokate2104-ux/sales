"""
analyzer.py
-----------
Forecast Analyzer — Step 1 of the Forecast Intelligence Engine.

Computes descriptive statistics for a sales time series: how much data
exists, how it is spaced in time, and how it behaves (trend, seasonality,
variance, zero-sales rate, outliers). These statistics feed the Series
Classifier (classifier.py); they do not forecast anything themselves.

Reuses analytics.get_col() for business-name -> column resolution, exactly
like every other module in this codebase, so schema-mapping behaviour is
identical here to the KPI engine and query engine.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from analytics import get_col, _clean_numeric
from .types import SeriesStats
from .utils import (
    infer_frequency,
    count_missing_timestamps,
    outlier_percentage,
    trend_strength,
    seasonality_strength,
)


def analyze_series(dates: pd.Series, values: pd.Series, group: str = "ALL") -> SeriesStats:
    """
    Compute SeriesStats for one series of (date, revenue) observations.
    `group` is a label identifying which product/segment this series
    represents ("ALL" for the whole dataset aggregated by date).
    """
    dates = pd.to_datetime(dates, errors="coerce")
    valid = dates.notna()
    dates, values = dates[valid], values[valid]

    n_observations = int(len(dates))
    numeric_values = _clean_numeric(values) if n_observations else values

    if n_observations == 0:
        return SeriesStats(
            group=group, n_observations=0, start_date="", end_date="",
            frequency="unknown", missing_timestamps=0, missing_values=0,
            trend_strength=0.0, seasonality_strength=0.0, variance=0.0,
            coefficient_of_variation=0.0, pct_zero_sales=0.0,
            outlier_percentage=0.0, history_length_days=0,
        )

    start_date = str(dates.min().date())
    end_date = str(dates.max().date())
    history_length_days = int((dates.max() - dates.min()).days)

    frequency = infer_frequency(dates) if n_observations >= 2 else "unknown"
    missing_timestamps = count_missing_timestamps(dates, frequency) if n_observations >= 2 else 0
    missing_values = int(values.isna().sum())

    variance = float(numeric_values.var()) if n_observations >= 2 else 0.0
    mean_val = float(numeric_values.mean())
    coefficient_of_variation = round(abs((variance ** 0.5) / mean_val), 4) if mean_val else 0.0
    pct_zero_sales = round(float((numeric_values == 0).mean()) * 100, 2)
    outlier_pct = outlier_percentage(numeric_values)

    # Aggregate to monthly totals for trend/seasonality — mirrors the exact
    # monthly aggregation analytics.run_forecast() already uses, so "trend"
    # and "seasonal" mean the same thing in this module as in the existing
    # forecaster.
    trend, season = 0.0, 0.0
    if n_observations >= 2:
        periods = dates.dt.to_period("M")
        monthly = numeric_values.groupby(periods).sum().sort_index()
        if len(monthly) >= 3:
            trend = trend_strength(monthly.values)
        if len(monthly) >= 24:
            season = seasonality_strength(monthly.values, period=12)

    return SeriesStats(
        group=group,
        n_observations=n_observations,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        missing_timestamps=int(missing_timestamps),
        missing_values=missing_values,
        trend_strength=trend,
        seasonality_strength=season,
        variance=round(variance, 2),
        coefficient_of_variation=coefficient_of_variation,
        pct_zero_sales=pct_zero_sales,
        outlier_percentage=outlier_pct,
        history_length_days=history_length_days,
    )


def analyze_dataset(df: pd.DataFrame, mappings: list, group_col: Optional[str] = None) -> Dict[str, SeriesStats]:
    """
    Analyse the dataset either as one aggregated series ("ALL") or as one
    series per distinct value of `group_col` (e.g. a product identifier
    column), if provided.
    """
    date_col = get_col(df, mappings, "invoice date")
    rev_col = get_col(df, mappings, "net sales")
    if not date_col or not rev_col:
        raise ValueError(
            "Forecast analysis requires both an invoice date and a net sales "
            "column to be resolved by the schema mapper."
        )

    results: Dict[str, SeriesStats] = {}
    if group_col and group_col in df.columns:
        for group_val, group_df in df.groupby(group_col):
            label = str(group_val)
            results[label] = analyze_series(group_df[date_col], group_df[rev_col], group=label)
    else:
        results["ALL"] = analyze_series(df[date_col], df[rev_col], group="ALL")

    return results


def analyze_primary(df: pd.DataFrame, mappings: list) -> SeriesStats:
    """
    Convenience wrapper used by the Forecast page: analyses the same
    overall revenue-over-time series that analytics.run_forecast() already
    forecasts, so the new "Forecast Analysis" panel and the existing
    forecast chart always describe the same underlying series.
    """
    return analyze_dataset(df, mappings, group_col=None)["ALL"]
