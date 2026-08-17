"""
analyzer.py
-----------
Forecast Analyzer — Step 1 of the Forecast Intelligence Engine.

Computes descriptive statistics for a sales time series: how much data
exists, how it is spaced in time, and how it behaves (trend, seasonality,
variance, zero-sales rate, outliers). These statistics feed the Series
Classifier (classifier.py); they do not forecast anything themselves.

Phase 1, Step 7 — statistics now describe the MODELLED FORECASTING
SERIES, not raw transaction rows.
-----------------------------------------------------------------------
This module used to call `analyze_series(df[date_col], df[rev_col])` on
individual ERP rows. That made:

    n_observations           = number of invoices          (180)
    coefficient_of_variation = invoice-to-invoice spread   (1.44)
    pct_zero_sales           = share of zero-value invoices (0.0%)

None of which describes the series a model is actually asked to predict.
On the bundled dataset the effect was concrete: invoice-level volatility
classified the series "Highly Volatile", which routed candidate selection
toward XGBoost/Prophet/SARIMA, while the aggregated monthly series it
would actually be trained on has a coefficient of variation of ~0.59 —
an ordinary, well-behaved business series.

Statistics are now computed on the aggregated series at the SELECTED
forecast grain (see forecast/grain.py), so `n_observations` means
"modelled periods", `pct_zero_sales` means "periods totalling zero", and
seasonality is measured at a cycle appropriate to the actual frequency
(Step 8) rather than always assuming monthly/period=12.

The raw transaction count is preserved as `SeriesStats.n_transactions`,
and `analyze_series()` (raw-row analysis) is kept for backward
compatibility with any caller that genuinely wants it.

Reuses analytics.get_col() for business-name -> column resolution, exactly
like every other module in this codebase, so schema-mapping behaviour is
identical here to the KPI engine and query engine.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import get_col, _clean_numeric
from .types import SeriesStats
from .grain import GrainSelection, SeriesConstruction, build_forecast_series, select_forecast_grain
from .utils import (
    detect_seasonality,
    infer_frequency,
    count_missing_timestamps,
    outlier_percentage,
    trend_strength,
    seasonality_strength,
)


def analyze_forecast_series(
    construction: SeriesConstruction,
    group: str = "ALL",
    n_transactions: int = 0,
    missing_values: int = 0,
) -> SeriesStats:
    """
    Compute SeriesStats for an already-built forecasting series (Phase 1,
    Step 7). Every statistic describes the modelled PERIOD series carried
    by `construction`, never the underlying transaction rows.

    `construction` comes from forecast.grain.build_forecast_series(), so it
    also carries the observed/missing distinction — which is why
    `empty_periods` (no transaction recorded) and `pct_zero_sales`
    (period total genuinely zero) are reported as the two separate facts
    they are, instead of being collapsed into one number by blanket
    zero-filling.
    """
    series = construction.series
    n_periods = construction.n_periods
    frequency = construction.frequency

    if n_periods == 0:
        return SeriesStats(
            group=group, n_observations=0, start_date="", end_date="",
            frequency=frequency, missing_timestamps=0, missing_values=missing_values,
            trend_strength=0.0, seasonality_strength=0.0, variance=0.0,
            coefficient_of_variation=0.0, pct_zero_sales=0.0,
            outlier_percentage=0.0, history_length_days=0,
            n_transactions=n_transactions, observed_periods=0, empty_periods=0,
            series_density=0.0, zero_period_ratio=0.0, seasonal_period=None,
            fill_strategy=construction.fill_strategy,
        )

    values = series.astype(float)
    arr = values.to_numpy()
    ts_index = series.index.to_timestamp()

    variance = float(values.var()) if n_periods >= 2 else 0.0
    mean_val = float(values.mean())
    cv = round(abs((variance ** 0.5) / mean_val), 4) if mean_val else 0.0

    # Zero PERIODS in the modelled series — the quantity the classifier's
    # intermittent-demand rule actually needs. Distinct from
    # `empty_periods` below, which counts periods that were never observed.
    zero_periods = int((arr == 0.0).sum())
    pct_zero = round(zero_periods / n_periods * 100, 2)

    trend = trend_strength(arr) if n_periods >= 3 else 0.0
    # Step 8: seasonality is measured at a cycle appropriate to THIS
    # frequency, and only when the series covers enough full cycles.
    season, seasonal_period = detect_seasonality(arr, frequency)

    return SeriesStats(
        group=group,
        n_observations=n_periods,
        start_date=str(ts_index[0].date()),
        end_date=str(ts_index[-1].date()),
        frequency=frequency,
        missing_timestamps=construction.n_missing,
        missing_values=missing_values,
        trend_strength=trend,
        seasonality_strength=season,
        variance=round(variance, 2),
        coefficient_of_variation=cv,
        pct_zero_sales=pct_zero,
        outlier_percentage=outlier_percentage(values),
        history_length_days=int((ts_index[-1] - ts_index[0]).days),
        n_transactions=n_transactions,
        observed_periods=construction.n_observed,
        empty_periods=construction.n_missing,
        series_density=construction.density,
        zero_period_ratio=round(zero_periods / n_periods, 4),
        seasonal_period=seasonal_period,
        fill_strategy=construction.fill_strategy,
    )


def prepare_forecast_series(
    dates: pd.Series,
    values: pd.Series,
    horizon: int = 3,
    group: str = "ALL",
    frequency: Optional[str] = None,
) -> Tuple[SeriesConstruction, GrainSelection, SeriesStats]:
    """
    The single entry point that turns raw (date, revenue) transaction rows
    into everything the forecasting pipeline needs about the series:

        (construction, grain_selection, stats)

    This is where transaction grain becomes forecast grain (Phase 1,
    Step 3). Pass `frequency` explicitly only to bypass grain selection
    (used by tests and by callers that already know the grain); leaving it
    None runs the full candidate evaluation in forecast.grain.
    """
    if frequency is None:
        selection = select_forecast_grain(dates, values, horizon=horizon)
        frequency = selection.selected_frequency
    else:
        selection = GrainSelection(
            selected_frequency=frequency,
            candidate_frequencies={},
            reason=f"Frequency '{frequency}' was supplied explicitly by the caller; "
                   f"candidate grain evaluation was skipped.",
            selection_basis="caller-supplied",
        )

    construction = build_forecast_series(dates, values, frequency)
    n_transactions = int(pd.to_datetime(dates, errors="coerce").notna().sum())
    missing_values = int(pd.to_numeric(values, errors="coerce").isna().sum())
    stats = analyze_forecast_series(
        construction, group=group, n_transactions=n_transactions, missing_values=missing_values,
    )
    return construction, selection, stats


def analyze_series(dates: pd.Series, values: pd.Series, group: str = "ALL") -> SeriesStats:
    """
    Raw-transaction-row statistics — RETAINED FOR BACKWARD COMPATIBILITY
    ONLY. Do not use this to make forecasting decisions.

    Every number this returns describes individual ERP rows, not the
    series a model is trained on: `n_observations` is an invoice count and
    `coefficient_of_variation` is invoice-to-invoice spread. See the module
    docstring for why routing forecasting decisions through these numbers
    was wrong. Use `prepare_forecast_series()` / `analyze_forecast_series()`
    instead.
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
        n_transactions=n_observations,
    )


def _resolve_columns(df: pd.DataFrame, mappings: list) -> Tuple[str, str]:
    date_col = get_col(df, mappings, "invoice date")
    rev_col = get_col(df, mappings, "net sales")
    if not date_col or not rev_col:
        raise ValueError(
            "Forecast analysis requires both an invoice date and a net sales "
            "column to be resolved by the schema mapper."
        )
    return date_col, rev_col


def analyze_dataset(
    df: pd.DataFrame,
    mappings: list,
    group_col: Optional[str] = None,
    horizon: int = 3,
) -> Dict[str, SeriesStats]:
    """
    Analyse the dataset either as one aggregated series ("ALL") or as one
    series per distinct value of `group_col` (e.g. a product identifier
    column), if provided.

    Statistics describe the modelled forecasting series at the selected
    grain (Phase 1, Step 7). `horizon` is accepted so grain selection can
    apply its horizon-compatibility rule; it defaults to the same value
    forecast_engine.run_auto_forecast() defaults to, so existing callers
    that omit it are unaffected.
    """
    date_col, rev_col = _resolve_columns(df, mappings)

    results: Dict[str, SeriesStats] = {}
    if group_col and group_col in df.columns:
        for group_val, group_df in df.groupby(group_col):
            label = str(group_val)
            _, _, stats = prepare_forecast_series(
                group_df[date_col], group_df[rev_col], horizon=horizon, group=label,
            )
            results[label] = stats
    else:
        _, _, stats = prepare_forecast_series(
            df[date_col], df[rev_col], horizon=horizon, group="ALL",
        )
        results["ALL"] = stats

    return results


def analyze_primary(df: pd.DataFrame, mappings: list, horizon: int = 3) -> SeriesStats:
    """
    Convenience wrapper used by the Forecast page: analyses the same
    overall revenue-over-time series the Auto Forecasting Engine trains
    on, at the same selected grain, so the "Forecast Analysis" panel and
    the forecast itself always describe the same underlying series.
    """
    return analyze_dataset(df, mappings, group_col=None, horizon=horizon)["ALL"]


def analyze_primary_with_grain(
    df: pd.DataFrame, mappings: list, horizon: int = 3
) -> Tuple[SeriesConstruction, GrainSelection, SeriesStats]:
    """Same as analyze_primary() but also returns the series construction
    and the full grain-selection record, for callers (forecast_engine, the
    /api/forecast/analysis endpoint) that want to report which grains were
    considered and why one was chosen."""
    date_col, rev_col = _resolve_columns(df, mappings)
    return prepare_forecast_series(df[date_col], df[rev_col], horizon=horizon, group="ALL")
