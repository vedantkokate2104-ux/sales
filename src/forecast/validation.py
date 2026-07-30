"""
validation.py
--------------
Dataset Validation (Phase E, Task 2): checks run on the raw (date,
revenue) pairs BEFORE training starts, so problems with the data itself
are reported as clear, business-friendly messages up front instead of
surfacing later as a confusing model-training failure or a silently poor
forecast.

This does not duplicate forecast.analyzer's descriptive statistics
(Phase 1, unmodified) — it reuses the already-computed SeriesStats where
possible (frequency, missing_timestamps, history_length_days) and adds a
small set of pass/fail data-quality checks specific to "is this dataset
trainable, and how much should its forecast be trusted." Every check
degrades to a warning string; nothing here raises, so a marginal dataset
still gets the best forecast the pipeline can produce, with the caveats
disclosed plainly.

forecast_engine.py merges the messages returned here directly into the
existing `warnings` list on AutoForecastResult — no new API field, so the
Forecast page's existing warning-rendering code needs zero changes.
"""

import logging
from typing import List, Optional

import pandas as pd

from .types import SeriesStats

logger = logging.getLogger(__name__)

# Below this many valid dated observations, no model can be meaningfully
# trained at all (trainer.py's own hard floor for a series to be usable).
MIN_OBSERVATIONS_TO_TRAIN = 4

# Below this many valid observations (but at/above the hard floor above),
# training proceeds but confidence should be treated as low.
SHORT_HISTORY_OBSERVATIONS = 8

# If more than this fraction of the expected periods (at the detected
# frequency) between the first and last observation have no data at all,
# the zero-filled gaps are large enough to meaningfully distort trend and
# seasonality estimates and the user should know about it.
MAX_MISSING_TIMESTAMP_RATIO = 0.3

# Below this many days of total calendar history, treat any forecast as
# provisional regardless of how many raw rows exist.
SHORT_HISTORY_DAYS = 30


def validate_dataset(dates: pd.Series, values: pd.Series, stats: Optional[SeriesStats] = None) -> List[str]:
    """
    Run every dataset-quality check and return a list of business-friendly
    warning strings (empty list = no issues found). Never raises.

    Args:
        dates  — the raw (unaggregated) date column for the series about
                 to be trained.
        values — the raw (unaggregated) revenue column, same rows as dates.
        stats  — Phase 1's already-computed SeriesStats for this series, if
                 available. When supplied, a few checks (missing-timestamp
                 ratio, frequency regularity, calendar history span) reuse
                 its numbers instead of recomputing them, so this module
                 never disagrees with the Dataset Analysis panel.
    """
    messages: List[str] = []

    parsed_dates = pd.to_datetime(dates, errors="coerce")
    valid_mask = parsed_dates.notna()
    n_valid = int(valid_mask.sum())

    if len(parsed_dates) == 0 or n_valid == 0:
        messages.append("No usable date values were found in this dataset — forecasting cannot proceed.")
        return messages

    # ── Missing values ────────────────────────────────────────────────────
    numeric_values = pd.to_numeric(values, errors="coerce")
    n_missing_values = int(numeric_values.isna().sum())
    if n_missing_values > 0:
        pct = round(n_missing_values / len(values) * 100, 1)
        messages.append(
            f"{n_missing_values} row(s) ({pct}%) are missing a sales value and were treated as zero."
        )

    # ── Duplicate timestamps ─────────────────────────────────────────────
    dup_count = int(parsed_dates[valid_mask].duplicated().sum())
    if dup_count > 0:
        plural = "y" if dup_count == 1 else "ies"
        messages.append(
            f"{dup_count} duplicate date entr{plural} found — sales on the same date were summed together."
        )

    # ── Minimum observations / short history (by row count) ────────────────
    if n_valid < MIN_OBSERVATIONS_TO_TRAIN:
        messages.append(
            f"Only {n_valid} valid date(s) found — at least {MIN_OBSERVATIONS_TO_TRAIN} are needed to "
            f"train any forecasting model. Results will be skipped or highly unreliable."
        )
    elif n_valid < SHORT_HISTORY_OBSERVATIONS:
        messages.append(
            f"Only {n_valid} data points are available — forecasts will have low confidence until "
            f"more history is collected."
        )

    # ── Constant series ──────────────────────────────────────────────────
    filled_values = numeric_values.fillna(0)
    if len(filled_values) > 1 and filled_values.nunique() == 1:
        messages.append(
            "Every value in this dataset is identical — there is no variation for a model to learn from."
        )

    if stats is not None:
        # ── Missing timestamps (reuses Phase 1's already-computed count) ──
        if stats.n_observations > 0:
            expected = stats.n_observations + stats.missing_timestamps
            if expected > 0 and (stats.missing_timestamps / expected) > MAX_MISSING_TIMESTAMP_RATIO:
                pct = round(stats.missing_timestamps / expected * 100, 1)
                messages.append(
                    f"{pct}% of expected {stats.frequency} periods have no data — the gaps were "
                    f"filled with zero, which may distort the trend and seasonality estimates."
                )

        # ── Irregular / unrecognised frequency ──────────────────────────
        if stats.frequency in ("irregular", "unknown"):
            messages.append(
                "Dates in this dataset don't follow a regular daily/weekly/monthly/quarterly/yearly "
                "pattern — a monthly grain was used as a fallback for training."
            )

        # ── Short history in calendar time (independent of row count) ──
        if 0 < stats.history_length_days < SHORT_HISTORY_DAYS:
            messages.append(
                f"This dataset only spans {stats.history_length_days} day(s) of history — treat "
                f"any forecast from it as provisional."
            )

    return messages
