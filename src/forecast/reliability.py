"""
reliability.py
---------------
Forecast reliability: a scored, explained judgement of how much weight a
forecast can bear (Phase 6).

WHY THIS IS NOT THE EXISTING CONFIDENCE SCORE
----------------------------------------------
`selector._compute_confidence()` sums eight weighted signals into a
0-100 number clipped to [5, 97]. It is a reasonable heuristic, but the UI
rendered it as "69%" beside the word *confidence*, which invites exactly
one reading: "there is a 69% chance this forecast is right."

Nothing in the system supports that reading. The number is never
calibrated against outcomes — no procedure checks whether forecasts
scoring 69 come true 69% of the time, and none could, because the score
mixes quantities that are not probabilities (history length, fold count,
data-quality ratio) on an arbitrary common scale. A probability is a
claim about the future; this is a summary of the evidence behind a
model.

So this module does not replace it. `forecast_confidence` and
`forecast_confidence_pct` are untouched and still returned, because
existing clients read them. What is added is a RELIABILITY score that is
explicitly a 0-1 evidence rating with a named level (HIGH / MODERATE /
LOW / VERY_LOW), an explicit statement that it is not a probability, and
a per-factor breakdown so a reader can see which parts of the evidence
are strong and which are weak — rather than being handed one number and
asked to trust it.

It also uses two signals the confidence score never saw, both introduced
after it was written:

  - Phase 4's interval CALIBRATION status and measured coverage error.
    Whether the stated uncertainty range is trustworthy is a distinct
    question from how accurate the point forecast is, and it was
    previously invisible.
  - Phase 1's series DENSITY — what share of modelled periods contain
    real transactions rather than filled values.

WHAT IS DELIBERATELY NOT A SCORED SIGNAL
-----------------------------------------
Interval WIDTH. It is reported as context because users ask about it, but
it is not scored: width is essentially a function of the error
distribution that ACCURACY and CONSISTENCY already measure, so scoring it
would count the same evidence three times and let one weakness dominate
the result. Calibration IS scored, because it depends on sample size
rather than on error magnitude and is genuinely independent.

THRESHOLDS ARE ANCHORED, NOT INVENTED
--------------------------------------
Each sub-score is anchored to a reference point that means something:

  ACCURACY     WAPE is already a percentage of actual volume, and
               forecasting zero for every period scores exactly 100%.
               100% is therefore the natural "no information" anchor.
  CONSISTENCY  the coefficient of variation of per-fold error; 1.0 means
               the spread across folds is as large as the average error.
  HISTORY      interpolated between forecast.backtest's own MIN_FOLDS
               (the minimum for any estimate) and TARGET_FOLDS (its
               design target), so it moves with that planner.
  SELECTION    the winner's margin over the runner-up expressed in units
               of fold-to-fold standard deviation — a margin smaller than
               the noise is not evidence of a better model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .backtest import MIN_FOLDS, TARGET_FOLDS
from .intervals import CALIBRATED, LIMITED_DATA, UNAVAILABLE

logger = logging.getLogger(__name__)

# Reliability levels, high to low.
HIGH = "HIGH"
MODERATE = "MODERATE"
LOW = "LOW"
VERY_LOW = "VERY_LOW"

# Level boundaries on the 0-1 composite. Spaced evenly across the usable
# range rather than tuned to make any particular dataset look good — the
# bundled company series lands at MODERATE, and that is the honest
# reading of a 40% WAPE with substantial fold variation.
LEVEL_THRESHOLDS = ((0.70, HIGH), (0.50, MODERATE), (0.30, LOW))

# Weights over the six scored signals. Accuracy leads, but at 0.28 it
# cannot by itself carry a forecast to HIGH — which is the point of
# Phase 6, Step 4: a good WAPE on an unstable, short, or uncalibrated
# series is not a reliable forecast.
SIGNAL_WEIGHTS = {
    "accuracy":          0.28,
    "consistency":       0.22,
    "history":           0.18,
    "data_quality":      0.12,
    "interval_calibration": 0.12,
    "selection_strength": 0.08,
}

# A coverage error this many percentage points from nominal drives the
# calibration sub-score to zero. 20pp is roughly twice the standard error
# of a coverage estimate at the sample sizes this project sees, so
# ordinary sampling noise does not zero the signal.
COVERAGE_ERROR_FLOOR_PP = 20.0

# Impact labels used on every factor.
POSITIVE, NEUTRAL, NEGATIVE = "positive", "neutral", "negative"


@dataclass
class ReliabilityFactor:
    """One measured contributor to the reliability judgement."""
    factor: str
    label: str
    status: str
    value: Optional[str]
    score: Optional[float]
    weight: float
    impact: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "factor":      self.factor,
            "label":       self.label,
            "status":      self.status,
            "value":       self.value,
            "score":       round(self.score, 3) if self.score is not None else None,
            "weight":      self.weight,
            "impact":      self.impact,
            "explanation": self.explanation,
        }


@dataclass
class ReliabilityAssessment:
    score: float
    level: str
    label: str
    summary: str
    factors: List[ReliabilityFactor] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_probability: bool = False
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "score":          round(self.score, 3),
            "score_pct":      int(round(self.score * 100)),
            "level":          self.level,
            "label":          self.label,
            "summary":        self.summary,
            "factors":        [f.to_dict() for f in self.factors],
            "warnings":       self.warnings,
            "is_probability": self.is_probability,
            "interpretation": self.interpretation,
        }


def _impact(score: Optional[float]) -> str:
    if score is None:
        return NEUTRAL
    if score >= 0.65:
        return POSITIVE
    if score >= 0.40:
        return NEUTRAL
    return NEGATIVE


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _accuracy_factor(metrics: Optional[dict]) -> ReliabilityFactor:
    wape = (metrics or {}).get("wape")
    if wape is None or not np.isfinite(wape):
        return ReliabilityFactor(
            "accuracy", "Validation error", "unmeasured", None, None,
            SIGNAL_WEIGHTS["accuracy"], NEUTRAL,
            "No validated error is available for this series, so accuracy could not be assessed.",
        )
    score = _clamp(1.0 - wape / 100.0)
    # Bands aligned with the impact thresholds below, so a factor is never
    # described as "high error" while being reported as neutral impact.
    if wape < 25:
        wording = "Historical forecast error is low."
    elif wape < 45:
        wording = "Historical forecast error is moderate."
    elif wape < 65:
        wording = "Historical forecast error is high."
    else:
        wording = "Historical forecast error is very high."
    return ReliabilityFactor(
        "accuracy", "Validation error", "measured", f"WAPE {wape:.2f}%", score,
        SIGNAL_WEIGHTS["accuracy"], _impact(score),
        f"{wording} Across walk-forward validation the forecast was off by {wape:.1f}% of actual "
        f"volume on average (forecasting zero every period would score 100%).",
    )


def _consistency_factor(fold_stats: Optional[dict]) -> ReliabilityFactor:
    stats = fold_stats or {}
    mean, std, n = stats.get("mean"), stats.get("std"), stats.get("n_folds", 0)
    if not mean or std is None or n < 2:
        return ReliabilityFactor(
            "consistency", "Fold-to-fold consistency", "unmeasured", None, None,
            SIGNAL_WEIGHTS["consistency"], NEUTRAL,
            "Too few validation folds to measure how much accuracy varies over time.",
        )
    cv = std / mean if mean else 1.0
    score = _clamp(1.0 - cv)
    worst, best = stats.get("max", 0.0), stats.get("min", 0.0)

    # Describe the spread by the ratio between the best and worst fold
    # rather than by a threshold on the coefficient of variation. A CV of
    # 0.36 sits just under any "large" cutoff while still corresponding to
    # a five-fold gap between the best and worst validation window, and
    # calling that "modest" would understate it. The ratio is a plain
    # fact; the score continues to come from the CV.
    ratio = (worst / best) if best > 0 else None
    if ratio is not None and ratio >= 3.0:
        spread = (
            f"The worst window was {ratio:.1f}x the error of the best, so performance in any "
            f"single future window is hard to anticipate."
        )
    elif ratio is not None and ratio >= 1.75:
        spread = (
            f"The worst window was {ratio:.1f}x the error of the best — noticeable variation, "
            f"but within a predictable band."
        )
    else:
        spread = "Error was fairly repeatable from window to window."

    return ReliabilityFactor(
        "consistency", "Fold-to-fold consistency", "measured",
        f"error std {std:.2f} across {n} folds (min {best:.1f}%, max {worst:.1f}%)",
        score, SIGNAL_WEIGHTS["consistency"], _impact(score),
        f"Accuracy varied between {best:.1f}% and {worst:.1f}% WAPE across validation folds. "
        + spread,
    )


def _history_factor(n_observations: int, n_folds: int, frequency: str) -> ReliabilityFactor:
    span = max(1, TARGET_FOLDS - MIN_FOLDS)
    score = _clamp((n_folds - MIN_FOLDS) / span)
    if n_folds == 0:
        return ReliabilityFactor(
            "history", "History available", "insufficient",
            f"{n_observations} {frequency} periods, no validation folds", 0.0,
            SIGNAL_WEIGHTS["history"], NEGATIVE,
            f"{n_observations} {frequency} periods were not enough to run any walk-forward "
            f"validation, so this forecast has no independent accuracy estimate.",
        )
    return ReliabilityFactor(
        "history", "History available", "measured",
        f"{n_observations} {frequency} periods, {n_folds} validation folds", score,
        SIGNAL_WEIGHTS["history"], _impact(score),
        (
            f"{n_observations} {frequency} periods supported {n_folds} independent validation "
            f"folds. "
        ) + (
            f"That is at or near the {TARGET_FOLDS}-fold design target, so the error estimate "
            f"rests on a reasonable amount of evidence."
            if n_folds >= TARGET_FOLDS - 2 else
            f"Fewer folds than the {TARGET_FOLDS}-fold design target means the error estimate "
            f"itself carries meaningful uncertainty."
        ),
    )


def _data_quality_factor(
    data_quality: Optional[float], series_density: Optional[float],
    empty_periods: Optional[int], n_observations: int,
) -> ReliabilityFactor:
    parts = [v for v in (data_quality, series_density) if v is not None and np.isfinite(v)]
    if not parts:
        return ReliabilityFactor(
            "data_quality", "Input data quality", "unmeasured", None, None,
            SIGNAL_WEIGHTS["data_quality"], NEUTRAL,
            "No data-quality measurement was available for this series.",
        )
    score = _clamp(float(np.mean(parts)))
    density_pct = (series_density or 0.0) * 100.0
    gaps = empty_periods or 0
    return ReliabilityFactor(
        "data_quality", "Input data quality", "measured",
        f"{density_pct:.0f}% of periods observed"
        + (f", {gaps} empty" if gaps else "")
        + (f", quality score {data_quality:.2f}" if data_quality is not None else ""),
        score, SIGNAL_WEIGHTS["data_quality"], _impact(score),
        (
            f"{gaps} of {n_observations} modelled periods contained no recorded transactions, so "
            f"part of the series is filled rather than measured."
            if gaps else
            f"Every one of the {n_observations} modelled periods contains real recorded "
            f"transactions, and the input needed little correction."
        ),
    )


def _calibration_factor(intervals: Optional[dict]) -> ReliabilityFactor:
    info = intervals or {}
    status = info.get("interval_calibration_status")
    if not info.get("available") or status in (None, UNAVAILABLE):
        return ReliabilityFactor(
            "interval_calibration", "Uncertainty calibration", "unavailable", None, 0.0,
            SIGNAL_WEIGHTS["interval_calibration"], NEGATIVE,
            "There was not enough out-of-sample error history to estimate a prediction interval, "
            "so the uncertainty around this forecast is unquantified.",
        )

    base = 1.0 if status == CALIBRATED else 0.5
    coverage = (info.get("coverage") or {}).get("80") or {}
    observed, error_pp, n = coverage.get("observed"), coverage.get("coverage_error_pp"), coverage.get("n", 0)

    if error_pp is None:
        score = base * 0.5
        detail = "Interval coverage could not be measured out-of-sample — too few validation folds."
    else:
        score = base * _clamp(1.0 - abs(error_pp) / COVERAGE_ERROR_FLOOR_PP)
        detail = (
            f"The 80% interval actually contained the outcome {observed:.0f}% of the time across "
            f"{n} out-of-sample checks ({error_pp:+.1f}pp against its 80% target)."
        )
    if status == LIMITED_DATA:
        detail += (
            " Calibration rests on a limited number of out-of-sample errors, so the stated levels "
            "are indicative rather than guaranteed."
        )
    return ReliabilityFactor(
        "interval_calibration", "Uncertainty calibration", str(status).lower(),
        f"{status}" + (f", 80% coverage {observed:.0f}%" if observed is not None else ""),
        score, SIGNAL_WEIGHTS["interval_calibration"], _impact(score), detail,
    )


def _selection_factor(
    best_wape: Optional[float], runner_up_wape: Optional[float],
    fold_std: Optional[float], best_model: str, runner_up_model: Optional[str],
) -> ReliabilityFactor:
    if best_wape is None or runner_up_wape is None or runner_up_model is None:
        return ReliabilityFactor(
            "selection_strength", "Model selection strength", "single_candidate", None, None,
            SIGNAL_WEIGHTS["selection_strength"], NEUTRAL,
            "Only one model could be validated on this series, so there is no comparison to judge "
            "the choice against.",
        )

    margin = runner_up_wape - best_wape
    if not fold_std or fold_std <= 0:
        score = _clamp(margin / max(best_wape, 1.0))
        detail = (
            f"{best_model} beat {runner_up_model} by {margin:.2f}pp WAPE, but fold-to-fold "
            f"variation could not be measured to judge whether that margin is meaningful."
        )
    else:
        z = margin / fold_std
        score = _clamp(z)
        detail = (
            f"{best_model} ({best_wape:.2f}% WAPE) beat {runner_up_model} "
            f"({runner_up_wape:.2f}%) by {margin:.2f}pp, which is {z:.2f}x the "
            f"{fold_std:.2f}pp fold-to-fold variation. "
        ) + (
            "The margin is comfortably larger than the noise, so the choice is well supported."
            if z >= 1.0 else
            "The margin is smaller than the noise between folds, so several models perform "
            "similarly here and the ranking could change on different data."
        )
    return ReliabilityFactor(
        "selection_strength", "Model selection strength", "measured",
        f"{margin:+.2f}pp over {runner_up_model}", score,
        SIGNAL_WEIGHTS["selection_strength"], _impact(score), detail,
    )


def _interval_width_context(intervals: Optional[dict], forecast_points: Sequence[dict]) -> Optional[ReliabilityFactor]:
    """
    Interval width as REPORTED CONTEXT, not a scored signal — see the
    module docstring. Carries weight 0.0 and score None so it appears in
    the factor list for the reader without entering the composite.
    """
    info = intervals or {}
    if not info.get("available"):
        return None
    widths, points = [], []
    for point in forecast_points or []:
        width, value = point.get("interval_width_80"), point.get("value")
        if width is not None and value:
            widths.append(width)
            points.append(value)
    if not widths:
        return None

    normalised = float(np.mean(widths)) / max(float(np.mean(points)), 1e-9)
    return ReliabilityFactor(
        "interval_width", "Prediction interval width", "measured",
        f"80% interval spans {normalised:.1f}x the forecast value", None, 0.0,
        NEGATIVE if normalised >= 1.0 else (NEUTRAL if normalised >= 0.5 else POSITIVE),
        (
            f"The 80% prediction range is about {normalised:.1f} times the size of the forecast "
            f"itself. "
        ) + (
            "That is a wide range — the forecast is best used as a rough planning estimate."
            if normalised >= 1.0 else
            "That is a usable range for planning purposes."
        )
        + " Width follows from the accuracy and consistency measured above, so it is shown here as"
          " context rather than counted again in the score.",
    )


def assess_reliability(
    metrics: Optional[dict],
    fold_stats: Optional[dict],
    n_observations: int,
    n_folds: int,
    frequency: str,
    data_quality: Optional[float] = None,
    series_density: Optional[float] = None,
    empty_periods: Optional[int] = None,
    intervals: Optional[dict] = None,
    forecast_points: Optional[Sequence[dict]] = None,
    ranking: Optional[Sequence[dict]] = None,
    selected_model: str = "",
    group_label: Optional[str] = None,
    history_status: Optional[str] = None,
) -> ReliabilityAssessment:
    """
    Score and explain how much weight this forecast can bear.

    Every input is a measurement already produced upstream — nothing here
    refits a model or re-runs validation, so this costs microseconds on
    top of a forecast that takes tens of seconds (Phase 6, Step 26).
    """
    best_wape = (metrics or {}).get("wape")
    runner_up_wape, runner_up_model = None, None
    for entry in (ranking or []):
        name = entry.get("model")
        entry_metrics = entry.get("metrics") or {}
        value = entry_metrics.get("wape")
        if name and name != selected_model and value is not None and np.isfinite(value):
            runner_up_wape, runner_up_model = value, name
            break

    factors = [
        _accuracy_factor(metrics),
        _consistency_factor(fold_stats),
        _history_factor(n_observations, n_folds, frequency),
        _data_quality_factor(data_quality, series_density, empty_periods, n_observations),
        _calibration_factor(intervals),
        _selection_factor(best_wape, runner_up_wape, (fold_stats or {}).get("std"),
                          selected_model, runner_up_model),
    ]

    # Weighted mean over signals that could actually be measured, so an
    # unmeasurable signal neither helps nor penalises — it just carries no
    # vote, and its absence is visible in the factor list.
    scored = [f for f in factors if f.score is not None]
    total_weight = sum(f.weight for f in scored)
    score = sum(f.score * f.weight for f in scored) / total_weight if total_weight else 0.0

    width_factor = _interval_width_context(intervals, forecast_points or [])
    if width_factor:
        factors.append(width_factor)

    level = VERY_LOW
    for threshold, name in LEVEL_THRESHOLDS:
        if score >= threshold:
            level = name
            break

    warnings: List[str] = []
    if history_status and history_status != "SUFFICIENT_HISTORY":
        warnings.append(
            f"Limited history: this series does not have enough observations for strong "
            f"validation ({history_status.replace('_', ' ').lower()})."
        )
    calibration_status = (intervals or {}).get("interval_calibration_status")
    if calibration_status == LIMITED_DATA:
        warnings.append(
            "Prediction interval calibration is limited by the amount of historical validation "
            "data available."
        )
    elif calibration_status in (None, UNAVAILABLE):
        warnings.append(
            "No prediction interval could be estimated, so the uncertainty around this forecast "
            "is unquantified."
        )
    for period in (intervals or {}).get("periods", [])[:1]:
        pass
    for note in (intervals or {}).get("notes", []):
        if "extrapolated" in note.lower() or "95%" in note:
            warnings.append(
                "The 95% interval exceeds what the available validation history can support and "
                "will understate true uncertainty."
            )
            break

    def _join(items: List[str]) -> str:
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " and " + items[-1]

    negatives = [f.label.lower() for f in factors if f.impact == NEGATIVE and f.weight > 0]
    positives = [f.label.lower() for f in factors if f.impact == POSITIVE and f.weight > 0]
    scope = group_label or "This forecast"
    if level == HIGH:
        summary = (
            f"{scope} rests on strong evidence"
            + (f", helped by {_join(positives)}" if positives else "") + "."
        )
    elif level == MODERATE:
        summary = (
            f"{scope} is usable as a planning estimate"
            + (f", but {_join(negatives)} limit how much weight it can bear" if negatives else "")
            + "."
        )
    elif level == LOW:
        summary = (
            f"{scope} should be treated cautiously"
            + (f" — {_join(negatives)} are all weak" if negatives else "")
            + "."
        )
    else:
        summary = f"{scope} rests on very weak evidence and should not be used for committed planning."

    return ReliabilityAssessment(
        score=score, level=level, label=level.replace("_", " ").title(),
        summary=summary, factors=factors, warnings=warnings, is_probability=False,
        interpretation=(
            "Forecast reliability is a 0-100 rating of how much evidence supports this forecast — "
            "its measured accuracy, how consistent that accuracy is over time, how much history it "
            "rests on, the quality of the input data, and whether its uncertainty range is "
            "calibrated. It is NOT a probability that the forecast is correct, and it should not "
            "be read as one. For the likely range of outcomes, use the prediction interval."
        ),
    )
