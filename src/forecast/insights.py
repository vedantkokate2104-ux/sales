"""
insights.py
------------
Model-selection explanation and deterministic business insights
(Phase 6, Steps 6-14 and 20).

EVERY STATEMENT HERE IS DERIVED FROM A MEASUREMENT
---------------------------------------------------
Nothing in this module generates free-form narrative. Each insight is a
template selected by a measured value, and each carries the `basis` — the
statistic that produced it — so a reader can check the claim against the
number behind it.

Two rules are enforced by construction:

  NO CAUSAL CLAIMS. The dataset records what was invoiced, not why. So
  this module says "revenue rose over the observed period" and never
  "revenue rose because demand recovered". It has no access to market
  conditions, customer behaviour, competitor activity or seasonality
  causes, and it does not speculate about them. Every verb is
  descriptive.

  NO SCOPE BLEED. Everything is computed from the series actually being
  forecast. A region's explanation uses that region's own error, trend
  and seasonality; the company's numbers never appear in it. This matters
  because Phase 5 made segment forecasts routine, and a company WAPE
  quoted as the reason for a region's reliability would be both wrong and
  convincing.

TREND IS NOT FIRST-VERSUS-LAST
-------------------------------
`classify_trend()` uses the ordinary-least-squares slope over the whole
series together with Phase 1's `trend_strength`, so a series that
finishes high after wandering is not reported as rising. When the
measured strength is weak, the honest answer — "no strong trend" — is
returned regardless of which way the slope happens to point.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Trend ──────────────────────────────────────────────────────────────────
RISING, FALLING, STABLE, UNCLEAR = "RISING", "FALLING", "STABLE", "UNCLEAR"

# Below this Phase 1 trend strength there is no directional signal worth
# reporting, whichever way the slope points.
TREND_STRENGTH_WEAK = 0.20
# Relative slope (per-period change as a share of the series mean) below
# which a direction is not practically meaningful even if detectable.
TREND_SLOPE_NEGLIGIBLE = 0.005

# ── Seasonality bands, on Phase 1's seasonality_strength (0-1) ─────────────
SEASONAL_STRONG, SEASONAL_MODERATE, SEASONAL_WEAK = 0.60, 0.35, 0.15

# ── Volatility bands, on the series coefficient of variation ──────────────
VOLATILITY_HIGH, VOLATILITY_MODERATE = 0.75, 0.40

# ── Model-selection strength ──────────────────────────────────────────────
STRONG, MODERATE_STRENGTH, WEAK = "STRONG", "MODERATE", "WEAK"
# Margin over the runner-up, in units of fold-to-fold standard deviation.
SELECTION_STRONG_Z, SELECTION_MODERATE_Z = 1.0, 0.5


@dataclass
class Insight:
    """One data-derived statement, with the measurement behind it."""
    kind: str
    headline: str
    detail: str
    basis: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "headline": self.headline,
                "detail": self.detail, "basis": self.basis}


@dataclass
class ModelExplanation:
    selected_model: str
    reason: str
    selection_strength: str
    strength_reason: str
    alternatives: List[dict] = field(default_factory=list)
    scope: str = "Company Total"

    def to_dict(self) -> dict:
        return {
            "selected_model":     self.selected_model,
            "reason":             self.reason,
            "selection_strength": self.selection_strength,
            "strength_reason":    self.strength_reason,
            "alternatives":       self.alternatives,
            "scope":              self.scope,
        }


# ── Model selection (Steps 6 & 7) ──────────────────────────────────────────

def explain_model_selection(
    selected_model: str,
    metrics: Optional[dict],
    ranking: Optional[Sequence[dict]],
    fold_stats: Optional[dict],
    n_folds: int,
    scope: str = "Company Total",
) -> ModelExplanation:
    """
    Why this model — and how firmly.

    `ranking` is the validated candidates for THIS series, so a segment's
    explanation lists that segment's own alternatives. Strength is judged
    by comparing the winner's margin against the fold-to-fold standard
    deviation, because a margin smaller than the noise between validation
    windows is not evidence that one model generalises better.
    """
    alternatives = []
    for entry in (ranking or []):
        entry_metrics = entry.get("metrics") or {}
        wape = entry_metrics.get("wape")
        if wape is None or not np.isfinite(wape):
            continue
        alternatives.append({
            "model": entry.get("model"),
            "wape": round(float(wape), 2),
            "folds": entry.get("validation_folds", 0),
            "selected": entry.get("model") == selected_model,
        })
    alternatives.sort(key=lambda a: a["wape"])

    best_wape = (metrics or {}).get("wape")
    others = [a for a in alternatives if not a["selected"]]

    if best_wape is None:
        return ModelExplanation(
            selected_model=selected_model,
            reason=f"{selected_model} was fit on {scope}, but no validated error is available "
                   f"to compare it against alternatives.",
            selection_strength=WEAK,
            strength_reason="No validation metrics were produced for this series.",
            alternatives=alternatives, scope=scope,
        )

    reason = (
        f"{selected_model} achieved the lowest validated error for {scope} "
        f"({best_wape:.2f}% WAPE across {n_folds} leakage-safe walk-forward folds)."
    )

    if not others:
        return ModelExplanation(
            selected_model=selected_model, reason=reason,
            selection_strength=WEAK,
            strength_reason="No alternative model could be validated on this series, so the "
                            "choice reflects availability rather than a comparison.",
            alternatives=alternatives, scope=scope,
        )

    runner_up = others[0]
    margin = runner_up["wape"] - best_wape
    fold_std = (fold_stats or {}).get("std")

    if not fold_std or fold_std <= 0:
        strength = MODERATE_STRENGTH if margin > 0 else WEAK
        strength_reason = (
            f"{selected_model} beat {runner_up['model']} by {margin:.2f}pp, but fold-to-fold "
            f"variation could not be measured, so the margin cannot be judged against noise."
        )
    else:
        z = margin / fold_std
        if z >= SELECTION_STRONG_Z:
            strength = STRONG
            strength_reason = (
                f"{selected_model} beat {runner_up['model']} by {margin:.2f}pp WAPE — "
                f"{z:.2f}x the {fold_std:.2f}pp variation between folds — so the advantage is "
                f"clearly larger than validation noise."
            )
        elif z >= SELECTION_MODERATE_Z:
            strength = MODERATE_STRENGTH
            strength_reason = (
                f"{selected_model} ranked first, beating {runner_up['model']} by {margin:.2f}pp, "
                f"but that margin is only {z:.2f}x the {fold_std:.2f}pp fold-to-fold variation — "
                f"a real but not decisive advantage."
            )
        else:
            strength = WEAK
            strength_reason = (
                f"{selected_model} ranked first, but its {margin:.2f}pp advantage over "
                f"{runner_up['model']} is smaller than the {fold_std:.2f}pp variation between "
                f"validation folds. Several models performed similarly here, and the ranking "
                f"could change on different data."
            )

    return ModelExplanation(
        selected_model=selected_model, reason=reason,
        selection_strength=strength, strength_reason=strength_reason,
        alternatives=alternatives, scope=scope,
    )


# ── Series characteristics (Steps 12-14) ───────────────────────────────────

def classify_trend(
    series: Optional[pd.Series], trend_strength: Optional[float]
) -> Dict[str, object]:
    """
    Direction of the series over its whole span, using the OLS slope
    rather than the first and last observations — a single unusual month
    at either end should not decide the answer.

    When Phase 1's `trend_strength` is weak, no direction is reported even
    if the slope has a sign, because at that strength the sign is not
    distinguishable from noise.
    """
    strength = float(trend_strength) if trend_strength is not None else 0.0

    if series is None or len(series) < 3:
        return {
            "direction": UNCLEAR, "strength": round(strength, 4),
            "slope_per_period": None, "relative_slope": None,
            "explanation": "Too few periods to assess a trend.",
        }

    values = np.asarray(series, dtype=float)
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    mean = float(np.mean(values))
    relative = slope / mean if mean else 0.0

    if strength < TREND_STRENGTH_WEAK or abs(relative) < TREND_SLOPE_NEGLIGIBLE:
        direction = STABLE
        explanation = (
            f"No strong trend was detected (trend strength {strength:.2f}). Revenue moved around "
            f"a broadly flat level over the observed periods."
        )
    elif slope > 0:
        direction = RISING
        explanation = (
            f"Revenue rose over the observed periods, by about {abs(relative) * 100:.1f}% of the "
            f"series average per period (trend strength {strength:.2f})."
        )
    else:
        direction = FALLING
        explanation = (
            f"Revenue fell over the observed periods, by about {abs(relative) * 100:.1f}% of the "
            f"series average per period (trend strength {strength:.2f})."
        )

    return {
        "direction": direction, "strength": round(strength, 4),
        "slope_per_period": round(slope, 2), "relative_slope": round(relative, 4),
        "explanation": explanation,
    }


def classify_seasonality(
    strength: Optional[float], seasonal_period: Optional[int], frequency: str
) -> Dict[str, object]:
    """
    Interpret Phase 1's measured seasonality strength against the cycle it
    was measured at. A strength of 0.198 at cycle 12 is a WEAK pattern and
    is described as such — not as "seasonality detected".
    """
    if strength is None or not seasonal_period:
        return {
            "level": "NONE", "strength": None, "period": None,
            "explanation": f"There is not enough history to test for a seasonal cycle in this "
                           f"{frequency} series, so no seasonal pattern was measured.",
        }

    value = float(strength)
    unit = {"monthly": "month", "weekly": "week", "quarterly": "quarter", "daily": "day"}.get(
        frequency, "period")
    if value >= SEASONAL_STRONG:
        level, wording = "STRONG", "A strong repeating seasonal pattern"
    elif value >= SEASONAL_MODERATE:
        level, wording = "MODERATE", "A moderate repeating seasonal pattern"
    elif value >= SEASONAL_WEAK:
        level, wording = "WEAK", "Only a weak repeating seasonal pattern"
    else:
        level, wording = "NONE", "No meaningful repeating seasonal pattern"

    return {
        "level": level, "strength": round(value, 4), "period": seasonal_period,
        "explanation": (
            f"{wording} was measured over a {seasonal_period}-{unit} cycle "
            f"(strength {value:.2f})."
        ) + (
            " Seasonal models therefore have little structure to exploit here."
            if level in ("WEAK", "NONE") else
            " Seasonal models can use this structure."
        ),
    }


def classify_volatility(coefficient_of_variation: Optional[float]) -> Dict[str, object]:
    """Describe how much the series itself moves period to period —
    distinct from how much the FORECAST ERROR moves, which reliability
    reports separately."""
    if coefficient_of_variation is None:
        return {"level": "UNKNOWN", "value": None,
                "explanation": "Series variability could not be measured."}

    cv = float(coefficient_of_variation)
    if cv >= VOLATILITY_HIGH:
        level, wording = "HIGH", "highly variable"
    elif cv >= VOLATILITY_MODERATE:
        level, wording = "MODERATE", "moderately variable"
    else:
        level, wording = "LOW", "relatively steady"
    return {
        "level": level, "value": round(cv, 4),
        "explanation": (
            f"Period-to-period revenue is {wording} (coefficient of variation {cv:.2f}). "
        ) + (
            "High variability in the underlying series limits how accurate any forecasting method "
            "can be."
            if level == "HIGH" else
            "This level of variability is workable for forecasting."
            if level == "LOW" else
            "This level of variability puts a practical floor under achievable forecast error."
        ),
    }


def summarise_data_quality(
    n_observations: int, frequency: str, series_density: Optional[float],
    empty_periods: Optional[int], data_quality_score: Optional[float],
    n_transactions: Optional[int] = None,
) -> Dict[str, object]:
    """One user-facing sentence about input integrity, from measured
    values only — no technical dump."""
    density = series_density if series_density is not None else 1.0
    gaps = empty_periods or 0
    quality = data_quality_score if data_quality_score is not None else 1.0
    combined = float(np.mean([density, quality]))

    if combined >= 0.9 and gaps == 0:
        level, wording = "GOOD", "Good"
    elif combined >= 0.7:
        level, wording = "ADEQUATE", "Adequate"
    else:
        level, wording = "LIMITED", "Limited"

    detail = (
        f"{wording} — {n_observations} {frequency} observations"
        + (f" from {n_transactions} transactions" if n_transactions else "")
        + (f", with complete period coverage." if gaps == 0
           else f", but {gaps} period(s) contain no recorded transactions, which reduces how much "
                f"of the series is measured rather than filled.")
    )
    return {"level": level, "score": round(combined, 3), "explanation": detail}


def explain_uncertainty(
    intervals: Optional[dict], metrics: Optional[dict], fold_stats: Optional[dict],
    volatility: Optional[dict], history_status: Optional[str],
) -> Dict[str, object]:
    """
    Why this forecast is uncertain — listing only the drivers that
    actually apply to this series (Step 8). Ordered so the largest
    contributor appears first.
    """
    # Phrased as NOUN PHRASES, not clauses, because they are concatenated
    # after "driven mainly by" — "driven mainly by historical error is
    # high" is not a sentence.
    drivers: List[str] = []
    wape = (metrics or {}).get("wape")
    if wape is not None and wape >= 40:
        drivers.append(f"high historical forecast error ({wape:.1f}% WAPE)")
    stats = fold_stats or {}
    if stats.get("std") and stats.get("mean") and stats["std"] / stats["mean"] >= 0.4:
        drivers.append(
            f"substantial variation in accuracy between validation folds "
            f"({stats['min']:.0f}%-{stats['max']:.0f}% WAPE)"
        )
    if (volatility or {}).get("level") == "HIGH":
        drivers.append("high variability in the underlying revenue series itself")
    if history_status and history_status != "SUFFICIENT_HISTORY":
        drivers.append("limited history available for validation")

    info = intervals or {}
    if not info.get("available"):
        drivers.append("the absence of any estimable prediction interval")
    else:
        if info.get("interval_calibration_status") == "LIMITED_DATA":
            drivers.append(
                "interval calibration resting on a limited number of out-of-sample errors"
            )
        for note in info.get("notes", []):
            if "extrapolated" in note.lower():
                drivers.append(
                    "a 95% level that exceeds what the available validation history supports"
                )
                break

    if not drivers:
        return {
            "drivers": ["no individual driver stands out"],
            "primary": "no individual driver stands out",
            "explanation": (
                "No single driver stands out — uncertainty is at the level the measured error "
                "implies."
            ),
        }

    return {
        "drivers": drivers,
        "primary": drivers[0],
        "explanation": "Forecast uncertainty here is driven mainly by " + drivers[0] + ".",
    }


# ── Assembly (Step 10 & 20) ────────────────────────────────────────────────

def build_insights(
    trend: Dict[str, object],
    seasonality: Dict[str, object],
    volatility: Dict[str, object],
    data_quality: Dict[str, object],
    uncertainty: Dict[str, object],
    reliability_level: str,
    reliability_summary: str,
    model_explanation: ModelExplanation,
    scope: str = "Company Total",
) -> List[Insight]:
    """
    The concise, data-driven insight list the UI renders. Each entry is
    selected by a measurement and carries the statistic that produced it.
    """
    insights = [
        Insight("trend", f"Revenue trend: {str(trend['direction']).title()}",
                str(trend["explanation"]), f"trend strength {trend.get('strength')}"),
        Insight("volatility", f"Series variability: {str(volatility['level']).title()}",
                str(volatility["explanation"]), f"coefficient of variation {volatility.get('value')}"),
        Insight("seasonality", f"Seasonality: {str(seasonality['level']).title()}",
                str(seasonality["explanation"]),
                f"seasonal strength {seasonality.get('strength')} at cycle {seasonality.get('period')}"),
        Insight("data_quality", f"Data quality: {str(data_quality['level']).title()}",
                str(data_quality["explanation"]), f"combined score {data_quality.get('score')}"),
        Insight("reliability", f"Forecast reliability: {reliability_level.replace('_', ' ').title()}",
                reliability_summary, "see reliability factors"),
        Insight("model", f"Model: {model_explanation.selected_model}",
                model_explanation.strength_reason,
                f"selection strength {model_explanation.selection_strength}"),
        Insight("uncertainty", "Main source of uncertainty",
                str(uncertainty["explanation"]), str(uncertainty.get("primary"))),
    ]

    # A single recommendation, keyed to reliability. Deliberately about
    # HOW TO USE the forecast — the one actionable statement the measured
    # evidence actually supports.
    if reliability_level == "HIGH":
        recommendation = (
            f"{scope} can be used for planning with normal review. Track outcomes against the 80% "
            f"prediction range."
        )
    elif reliability_level == "MODERATE":
        recommendation = (
            f"Use {scope} as a planning estimate rather than a precise target, and plan against "
            f"the 80% prediction range rather than the single point value."
        )
    elif reliability_level == "LOW":
        recommendation = (
            f"Treat {scope} as directional only. The prediction range is wide enough that "
            f"committing to the point value would be unwise."
        )
    else:
        recommendation = (
            f"{scope} does not currently support reliable planning. More history, or forecasting "
            f"at a broader level, would give a firmer basis."
        )
    insights.append(Insight("recommendation", "Recommendation", recommendation,
                            f"reliability level {reliability_level}"))
    return insights
