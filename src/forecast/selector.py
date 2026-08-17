"""
selector.py
------------
Automatic Model Selection (Phase 2, Step 4). Ranks every trained model by
validation performance and picks the best one, with a human-readable
explanation. This is distinct from Phase 1's model_selector.py, which
recommends CANDIDATE models before any training happens; this module picks
the FINAL model after training and evaluation.

Redesign pass (Tasks 7, 8, 10, 12, 13 of the forecasting-subsystem
redesign — builds on top of the earlier production-quality pass, does not
undo it):

  - Ranking (Task 7) stays anchored on the WAPE/SMAPE/RMSE/MAE lexicographic
    key from the earlier pass — that remains the primary, empirical
    ordering, because the stated objective is to MAXIMISE accuracy, and a
    single blended "weighted score" across metrics of very different scale
    (a percentage-of-actuals metric and an absolute-currency metric) is
    easy to game and hard to reason about. What is new: a small
    business-aware TIE-BREAK (Task 12) that only ever reorders models
    already within _TIE_BREAK_RELATIVE_BAND of each other — never a model
    with a meaningfully worse empirical error over one with a better one —
    using stability, validated-fold count, and how well the model's
    capabilities (trend/seasonal/robust-to-volatility) match this
    particular series' detected characteristics.
  - Confidence (Task 8) is now built from EIGHT weighted signals instead
    of the earlier pass' six, each capped, summing to a 0-100 score:
    history length, validation folds, fold-to-fold stability, accuracy
    (primary metric), margin over the runner-up, WAPE/SMAPE agreement,
    series-characteristic fit (new — trend/seasonality strength matched to
    model capability), and input data quality (new — how much correction
    the raw data needed, from forecast.preprocessing). The legacy 0-1
    `confidence` field is kept (Task 17 — do not break the existing
    contract the frontend already reads); the 0-100 score is additionally
    exposed as `confidence_pct` on SelectionResult, an ADDITIVE field.
  - Ensemble blending (Task 10) is now N-way (up to
    _ENSEMBLE_MAX_MODELS) and adaptive: every model within
    _ENSEMBLE_RELATIVE_BAND of the best model's error is included, weighted
    by inverse validation error (Bates-Granger forecast combination),
    instead of the earlier pass' fixed top-2/10%-band rule. A model that
    is not close to competitive is never pulled in just to pad the
    ensemble out.
  - Explanations (Task 13) now additionally cite the specific
    hyperparameters a model's own suggest_hyperparameters() search chose,
    whether a business-aware tie-break changed the pick, and the input
    data quality score when it was materially imperfect — on top of the
    earlier pass' metric/fold/series-characteristic reasons.
"""

import math
from typing import Dict, List, Optional, Tuple

from .backtest import BacktestResult, EnsembleBacktest, evaluate_ensemble
from .results import ModelEvalResult, SelectionResult
from .types import SeriesStats, SeriesClassification

# Phase 2, Step 9 — complexity ordering, simplest first. Used ONLY to
# break an empirical tie: when two models are statistically
# indistinguishable on validation error, the simpler one is preferred,
# because it has fewer ways to be wrong on data it has not seen and its
# behaviour is easier to explain. A more complex model has to actually
# demonstrate better generalisation (a lower error outside the tie band)
# to be chosen over a simpler one.
_COMPLEXITY_RANK = {
    "Naive": 0,
    "Seasonal Naive": 1,
    "Moving Average": 1,
    "Simple Exponential Smoothing": 2,
    "Holt Linear": 3,
    "Holt-Winters": 4,
    "SARIMA": 5,
    "Croston": 3,
    "Prophet": 6,
    "XGBoost": 7,
}
_MAX_COMPLEXITY = max(_COMPLEXITY_RANK.values())

# Phase 2, Step 10 — an ensemble must beat the best single model's pooled
# validation error by at least this relative margin to be selected. A
# blend is harder to explain and to reason about than one model, so an
# indistinguishable improvement is not worth it. There is no longer any
# unconditional bonus for being an ensemble.
_ENSEMBLE_MIN_IMPROVEMENT = 0.01

# Task 7 (fallback path) — priority when no model has validation metrics at
# all. Most statistically sophisticated / typically most accurate first, so
# a "no metrics available" dataset never defaults to Moving Average just
# because it happened to train first.
_FALLBACK_PRIORITY = [
    "Prophet", "SARIMA", "Holt-Winters", "Holt Linear",
    "Simple Exponential Smoothing", "Moving Average", "Naive",
]

# Task 12 — which models can, in principle, represent a trend / a seasonal
# cycle / stay robust under high volatility / are simple-and-robust for
# short or thin history. Used both for the business-aware tie-break and to
# keep Task 13's explanations specific instead of generic (e.g. never claim
# a model "captured seasonality" if it has no seasonal component at all).
_TREND_CAPABLE = {"Holt Linear", "Holt-Winters", "SARIMA", "Prophet", "XGBoost"}
_SEASONAL_CAPABLE = {"Holt-Winters", "SARIMA", "Prophet", "XGBoost"}
_VOLATILITY_ROBUST = {"XGBoost", "Prophet", "SARIMA"}
_SIMPLE_ROBUST = {"Naive", "Moving Average", "Simple Exponential Smoothing"}
_SHORT_HISTORY_OBS = 12  # below this, a simple model's robustness starts to matter more than a sophisticated model's flexibility

# Task 12 — models within this fraction of the leader's own error are
# treated as an empirical tie and may be reordered by the business-aware
# tie-break; deliberately much tighter than the ensemble band below, since
# reordering WHICH model is called "the winner" is a stronger claim than
# including a close competitor in a blended ensemble.
_TIE_BREAK_RELATIVE_BAND = 0.05

# Task 10 — adaptive ensemble: every model within this fraction of the
# leader's error is included (inverse-error weighted), up to
# _ENSEMBLE_MAX_MODELS members. Widened from the earlier pass' fixed
# top-2/10% rule so a genuinely close 3-way or 4-way race is not
# arbitrarily narrowed to two, while still excluding models that are not
# actually competitive.
_ENSEMBLE_RELATIVE_BAND = 0.25
_ENSEMBLE_MAX_MODELS = 4


def _primary_metric_value(metrics: Optional[Dict[str, float]]) -> Optional[float]:
    """WAPE if available, else SMAPE, else None — "the best available
    metric" used consistently everywhere a single accuracy number is
    needed (tie-break/ensemble banding, confidence's margin/accuracy
    components), independent of the full 4-metric lexicographic rank key
    used for the primary sort."""
    if not metrics:
        return None
    for key in ("wape", "smape"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _metric_or_inf(metrics: Optional[Dict[str, float]], key: str) -> float:
    v = (metrics or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("inf")


def _rank_key(result: ModelEvalResult) -> Tuple[float, float, float, float]:
    """
    Task 7 — multi-metric ranking key, ascending (lower is better): WAPE,
    then SMAPE, then RMSE, then MAE. WAPE is primary because it is a ratio
    of SUMS (robust to individual zero/near-zero actuals — see
    evaluator.py); MAPE is deliberately excluded so it can never decide a
    winner, even as a tie-break. Missing metrics sort as +inf (worst), so
    a model missing one or more of these still ranks — just always behind
    any model that has a real number for that tier.
    """
    m = result.metrics
    return (
        _metric_or_inf(m, "wape"),
        _metric_or_inf(m, "smape"),
        _metric_or_inf(m, "rmse"),
        _metric_or_inf(m, "mae"),
    )


def _pick_fallback(trained_no_metrics: List[ModelEvalResult]) -> ModelEvalResult:
    """Fallback when nothing has validation metrics — see module docstring."""
    by_name = {r.model: r for r in trained_no_metrics}
    for name in _FALLBACK_PRIORITY:
        if name in by_name:
            return by_name[name]
    return trained_no_metrics[0]  # defensive: a model outside the known priority list


def _stability_component(result: ModelEvalResult) -> float:
    if result.stability is None:
        return 0.5  # unknown stability -- neutral, neither penalised nor rewarded
    return max(0.0, min(1.0, result.stability))


def _business_fit_score(
    result: ModelEvalResult,
    stats: Optional[SeriesStats],
    classification: Optional[SeriesClassification],
) -> float:
    """
    Task 12 — 0-1 "does this model's capability match what this series
    actually looks like" score. Starts at a neutral 0.5 (no information
    either way) and is nudged up when the model's known capability
    (trend/seasonal/volatility-robust/simple-robust) matches a detected
    series characteristic, nudged down when a sophisticated model is
    applied to a series too short to justify it. This is a TIE-BREAK and
    CONFIDENCE signal only — it never substitutes for the empirical
    accuracy ranking (see _apply_business_aware_tie_break's banding).
    """
    score = 0.5
    if stats is not None:
        if stats.trend_strength >= 0.4:
            score += 0.15 if result.model in _TREND_CAPABLE else -0.05
        if stats.seasonality_strength >= 0.4:
            score += 0.15 if result.model in _SEASONAL_CAPABLE else -0.05
        if stats.n_observations < _SHORT_HISTORY_OBS and result.model in _SIMPLE_ROBUST:
            score += 0.10
    if classification is not None and classification.series_type == "Highly Volatile":
        score += 0.10 if result.model in _VOLATILITY_ROBUST else -0.05
    return max(0.0, min(1.0, score))


def _simplicity_component(result: ModelEvalResult) -> float:
    """1.0 for the simplest model in the library, 0.0 for the most
    complex. Phase 2, Step 9 — see _COMPLEXITY_RANK."""
    rank = _COMPLEXITY_RANK.get(result.model)
    if rank is None:
        return 0.5  # unknown model — neutral, neither rewarded nor penalised
    return 1.0 - (rank / _MAX_COMPLEXITY)


def _tie_break_score(
    result: ModelEvalResult,
    stats: Optional[SeriesStats],
    classification: Optional[SeriesClassification],
) -> float:
    """
    Composite score used ONLY to order models that are already an
    empirical accuracy tie (see _TIE_BREAK_RELATIVE_BAND). Weighted blend
    of fold-to-fold stability, simplicity, how many folds actually
    validated the model, and how well its capabilities match the series.

    Phase 2 reweighting: stability now carries the most weight and
    simplicity is a first-class term. Both express the same underlying
    preference — when the evidence cannot distinguish two models, take the
    one whose behaviour is most predictable and whose failure modes are
    easiest to understand, rather than the one that merely looks more
    sophisticated.
    """
    fit = _business_fit_score(result, stats, classification)
    stability = _stability_component(result)
    simplicity = _simplicity_component(result)
    folds_norm = min(1.0, (result.validation_folds or 0) / 8.0)
    return 0.30 * stability + 0.30 * simplicity + 0.20 * fit + 0.20 * folds_norm


def _apply_business_aware_tie_break(
    scored: List[ModelEvalResult],
    stats: Optional[SeriesStats],
    classification: Optional[SeriesClassification],
) -> Tuple[List[ModelEvalResult], bool]:
    """
    Task 12 — reorders only the leading group of models whose primary
    metric is within _TIE_BREAK_RELATIVE_BAND of the current leader
    (an empirical tie too close to call on accuracy alone), by
    _tie_break_score. Everything outside that leading group keeps its
    original (already accuracy-sorted) order untouched. Returns
    (possibly-reordered list, whether the top pick actually changed) so
    the caller can mention the tie-break in the explanation only when it
    mattered.
    """
    if len(scored) < 2:
        return scored, False
    best_v = _primary_metric_value(scored[0].metrics)
    if best_v is None or best_v <= 0:
        return scored, False

    tie_group_len = 1
    for r in scored[1:]:
        v = _primary_metric_value(r.metrics)
        if v is None or (v - best_v) / best_v > _TIE_BREAK_RELATIVE_BAND:
            break
        tie_group_len += 1
    if tie_group_len < 2:
        return scored, False

    tie_group = scored[:tie_group_len]
    reordered_group = sorted(tie_group, key=lambda r: -_tie_break_score(r, stats, classification))
    changed = reordered_group[0].model != tie_group[0].model
    return reordered_group + scored[tie_group_len:], changed


def _compute_confidence(
    best: ModelEvalResult,
    runner_up: Optional[ModelEvalResult],
    n_observations: int,
    stats: Optional[SeriesStats] = None,
    classification: Optional[SeriesClassification] = None,
    data_quality: Optional[float] = None,
) -> Tuple[float, int]:
    """
    Task 8 — confidence from eight weighted, capped signals summing to a
    0-100 score (returned alongside the legacy 0-1 value the frontend
    already reads, so nothing downstream needs to change scale):

      1. History available            (0-15)
      2. Validation folds              (0-15)
      3. Fold-to-fold stability        (0-15)  -- also serves as this
                                                   codebase's proxy for
                                                   "forecast/residual
                                                   stability" (no raw
                                                   residual series is
                                                   available here)
      4. Accuracy (primary metric)     (0-20)  -- largest single
                                                   component, still well
                                                   under half the total
      5. Margin over the runner-up     (0-10)
      6. WAPE/SMAPE agreement          (0-10)
      7. Series-characteristic fit     (0-10)  -- NEW: trend/seasonality
                                                   strength matched to
                                                   this model's known
                                                   capability
      8. Input data quality            (0-5)   -- NEW: how much
                                                   correction the raw
                                                   data needed (see
                                                   forecast.preprocessing)

    Clipped to [5, 97] so nothing ever reads as absolutely certain or
    absolutely worthless.
    """
    m = best.metrics or {}

    n = n_observations or 0
    if n >= 52:
        hist = 15.0
    elif n >= 24:
        hist = 12.0
    elif n >= 12:
        hist = 7.0
    elif n >= 6:
        hist = 3.0
    else:
        hist = 0.0

    folds = best.validation_folds or 0
    if folds >= 8:
        fold_pts = 15.0
    elif folds >= 4:
        fold_pts = 11.0
    elif folds >= 1:
        fold_pts = 6.0
    else:
        fold_pts = 0.0

    stability_pts = 15.0 * _stability_component(best)

    primary = _primary_metric_value(m)
    if primary is not None:
        accuracy_pts = 20.0 * max(0.0, 1 - min(1.0, primary / 80.0))
    else:
        accuracy_pts = 6.0  # no usable accuracy metric -- small flat credit, not zero, since the model DID train

    margin_pts = 0.0
    if runner_up is not None:
        rp = _primary_metric_value(runner_up.metrics)
        if primary is not None and rp is not None and rp > 0:
            relative_margin = max(0.0, (rp - primary) / rp)
            margin_pts = min(10.0, relative_margin * 10.0)

    wape_v, smape_v = m.get("wape"), m.get("smape")
    if wape_v is not None and smape_v is not None:
        gap = abs(wape_v - smape_v) / max(wape_v, smape_v, 1.0)
        agreement_pts = 10.0 * max(0.0, 1 - min(1.0, gap))
    else:
        agreement_pts = 3.0

    fit_pts = 10.0 * _business_fit_score(best, stats, classification)

    if data_quality is None:
        quality_pts = 3.5  # unknown -- mild positive default, not penalised for missing information
    else:
        quality_pts = 5.0 * max(0.0, min(1.0, data_quality))

    total = hist + fold_pts + stability_pts + accuracy_pts + margin_pts + agreement_pts + fit_pts + quality_pts
    pct = round(max(5.0, min(97.0, total)))
    return round(pct / 100.0, 2), int(pct)


def _ensemble_candidates(scored: List[ModelEvalResult]) -> List[str]:
    """
    Which models are close enough to the leader to be worth blending. This
    only proposes a membership set — whether the resulting blend is
    actually used is decided by measuring it (see
    `_select_evaluated_ensemble`), never assumed.
    """
    if len(scored) < 2:
        return []
    best_v = _primary_metric_value(scored[0].metrics)
    if best_v is None or best_v <= 0:
        return []

    members = [scored[0].model]
    for r in scored[1:]:
        if len(members) >= _ENSEMBLE_MAX_MODELS:
            break
        v = _primary_metric_value(r.metrics)
        if v is None or (v - best_v) / best_v > _ENSEMBLE_RELATIVE_BAND:
            break  # sorted ascending -- nothing further out can qualify either
        members.append(r.model)
    return members if len(members) >= 2 else []


def _select_evaluated_ensemble(
    scored: List[ModelEvalResult],
    backtests: Optional[Dict[str, BacktestResult]],
) -> Tuple[Optional[EnsembleBacktest], Optional[str]]:
    """
    Phase 2, Step 10 — build the ensemble's ACTUAL predictions on every
    validation fold, measure its real error, and only adopt it if it beats
    the best single model.

    Previously an ensemble was assembled from the ranking whenever two
    models fell inside a band, its forecast was blended, and the metrics
    REPORTED were the best component's — so the number on screen described
    something other than the thing producing the forecast, and the blend
    additionally received an unconditional +3 confidence bonus for
    existing. Both are gone.

    Returns (ensemble_or_None, decision_note). The note explains the
    outcome either way, including when a measured ensemble was rejected
    for not being better, so the decision is visible rather than silent.
    """
    if not backtests:
        return None, None

    names = _ensemble_candidates(scored)
    if len(names) < 2:
        return None, None

    components = [backtests[n] for n in names if n in backtests and backtests[n].folds]
    if len(components) < 2:
        return None, None

    ensemble = evaluate_ensemble(components)
    if ensemble is None or not ensemble.pooled_metrics:
        return None, None

    ensemble_v = _primary_metric_value(ensemble.pooled_metrics)
    best_v = _primary_metric_value(scored[0].metrics)
    if ensemble_v is None or best_v is None:
        return None, None

    improvement = (best_v - ensemble_v) / best_v if best_v > 0 else 0.0
    if improvement >= _ENSEMBLE_MIN_IMPROVEMENT:
        note = (
            f"A weighted ensemble of {' + '.join(ensemble.components)} was built and scored on the "
            f"same {len(ensemble.folds)} validation folds as its components. It achieved "
            f"{round(ensemble_v, 2)}% WAPE against {scored[0].model}'s {round(best_v, 2)}% — an "
            f"improvement of {round(improvement * 100, 1)}%, so the ensemble was selected. The "
            f"metrics reported for it are the ensemble's own measured performance, not any "
            f"component's."
        )
        return ensemble, note

    note = (
        f"A weighted ensemble of {' + '.join(ensemble.components)} was built and scored on the same "
        f"{len(ensemble.folds)} validation folds, but achieved {round(ensemble_v, 2)}% WAPE against "
        f"{scored[0].model}'s {round(best_v, 2)}% — not the {round(_ENSEMBLE_MIN_IMPROVEMENT * 100)}% "
        f"improvement required to justify the added complexity, so the single model was kept."
    )
    return None, note


def _explain_selection(
    best: ModelEvalResult,
    runner_up: Optional[ModelEvalResult],
    stats: Optional[SeriesStats],
    classification: Optional[SeriesClassification],
    is_ensemble: bool = False,
    tie_break_changed_pick: bool = False,
    data_quality: Optional[float] = None,
) -> List[str]:
    """Task 13 — reasons are built only from things actually true of THIS
    model and THIS series, never a fixed boilerplate string."""
    m = best.metrics or {}
    reasons: List[str] = []

    if m.get("wape") is not None:
        reasons.append(f"Lowest WAPE among tested models ({m['wape']}%) — the most reliable accuracy measure for this kind of revenue data.")
    elif m.get("smape") is not None:
        reasons.append(f"Lowest SMAPE among tested models ({m['smape']}%) — WAPE was not computable for this series (its validation window had no non-zero actuals), so SMAPE was used instead.")
    elif m.get("rmse") is not None:
        reasons.append(f"Lowest RMSE among tested models ({m['rmse']}) — no percentage-based metric was computable for this series.")

    if best.validation_folds and best.validation_folds >= 3:
        reasons.append(f"Stable during walk-forward validation across {best.validation_folds} folds.")
    elif (best.validation_folds or 0) == 0:
        reasons.append("Trained on all available history; too little data for repeated walk-forward validation, so this is not yet a cross-validated accuracy estimate.")

    if best.stability is not None and best.stability >= 0.7:
        reasons.append(f"Consistent error from fold to fold (stability {best.stability}), not just a lucky average.")

    if best.hyperparameters:
        params_str = ", ".join(f"{k}={v}" for k, v in best.hyperparameters.items())
        reasons.append(f"Hyperparameters were tuned automatically for this series ({params_str}) rather than left at fixed defaults.")

    if stats is not None:
        if stats.trend_strength >= 0.4 and best.model in _TREND_CAPABLE:
            reasons.append(f"Captured this series' trend (strength {stats.trend_strength}) directly.")
        if stats.seasonality_strength >= 0.4 and best.model in _SEASONAL_CAPABLE:
            reasons.append(f"Captured this series' seasonal pattern (strength {stats.seasonality_strength}) directly.")
        if stats.history_length_days >= 730:
            reasons.append(f"Long historical data available ({stats.history_length_days} days).")
        elif stats.n_observations < 12:
            reasons.append(f"Limited history ({stats.n_observations} observations) — treat this forecast with caution.")

    if classification is not None:
        if classification.series_type == "Highly Volatile" and best.model in _VOLATILITY_ROBUST:
            reasons.append("More robust to this series' high volatility than a simple average or naive baseline.")
        reasons.append(f"Series classified as {classification.series_type} (confidence {classification.confidence}).")

    if tie_break_changed_pick:
        reasons.append(
            "This model was statistically tied on accuracy with at least one other candidate; it was "
            "preferred because its capabilities better match this series' detected trend/seasonality and "
            "it was more stable and better-validated across walk-forward folds."
        )

    if data_quality is not None and data_quality < 0.9:
        reasons.append(
            f"The input data required cleaning before training (data quality score {round(data_quality, 2)}/1.0) "
            f"— this was factored into the confidence score above."
        )

    if is_ensemble:
        reasons.insert(
            0,
            "Multiple models had statistically similar validation accuracy, so their forecasts were "
            "combined into a weighted ensemble rather than picking one arbitrarily.",
        )

    return reasons


def select_best_model(
    results: List[ModelEvalResult],
    stats: Optional[SeriesStats] = None,
    classification: Optional[SeriesClassification] = None,
    data_quality: Optional[float] = None,
    backtests: Optional[Dict[str, BacktestResult]] = None,
) -> SelectionResult:
    """
    Rank all trained models and select the best (Tasks 7, 10, 12). If no
    model produced validation metrics (e.g. dataset too short to
    validate), fall back to the most capable model that still trained, in
    a fixed priority order, rather than raising or picking arbitrarily.

    `data_quality` is an optional 0-1 score from
    forecast.preprocessing.compute_data_quality_score(); passing None
    (the default) keeps this fully backward compatible with any existing
    caller that does not yet have that value available (Task 17).
    """
    scored = [r for r in results if r.trained and r.metrics is not None]

    if not scored:
        trained_no_metrics = [r for r in results if r.trained]
        if not trained_no_metrics:
            # Nothing could be trained at all — surface this explicitly
            # rather than letting forecast_engine crash on a KeyError.
            return SelectionResult(
                selected_model="None",
                reason=["No candidate model could be trained on this dataset."],
                metrics=None,
                confidence=0.0,
                ranking=results,
                confidence_pct=0,
            )
        fallback = _pick_fallback(trained_no_metrics)
        fallback.rank = 1
        return SelectionResult(
            selected_model=fallback.model,
            reason=[
                "Insufficient historical data for validation metrics on any model — selected "
                f"{fallback.model} as the most capable model that still trained successfully, "
                "following a fixed fallback order (Prophet > SARIMA > Holt-Winters > Holt Linear "
                "> Simple Exponential Smoothing > Moving Average > Naive) rather than an "
                "arbitrary pick."
            ],
            metrics=None,
            confidence=0.25,
            ranking=results,
            confidence_pct=25,
        )

    # Task 7: multi-metric ranking, WAPE primary.
    scored.sort(key=_rank_key)

    # Task 12: business-aware tie-break among models within an empirical
    # accuracy tie — never reorders anything outside that tie band.
    scored, tie_break_changed_pick = _apply_business_aware_tie_break(scored, stats, classification)

    for i, r in enumerate(scored, start=1):
        r.rank = i

    unscored = [r for r in results if r not in scored]
    full_ranking = scored + unscored

    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    n_obs = stats.n_observations if stats is not None else 0
    confidence, confidence_pct = _compute_confidence(
        best, runner_up, n_obs, stats=stats, classification=classification, data_quality=data_quality,
    )

    # Step 10: build the ensemble's real predictions across every fold,
    # measure them, and adopt it ONLY if it actually beats the best single
    # model. No unconditional bonus of any kind.
    ensemble, ensemble_note = _select_evaluated_ensemble(scored, backtests)

    if ensemble is not None:
        reasons = _explain_selection(
            best, runner_up, stats, classification,
            is_ensemble=True, tie_break_changed_pick=tie_break_changed_pick, data_quality=data_quality,
        )
        if ensemble_note:
            reasons.insert(0, ensemble_note)
        reasons.append(
            "Blend weights: "
            + ", ".join(f"{m} {round(w * 100)}%" for m, w in ensemble.weights.items())
            + f" — {ensemble.weight_basis}."
        )
        return SelectionResult(
            selected_model=ensemble.name,
            reason=reasons,
            # The ensemble's OWN measured metrics, never a component's.
            metrics=ensemble.pooled_metrics,
            # Confidence is computed from the same eight signals as any
            # other selection, with no ensemble-specific adjustment.
            confidence=confidence,
            ranking=full_ranking,
            ensemble_components=[
                {"model": m, "weight": w} for m, w in ensemble.weights.items()
            ],
            confidence_pct=confidence_pct,
            ensemble_evaluation=ensemble.to_dict(),
        )

    reasons = _explain_selection(
        best, runner_up, stats, classification,
        is_ensemble=False, tie_break_changed_pick=tie_break_changed_pick, data_quality=data_quality,
    )
    if ensemble_note:
        # An ensemble was measured and rejected — say so, rather than
        # leaving the decision invisible.
        reasons.append(ensemble_note)
    return SelectionResult(
        selected_model=best.model,
        reason=reasons,
        metrics=best.metrics,
        confidence=confidence,
        ranking=full_ranking,
        confidence_pct=confidence_pct,
    )
