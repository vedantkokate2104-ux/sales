"""
forecast_engine.py
-------------------
Top-level orchestrator for the Auto Forecasting Engine (Phase 2). Ties
Phase 1 (analyzer -> classifier -> model_selector, all unmodified) to
Phase 2 (trainer -> evaluator -> selector -> forecast generation) and
returns one consolidated AutoForecastResult for the Flask API / frontend.

This is the ONLY new module the frontend/API needs to call for the full
Phase 2 pipeline — everything else in forecast/ is an implementation
detail behind this function.

Phase E change: the dataset's frequency is now read once from Phase 1's
already-computed `stats.frequency` (SeriesStats, produced by analyzer.py's
existing utils.infer_frequency() call) and threaded through to
train_candidates(), instead of the trainer silently re-deriving it — or
worse, ignoring it and always assuming monthly. This guarantees the
"Dataset Analysis" panel and the trainer always agree on what cadence the
data is at. Dataset-quality validation (Task 2) also runs here, once, and
its messages are merged into the same `warnings` list the frontend already
renders — no new API field needed for that part.

Stability pass (runtime fault-tolerance redesign, Tasks 3 & 7): a model
finishing training/validation successfully does not guarantee its FINAL
`.forecast(horizon)` call — made here, after selection, outside
trainer.py's own per-candidate try/except — will also succeed (e.g. an
otherwise-healthy Prophet whose isolated call happens to fail only on this
particular request). Previously neither the single-model path nor the
ensemble-blending path guarded that call, so a failure here would
propagate all the way to Flask's route-level handler as an HTTP 500 even
though other trained models were sitting right there, ready to forecast.
Both paths now catch a forecast() failure and fall back to the next
trained model instead — the engine only reports "no forecast available"
if every trained model's forecast() call fails, matching Task 7's
"only return an error if every forecasting model fails."
"""

import logging
from typing import Optional

import pandas as pd

from analytics import get_col
from .analyzer import analyze_dataset      # Phase 1, unmodified
from .classifier import classify_series    # Phase 1, unmodified
from .model_selector import recommend_models  # Phase 1, unmodified
from .trainer import train_candidates
from .selector import select_best_model
from .preprocessing import compute_data_quality_score
from .results import AutoForecastResult, ForecastPoint
from .validation import validate_dataset

logger = logging.getLogger(__name__)


def run_auto_forecast(
    df: pd.DataFrame,
    mappings: list,
    horizon: int = 3,
    group_col: Optional[str] = None,
    group_value: Optional[str] = None,
) -> AutoForecastResult:
    """
    Run the full Phase 1 + Phase 2 pipeline for one series and return a
    single consolidated result.

    By default this analyses the whole dataset as one series (the same
    overall revenue-over-time series the existing Forecast page already
    charts). Passing group_col/group_value restricts the input to one
    product/segment first — the trainer and every model below are
    already series-agnostic, so per-product forecasting requires no
    further changes; only the frontend does not yet expose it (documented
    in CHANGELOG.md).
    """
    working_df = df
    if group_col and group_value is not None and group_col in df.columns:
        working_df = df[df[group_col].astype(str) == str(group_value)]

    # ── Phase 1 (reused, unmodified) ──────────────────────────────────────
    stats = analyze_dataset(working_df, mappings, group_col=None)["ALL"]
    classification = classify_series(stats)
    candidates = recommend_models(classification, stats)
    candidate_names = [c.model for c in candidates]

    date_col = get_col(working_df, mappings, "invoice date")
    rev_col = get_col(working_df, mappings, "net sales")
    if not date_col or not rev_col:
        raise ValueError(
            "Auto forecasting requires both an invoice date and a net sales "
            "column to be resolved by the schema mapper."
        )

    # ── Task 2: dataset-quality validation, run once before training ────────
    validation_warnings = validate_dataset(working_df[date_col], working_df[rev_col], stats)

    # ── Phase 2 ────────────────────────────────────────────────────────────
    # stats.frequency was already computed by Phase 1's analyzer — reused
    # here (rather than re-detected) so the trainer always aggregates at
    # the same cadence the Dataset Analysis panel already reports.
    eval_results, trained_models, training_warnings, series, preprocessing_info = train_candidates(
        working_df[date_col], working_df[rev_col], candidate_names, frequency=stats.frequency
    )
    warnings = validation_warnings + training_warnings

    # Task 8 — a single 0-1 summary of how much correction the raw data
    # needed, fed into selector.py's confidence calculation as one signal
    # among several (never the sole determinant of confidence).
    data_quality = compute_data_quality_score(
        counts=preprocessing_info,
        n_capped=preprocessing_info.get("n_capped", 0),
        n_series_points=preprocessing_info.get("n_series_points", 0),
        missing_timestamp_ratio=preprocessing_info.get("missing_timestamp_ratio", 0.0),
    )

    if not trained_models:
        return AutoForecastResult(
            horizon=horizon,
            selected_model="None",
            selection_reason=["No candidate model could be trained on this dataset."],
            evaluation_metrics=None,
            forecast_confidence=0.0,
            model_ranking=[r.to_dict() for r in eval_results],
            forecast=[],
            alternative_models=[],
            explainability=_explainability(stats, classification),
            warnings=warnings,
            frequency=stats.frequency,
            forecast_confidence_pct=0,
            data_quality_score=data_quality,
        )

    selection = select_best_model(eval_results, stats=stats, classification=classification, data_quality=data_quality)

    if selection.ensemble_components:
        # Task 8 — the selected "model" is a blend of the top two, not a
        # single MODEL_REGISTRY entry, so it is never looked up in
        # trained_models by name; each component is forecast
        # independently and the results are combined with the same
        # inverse-error weights selector.py already computed.
        forecast_pairs, ensemble_warning = _blend_ensemble_forecast(
            selection.ensemble_components, trained_models, horizon
        )
        if ensemble_warning:
            warnings.append(ensemble_warning)
        ensemble_names = {c["model"] for c in selection.ensemble_components}
        alternatives = [
            r.model for r in eval_results
            if r.trained and r.model not in ensemble_names
        ]
    else:
        forecast_pairs, used_model, forecast_warning = _forecast_with_fallback(
            selection.selected_model, trained_models, eval_results, horizon,
        )
        if forecast_warning:
            warnings.append(forecast_warning)
        if used_model is not None and used_model != selection.selected_model:
            # The originally-selected model failed only at this final
            # forecast-generation step; a different trained model
            # actually produced the numbers below — keep `selected_model`
            # truthful about that rather than silently mislabelling whose
            # forecast the user is looking at.
            selection.selected_model = used_model
        alternatives = [
            r.model for r in eval_results
            if r.trained and r.model != selection.selected_model
        ]

    # Task 7 — only report "no forecast" if literally nothing could
    # produce one (every trained model's forecast() call itself failed,
    # on top of already having survived training). This is different from
    # (and rarer than) `not trained_models` above: a model can train and
    # validate successfully and still fail specifically at this final
    # forecast-generation step (e.g. an intermittent Prophet failure).
    if not forecast_pairs:
        warnings.append("Every trained model failed to produce a forecast for the requested horizon.")
        return AutoForecastResult(
            horizon=horizon,
            selected_model="None",
            selection_reason=selection.reason + ["No trained model could generate a forecast."],
            evaluation_metrics=selection.metrics,
            forecast_confidence=0.0,
            model_ranking=[r.to_dict() for r in selection.ranking],
            forecast=[],
            alternative_models=[],
            explainability=_explainability(stats, classification),
            warnings=warnings,
            frequency=stats.frequency,
            forecast_confidence_pct=0,
            data_quality_score=data_quality,
        )

    forecast_points = [ForecastPoint(period=p, value=round(v, 2)) for p, v in forecast_pairs]

    return AutoForecastResult(
        horizon=horizon,
        selected_model=selection.selected_model,
        selection_reason=selection.reason,
        evaluation_metrics=selection.metrics,
        forecast_confidence=selection.confidence,
        model_ranking=[r.to_dict() for r in selection.ranking],
        forecast=[p.to_dict() for p in forecast_points],
        alternative_models=alternatives,
        explainability=_explainability(stats, classification),
        warnings=warnings,
        frequency=stats.frequency,
        forecast_confidence_pct=selection.confidence_pct,
        data_quality_score=data_quality,
    )


def _forecast_with_fallback(preferred_model: str, trained_models: dict, eval_results: list, horizon: int):
    """
    Task 3/7 (runtime fault-tolerance) — call .forecast(horizon) on the
    preferred (selector.py-chosen) model, and if that call itself fails —
    an ordinary exception is all this can ever see; process_isolation.py
    already ensures a native crash inside a model like Prophet can never
    reach this point as anything other than one — fall through the
    remaining trained models, in their ranked order, until one succeeds.

    A model finishing training/validation does not guarantee its LATER,
    separate .forecast() call (made here, after selection, outside
    trainer.py's own per-candidate try/except) will also succeed — e.g. an
    otherwise-healthy model whose isolated call happens to fail only on
    this particular request. Without this fallback, that single failure
    would surface as a hard error even though other trained models are
    available right here, ready to forecast.

    Returns (forecast_pairs, model_name_actually_used, warning_or_None).
    forecast_pairs is [] and model_name_actually_used is None only if
    EVERY trained model's forecast() call failed.
    """
    ordered_names = [preferred_model] + [
        r.model for r in eval_results if r.trained and r.model != preferred_model
    ]
    seen = set()
    failed: list = []

    for name in ordered_names:
        if name in seen or name not in trained_models:
            continue
        seen.add(name)
        model_obj = trained_models[name]
        try:
            forecast_pairs = model_obj.forecast(horizon)
            if name != preferred_model:
                logger.warning(
                    "%s failed to produce a forecast — fell back to %s", preferred_model, name,
                )
            warning = (
                f"'{preferred_model}' was selected but its forecast could not be used (either it "
                f"was not actually trained, or it failed specifically at this final step) — fell "
                f"back to '{name}'."
            ) if name != preferred_model else None
            return forecast_pairs, name, warning
        except Exception as exc:
            logger.exception("%s failed during final forecast generation", name)
            failed.append(name)
            continue

    warning = (
        f"Every trained model failed to produce a forecast ({', '.join(failed)})."
        if failed else None
    )
    return [], None, warning


def _blend_ensemble_forecast(ensemble_components, trained_models: dict, horizon: int):
    """
    Task 8 — build the blended forecast for a Weighted Ensemble selection:
    forecast() each component model independently, then combine period by
    period using the inverse-error weights selector.py computed. Every
    component model was fit on the SAME series (all candidates in
    trainer.py are trained on one shared `series`), so their forecast()
    outputs always cover the same horizon starting from the same last
    known period — the period labels are taken from the first component
    and simply reused, they are not independently recomputed.

    Returns (forecast_pairs, warning_or_None). Degrades to whichever
    component(s) actually forecast successfully rather than raising, and
    re-normalises weights across only those — treating a component whose
    forecast() call itself raised (Task 3/7 — a model can validate fine
    and still fail specifically at this later, separate call) exactly the
    same as one that was never trained in the first place.
    """
    component_forecasts = []
    missing = []
    for comp in ensemble_components:
        model_obj = trained_models.get(comp["model"])
        if model_obj is None:
            missing.append(comp["model"])
            continue
        try:
            component_forecasts.append((comp["weight"], model_obj.forecast(horizon)))
        except Exception:
            logger.exception("%s failed during ensemble forecast generation", comp["model"])
            missing.append(comp["model"])

    warning = None
    if missing:
        warning = (
            f"Ensemble component(s) {', '.join(missing)} were not available at forecast "
            f"time — blended using the remaining component(s) only."
        )

    if not component_forecasts:
        # Should not happen (every component name comes from trained_models
        # by construction in selector.py), but degrade safely rather than
        # raising if it ever does.
        return [], "Ensemble could not be assembled — no component model was available to forecast."

    if len(component_forecasts) == 1:
        return component_forecasts[0][1], warning

    periods = [p for p, _ in component_forecasts[0][1]]
    total_weight = sum(w for w, _ in component_forecasts)
    blended = [
        (
            periods[i],
            sum((w / total_weight) * pts[i][1] for w, pts in component_forecasts),
        )
        for i in range(len(periods))
    ]
    return blended, warning


def _explainability(stats, classification) -> dict:
    """Everything Step 7 (Forecast Explainability) needs about the series
    itself, independent of which model ended up selected."""
    return {
        "series_type":               classification.series_type,
        "classification_confidence": classification.confidence,
        "classification_reasons":    classification.reasons,
        "trend_strength":            stats.trend_strength,
        "seasonality_strength":      stats.seasonality_strength,
        "history_length_days":       stats.history_length_days,
        "n_observations":            stats.n_observations,
    }
