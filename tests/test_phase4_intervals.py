"""
test_phase4_intervals.py
-------------------------
Phase 4 test suite: prediction intervals and forecast reliability.

Covers the twenty areas Phase 4 specified: interval generation at both
levels, bound ordering and nesting, non-negativity, horizon-specific vs
pooled behaviour, small-sample and missing-history fallbacks, leakage,
empirical coverage, width/sharpness, varying series lengths, revenue
scales and frequencies, API backward compatibility, and the response
shape the browser consumes.

Stdlib `unittest`, no new dependency:

    python tests/test_phase4_intervals.py
"""

import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

from forecast.backtest import BacktestResult, FoldOutcome, pool_metrics, summarise_folds
from forecast.evaluator import compute_metrics
from forecast.intervals import (
    CALIBRATED,
    CONFORMAL,
    CONFORMAL_RANGE,
    DEFAULT_LEVELS,
    EXTRAPOLATED,
    HORIZON_SPECIFIC,
    LIMITED_DATA,
    MIN_ERRORS_FOR_INTERVAL,
    POOLED,
    UNAVAILABLE,
    _interval_score,
    build_interval_report,
    build_intervals,
    choose_basis,
    collect_relative_errors,
    conformal_indices,
    error_quantiles,
    evaluate_coverage,
    required_errors_for_level,
)
from forecast.results import AutoForecastResult, ForecastPoint


# ── helpers ────────────────────────────────────────────────────────────────

def _fold(index, actuals, predictions, train_size=24):
    """One synthetic walk-forward fold, shaped exactly like the ones
    forecast.backtest produces."""
    periods = [f"2024-{index:02d}-h{i}" for i in range(1, len(actuals) + 1)]
    return FoldOutcome(
        fold=index, train_size=train_size, origin_period=f"origin-{index}",
        periods=periods, actuals=list(map(float, actuals)),
        predictions=list(map(float, predictions)),
        metrics=compute_metrics(np.array(actuals, dtype=float),
                                np.array(predictions, dtype=float)),
    )


def _folds(n, horizon=3, base=1000.0, spread=0.25, seed=0, train_size=24):
    """`n` folds of `horizon` steps each, with controlled relative error."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(1, n + 1):
        predictions = [base] * horizon
        actuals = [base * (1.0 + rng.uniform(-spread, spread)) for _ in range(horizon)]
        out.append(_fold(i, actuals, predictions, train_size=train_size))
    return out


def _backtest(folds, model="Simple Exponential Smoothing"):
    result = BacktestResult(model=model, folds=list(folds))
    if folds:
        result.pooled_metrics = pool_metrics(folds)
        result.fold_stats, result.stability = summarise_folds(result.fold_metric_series("wape"))
    return result


def _points(n=3, value=1000.0, start=1):
    return [{"period": f"2025-{start + i:02d}", "value": value} for i in range(n)]


# ── 1-3: interval generation ───────────────────────────────────────────────

class TestIntervalGeneration(unittest.TestCase):

    def test_intervals_are_generated_for_every_forecast_period(self):
        report = build_interval_report(_points(3), _backtest(_folds(11)))
        self.assertTrue(report.available)
        self.assertEqual(len(report.periods), 3)
        for period in report.periods:
            self.assertIn("80", period.bounds)
            self.assertIn("95", period.bounds)

    def test_80_percent_interval_exists_with_its_level_recorded(self):
        report = build_interval_report(_points(3), _backtest(_folds(11)))
        bound = report.periods[0].bounds["80"]
        self.assertAlmostEqual(bound.level, 0.80, places=6)
        self.assertGreater(bound.width, 0.0)
        self.assertGreater(bound.n_errors, 0)

    def test_95_percent_interval_exists_with_its_level_recorded(self):
        report = build_interval_report(_points(3), _backtest(_folds(11)))
        bound = report.periods[0].bounds["95"]
        self.assertAlmostEqual(bound.level, 0.95, places=6)
        self.assertGreater(bound.width, 0.0)

    def test_report_records_method_and_error_counts(self):
        report = build_interval_report(_points(3), _backtest(_folds(11)))
        self.assertIn("conformal", report.method)
        self.assertEqual(report.n_errors_total, 33)
        self.assertEqual(report.n_errors_by_horizon, {1: 11, 2: 11, 3: 11})


# ── 4-6: ordering, nesting, width ──────────────────────────────────────────

class TestBoundOrdering(unittest.TestCase):

    def test_lower_bound_never_exceeds_the_point_forecast(self):
        report = build_interval_report(_points(3), _backtest(_folds(15, seed=1)))
        for period in report.periods:
            for bound in period.bounds.values():
                self.assertLessEqual(bound.lower, period.point,
                                     f"lower {bound.lower} above point {period.point}")

    def test_point_forecast_never_exceeds_the_upper_bound(self):
        report = build_interval_report(_points(3), _backtest(_folds(15, seed=2)))
        for period in report.periods:
            for bound in period.bounds.values():
                self.assertLessEqual(period.point, bound.upper)

    def test_80_percent_interval_is_never_wider_than_the_95_percent(self):
        for seed in range(6):
            report = build_interval_report(_points(3), _backtest(_folds(14, seed=seed)))
            for period in report.periods:
                self.assertLessEqual(
                    period.bounds["80"].width, period.bounds["95"].width,
                    f"seed {seed}: 80% wider than 95%",
                )

    def test_95_percent_interval_contains_the_80_percent(self):
        report = build_interval_report(_points(3), _backtest(_folds(20, seed=3)))
        for period in report.periods:
            tight, wide = period.bounds["80"], period.bounds["95"]
            self.assertLessEqual(wide.lower, tight.lower)
            self.assertGreaterEqual(wide.upper, tight.upper)

    def test_width_equals_upper_minus_lower(self):
        report = build_interval_report(_points(3), _backtest(_folds(12, seed=4)))
        for period in report.periods:
            for bound in period.bounds.values():
                self.assertAlmostEqual(bound.width, bound.upper - bound.lower, places=6)

    def test_interval_scales_with_the_point_forecast(self):
        folds = _folds(12, seed=5)
        small = build_interval_report(_points(1, value=100.0), _backtest(folds))
        large = build_interval_report(_points(1, value=100000.0), _backtest(folds))
        ratio = large.periods[0].bounds["80"].width / small.periods[0].bounds["80"].width
        self.assertAlmostEqual(ratio, 1000.0, delta=1.0,
                               msg="relative-error intervals must scale with the forecast level")


# ── 7: non-negative revenue ────────────────────────────────────────────────

class TestNonNegativeBounds(unittest.TestCase):

    def test_lower_bound_is_never_negative_even_with_extreme_errors(self):
        # Predictions massively above actuals -> relative errors near -1.
        folds = [_fold(i, [1.0, 1.0, 1.0], [1000.0, 1000.0, 1000.0]) for i in range(1, 13)]
        report = build_interval_report(_points(3, value=1000.0), _backtest(folds))
        for period in report.periods:
            for bound in period.bounds.values():
                self.assertGreaterEqual(bound.lower, 0.0)

    def test_relative_errors_are_bounded_below_by_minus_one(self):
        folds = _folds(15, seed=6)
        by_horizon = collect_relative_errors(folds)
        for errors in by_horizon.values():
            self.assertTrue(all(e >= -1.0 for e in errors),
                            "a non-negative actual over a positive prediction cannot fall below -1")

    def test_zero_actual_gives_a_zero_lower_bound_not_a_negative_one(self):
        folds = [_fold(i, [0.0, 0.0, 0.0], [500.0, 500.0, 500.0]) for i in range(1, 13)]
        report = build_interval_report(_points(3, value=500.0), _backtest(folds))
        self.assertAlmostEqual(report.periods[0].bounds["80"].lower, 0.0, places=6)


# ── 8-9: horizon-specific vs pooled ────────────────────────────────────────

class TestHorizonBasis(unittest.TestCase):

    def test_pooled_is_used_when_horizons_are_too_thin(self):
        report = build_interval_report(_points(3), _backtest(_folds(11, seed=7)))
        self.assertEqual(report.basis, POOLED)
        self.assertTrue(any("pooled" in note.lower() for note in report.notes))

    def test_horizon_specific_is_used_when_each_horizon_has_enough_errors(self):
        needed = required_errors_for_level(0.80, interior=True)
        report = build_interval_report(_points(3), _backtest(_folds(needed + 4, seed=8)))
        self.assertEqual(report.basis, HORIZON_SPECIFIC)

    def test_horizon_specific_widens_with_horizon_when_errors_do(self):
        # Error spread grows with horizon; horizon-specific must reflect it.
        rng = np.random.default_rng(9)
        folds = []
        for i in range(1, 31):
            predictions = [1000.0] * 3
            actuals = [1000.0 * (1.0 + rng.uniform(-0.05 * h, 0.05 * h)) for h in (1, 2, 3)]
            folds.append(_fold(i, actuals, predictions))
        report = build_interval_report(_points(3), _backtest(folds))
        self.assertEqual(report.basis, HORIZON_SPECIFIC)
        widths = [p.bounds["80"].width for p in report.periods]
        self.assertLess(widths[0], widths[2],
                        "horizon 3 should be less certain than horizon 1")

    def test_basis_choice_is_reported_with_a_reason(self):
        basis, explanation = choose_basis({1: [0.1] * 5, 2: [0.1] * 5, 3: [0.1] * 5}, [1, 2, 3])
        self.assertEqual(basis, POOLED)
        self.assertTrue(explanation)

    def test_pooled_gives_every_horizon_the_same_relative_interval(self):
        report = build_interval_report(_points(3), _backtest(_folds(11, seed=10)))
        widths = {round(p.bounds["80"].width, 6) for p in report.periods}
        self.assertEqual(len(widths), 1, "pooled errors imply one shared relative interval")


# ── 10-11: small samples and missing history ───────────────────────────────

class TestFallbacks(unittest.TestCase):

    def test_no_backtest_yields_an_honest_unavailable_report(self):
        report = build_interval_report(_points(3), None)
        self.assertFalse(report.available)
        self.assertEqual(report.calibration_status, UNAVAILABLE)
        self.assertIn("unavailable", (report.unavailable_reason or "").lower())
        self.assertEqual(report.periods, [])

    def test_empty_folds_yield_unavailable_rather_than_fabricated_bounds(self):
        report = build_interval_report(_points(3), _backtest([]))
        self.assertFalse(report.available)
        self.assertEqual(report.calibration_status, UNAVAILABLE)

    def test_too_few_errors_yield_unavailable(self):
        report = build_interval_report(_points(3), _backtest(_folds(1, horizon=1)))
        self.assertFalse(report.available)
        self.assertIn(str(MIN_ERRORS_FOR_INTERVAL), report.unavailable_reason)

    def test_no_forecast_points_yields_unavailable(self):
        report = build_interval_report([], _backtest(_folds(11)))
        self.assertFalse(report.available)

    def test_small_sample_is_flagged_limited_data(self):
        report = build_interval_report(_points(3), _backtest(_folds(11, seed=11)))
        self.assertEqual(report.calibration_status, LIMITED_DATA)

    def test_large_sample_reaches_calibrated_status(self):
        report = build_interval_report(_points(3), _backtest(_folds(40, seed=12)))
        self.assertEqual(report.calibration_status, CALIBRATED)

    def test_95_percent_level_is_flagged_extrapolated_on_small_samples(self):
        report = build_interval_report(_points(3), _backtest(_folds(11, seed=13)))
        self.assertEqual(report.periods[0].bounds["95"].status, EXTRAPOLATED)
        self.assertTrue(any("95%" in note for note in report.notes))

    def test_conformal_indices_reject_unattainable_levels(self):
        self.assertEqual(conformal_indices(11, 0.95), (None, None))
        self.assertNotEqual(conformal_indices(39, 0.95), (None, None))
        self.assertEqual(required_errors_for_level(0.95), 39)
        self.assertEqual(required_errors_for_level(0.80), 9)

    def test_degenerate_quantile_is_labelled_as_the_observed_range(self):
        _low, _high, status = error_quantiles([0.1 * i for i in range(11)], 0.80)
        self.assertEqual(status, CONFORMAL_RANGE)

    def test_interior_quantile_is_labelled_conformal(self):
        _low, _high, status = error_quantiles([0.01 * i for i in range(33)], 0.80)
        self.assertEqual(status, CONFORMAL)

    def test_report_never_returns_nan_or_inf(self):
        report = build_interval_report(_points(3), _backtest(_folds(15, seed=14)))
        for period in report.periods:
            self.assertTrue(np.isfinite(period.point))
            for bound in period.bounds.values():
                self.assertTrue(np.isfinite(bound.lower) and np.isfinite(bound.upper))
                self.assertTrue(np.isfinite(bound.width))


# ── 12: leakage ────────────────────────────────────────────────────────────

class TestNoLeakage(unittest.TestCase):

    def test_errors_come_only_from_the_folds_supplied(self):
        folds = _folds(10, seed=15)
        prefix = collect_relative_errors(folds[:4])
        self.assertEqual(sum(len(v) for v in prefix.values()), 12,
                         "only the four supplied folds may contribute errors")

    def test_coverage_is_prequential_not_in_sample(self):
        # Twelve well-behaved folds, then one wildly wrong fold at the end.
        # An in-sample calculation would build its interval from ALL folds,
        # so the extreme fold would widen the interval enough to cover
        # itself. Prequentially its interval comes only from the calm folds
        # before it, so it must MISS.
        folds = _folds(12, spread=0.05, seed=16)
        folds.append(_fold(13, [50000.0, 50000.0, 50000.0], [1000.0, 1000.0, 1000.0]))
        coverage, _sharpness = evaluate_coverage(folds)
        self.assertLess(coverage["80"]["observed"], 100.0,
                        "the extreme final fold must not be covered by an interval "
                        "built without it")

    def test_a_later_fold_cannot_change_an_earlier_prefix_interval(self):
        folds = _folds(12, seed=17)
        prefix_errors = collect_relative_errors(folds[:8])
        tampered = list(folds[:8]) + [_fold(99, [9e6] * 3, [1.0] * 3)]
        prefix_errors_again = collect_relative_errors(tampered[:8])
        self.assertEqual(prefix_errors, prefix_errors_again)

    def test_intervals_never_read_the_forecast_periods_actuals(self):
        # The forecast points carry only period and value; there is no
        # channel through which a future actual could enter.
        points = _points(3)
        self.assertTrue(all(set(p.keys()) == {"period", "value"} for p in points))
        report = build_interval_report(points, _backtest(_folds(12, seed=18)))
        self.assertTrue(report.available)


# ── 13-14: coverage and sharpness ──────────────────────────────────────────

class TestCoverageAndSharpness(unittest.TestCase):

    def test_coverage_is_measured_for_each_level(self):
        coverage, _ = evaluate_coverage(_folds(15, seed=19))
        for key in ("80", "95"):
            self.assertIn(key, coverage)
            self.assertIsNotNone(coverage[key]["observed"])
            self.assertGreater(coverage[key]["n"], 0)

    def test_coverage_error_is_observed_minus_nominal(self):
        coverage, _ = evaluate_coverage(_folds(15, seed=20))
        for key, entry in coverage.items():
            self.assertAlmostEqual(
                entry["coverage_error_pp"],
                round(entry["observed"] - entry["nominal"], 1), places=6,
            )

    def test_coverage_reports_its_own_sample_size_and_uncertainty(self):
        coverage, _ = evaluate_coverage(_folds(15, seed=21))
        self.assertIn("standard error", coverage["80"]["note"])

    def test_coverage_is_unmeasurable_with_too_few_folds(self):
        coverage, sharpness = evaluate_coverage(_folds(2, seed=22))
        self.assertIsNone(coverage["80"]["observed"])
        self.assertEqual(coverage["80"]["n"], 0)
        self.assertIsNone(sharpness["80"]["mean_width"])

    def test_sharpness_reports_widths_and_interval_score(self):
        _coverage, sharpness = evaluate_coverage(_folds(15, seed=23))
        for key in ("80", "95"):
            self.assertGreater(sharpness[key]["mean_width"], 0.0)
            self.assertGreater(sharpness[key]["median_width"], 0.0)
            self.assertGreater(sharpness[key]["mean_normalized_width"], 0.0)
            self.assertIsNotNone(sharpness[key]["interval_score"])

    def test_interval_score_penalises_misses_and_width(self):
        inside = _interval_score(100.0, 90.0, 110.0, 0.80)
        self.assertAlmostEqual(inside, 20.0, places=6)
        outside = _interval_score(200.0, 90.0, 110.0, 0.80)
        self.assertGreater(outside, inside, "a miss must score worse than a hit")
        wider = _interval_score(100.0, 50.0, 150.0, 0.80)
        self.assertGreater(wider, inside, "a wider interval must not score better for free")

    def test_wider_errors_produce_wider_intervals(self):
        narrow = build_interval_report(_points(1), _backtest(_folds(20, spread=0.05, seed=24)))
        wide = build_interval_report(_points(1), _backtest(_folds(20, spread=0.40, seed=24)))
        self.assertLess(narrow.periods[0].bounds["80"].width,
                        wide.periods[0].bounds["80"].width)


# ── 15-17: series lengths, scales, frequencies ─────────────────────────────

class TestPortability(unittest.TestCase):

    def test_various_fold_counts_all_produce_a_valid_or_honest_result(self):
        for n in (1, 2, 3, 5, 8, 11, 20, 40, 80):
            report = build_interval_report(_points(3), _backtest(_folds(n, seed=n)))
            if report.available:
                for period in report.periods:
                    self.assertLessEqual(period.bounds["80"].lower, period.point)
                    self.assertLessEqual(period.point, period.bounds["95"].upper)
            else:
                self.assertTrue(report.unavailable_reason, f"n={n} gave no reason")

    def test_revenue_scales_from_hundreds_to_crores(self):
        folds_template = _folds(15, seed=25)
        for scale in (1e2, 1e4, 1e6, 1e8):
            folds = [
                _fold(f.fold, [a * scale for a in f.actuals], [p * scale for p in f.predictions])
                for f in folds_template
            ]
            report = build_interval_report(_points(3, value=1000.0 * scale), _backtest(folds))
            self.assertTrue(report.available, f"scale {scale} produced no interval")
            bound = report.periods[0].bounds["80"]
            self.assertGreaterEqual(bound.lower, 0.0)
            self.assertTrue(np.isfinite(bound.upper))

    def test_horizon_of_one_period(self):
        report = build_interval_report(_points(1), _backtest(_folds(12, horizon=1, seed=26)))
        self.assertTrue(report.available)
        self.assertEqual(len(report.periods), 1)

    def test_longer_horizons_are_supported(self):
        report = build_interval_report(_points(6), _backtest(_folds(15, horizon=6, seed=27)))
        self.assertTrue(report.available)
        self.assertEqual(len(report.periods), 6)

    def test_frequency_labels_do_not_affect_the_method(self):
        # Intervals are computed from relative errors, so they are
        # frequency-agnostic by construction — weekly, monthly and
        # quarterly folds of identical shape must give identical bounds.
        folds = _folds(12, seed=28)
        weekly = build_interval_report([{"period": "2025-W01", "value": 1000.0}], _backtest(folds))
        monthly = build_interval_report([{"period": "2025-01", "value": 1000.0}], _backtest(folds))
        quarterly = build_interval_report([{"period": "2025Q1", "value": 1000.0}], _backtest(folds))
        for other in (monthly, quarterly):
            self.assertAlmostEqual(weekly.periods[0].bounds["80"].lower,
                                   other.periods[0].bounds["80"].lower, places=6)

    def test_predictions_at_or_below_zero_are_skipped_not_crashed(self):
        folds = [_fold(i, [100.0, 100.0], [0.0, 100.0]) for i in range(1, 13)]
        by_horizon = collect_relative_errors(folds)
        self.assertNotIn(1, by_horizon, "a zero prediction cannot yield a relative error")
        self.assertEqual(len(by_horizon.get(2, [])), 12)

    def test_missing_periods_in_folds_are_tolerated(self):
        # Folds of differing length, as happens when unobserved periods are
        # excluded from scoring.
        folds = [_fold(1, [100.0], [90.0]), _fold(2, [100.0, 110.0], [90.0, 95.0])]
        folds += _folds(10, seed=29)
        report = build_interval_report(_points(3), _backtest(folds))
        self.assertTrue(report.available)


# ── 18-20: API and UI response compatibility ───────────────────────────────

class TestApiCompatibility(unittest.TestCase):

    LEGACY_RESULT_FIELDS = [
        "horizon", "selected_model", "selection_reason", "evaluation_metrics",
        "forecast_confidence", "model_ranking", "forecast", "alternative_models",
        "explainability", "warnings", "frequency", "forecast_confidence_pct",
        "data_quality_score", "selected_frequency", "frequency_reason",
        "series_density", "zero_period_ratio", "candidate_frequencies",
        "n_modelled_periods", "n_transactions", "requested_horizon",
        "forecast_periods", "forecast_unit", "validation_horizon",
        "horizon_note", "validation", "ensemble_evaluation",
    ]

    def test_every_legacy_result_field_survives(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in self.LEGACY_RESULT_FIELDS:
            self.assertIn(field, payload, f"removing '{field}' would break existing clients")

    def test_new_interval_fields_are_additive_and_default_to_none(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in ("prediction_intervals", "interval_method",
                      "interval_calibration_status", "coverage_80", "coverage_95"):
            self.assertIn(field, payload)
            self.assertIsNone(payload[field])

    def test_forecast_point_without_bounds_keeps_its_original_shape(self):
        payload = ForecastPoint(period="2025-01", value=100.0).to_dict()
        self.assertEqual(payload, {"period": "2025-01", "value": 100.0},
                         "a point with no interval must serialise exactly as before")

    def test_forecast_point_with_bounds_adds_keys_without_removing_any(self):
        point = ForecastPoint(
            period="2025-01", value=100.0, lower_80=80.0, upper_80=130.0,
            lower_95=60.0, upper_95=170.0, interval_width_80=50.0, interval_width_95=110.0,
        )
        payload = point.to_dict()
        self.assertEqual(payload["period"], "2025-01")
        self.assertEqual(payload["value"], 100.0)
        for key in ("lower_80", "upper_80", "lower_95", "upper_95",
                    "interval_width_80", "interval_width_95",
                    "prediction_interval_80", "prediction_interval_95"):
            self.assertIn(key, payload)
        self.assertEqual(payload["prediction_interval_80"], [80.0, 130.0])

    def test_confidence_is_untouched_by_interval_fields(self):
        result = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.59, model_ranking=[], forecast=[], alternative_models=[],
            explainability={}, forecast_confidence_pct=59,
            prediction_intervals={"available": True}, interval_calibration_status=LIMITED_DATA,
        ).to_dict()
        self.assertEqual(result["forecast_confidence"], 0.59)
        self.assertEqual(result["forecast_confidence_pct"], 59)

    def test_report_payload_has_every_key_the_browser_reads(self):
        payload = build_interval_report(_points(3), _backtest(_folds(15, seed=30))).to_dict()
        for key in ("available", "interval_method", "interval_calibration_status",
                    "interval_basis", "periods", "coverage", "sharpness", "notes"):
            self.assertIn(key, payload)
        for entry in payload["periods"]:
            for key in ("period", "point_forecast", "lower_80", "upper_80",
                        "lower_95", "upper_95", "interval_width_80", "interval_width_95",
                        "prediction_interval_80", "prediction_interval_95"):
                self.assertIn(key, entry)
        self.assertIn("observed", payload["coverage"]["80"])
        self.assertIn("n", payload["coverage"]["80"])

    def test_unavailable_payload_carries_a_reason_the_ui_can_display(self):
        payload = build_interval_report(_points(3), None).to_dict()
        self.assertFalse(payload["available"])
        self.assertTrue(payload["unavailable_reason"])
        self.assertEqual(payload["periods"], [])

    def test_payload_is_json_serialisable(self):
        import json
        payload = build_interval_report(_points(3), _backtest(_folds(15, seed=31))).to_dict()
        text = json.dumps(payload)
        self.assertIn("interval_calibration_status", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
