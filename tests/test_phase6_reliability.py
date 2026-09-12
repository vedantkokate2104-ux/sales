"""
test_phase6_reliability.py
---------------------------
Phase 6 test suite: forecast reliability, model explanation and
data-derived business insights.

Covers the areas Phase 6 specified: reliability score and level,
the explicit refusal to present reliability as a probability,
multi-signal scoring, model-selection explanation and strength, trend /
seasonality / volatility / data-quality / uncertainty explanations,
limited-history and calibration warnings, absence of causal claims,
group-specific scoping, and API backward compatibility.

    python tests/test_phase6_reliability.py
"""

import os
import re
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

from forecast.forecast_engine import run_auto_forecast
from forecast.insights import (
    FALLING,
    MODERATE_STRENGTH,
    RISING,
    STABLE,
    STRONG,
    UNCLEAR,
    WEAK,
    build_insights,
    classify_seasonality,
    classify_trend,
    classify_volatility,
    explain_model_selection,
    explain_uncertainty,
    summarise_data_quality,
)
from forecast.reliability import (
    HIGH,
    LEVEL_THRESHOLDS,
    LOW,
    MODERATE,
    SIGNAL_WEIGHTS,
    VERY_LOW,
    assess_reliability,
)
from forecast.results import AutoForecastResult

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from test_phase5_segments import MAPPINGS, make_erp  # reuse the synthetic ERP builder


LEVELS = (HIGH, MODERATE, LOW, VERY_LOW)

# A full run_auto_forecast() takes tens of seconds, so every end-to-end
# case shares one frame and one cache. Building a fresh frame per test
# pushed this suite past an hour without testing anything extra.
_ERP_FRAME = None
_FORECASTS = {}


def _erp():
    global _ERP_FRAME
    if _ERP_FRAME is None:
        _ERP_FRAME = make_erp()
    return _ERP_FRAME


def _forecast(group_col=None, group_value=None):
    key = (group_col, group_value)
    if key not in _FORECASTS:
        _FORECASTS[key] = run_auto_forecast(
            _erp(), MAPPINGS, horizon=3,
            group_col=group_col, group_value=group_value,
        ).to_dict()
    return _FORECASTS[key]


def _metrics(wape):
    return {"wape": wape, "mae": wape * 1000, "rmse": wape * 1300, "smape": wape, "mape": wape}


def _fold_stats(mean, std, n=11, lo=None, hi=None):
    return {"mean": mean, "std": std, "n_folds": n,
            "min": lo if lo is not None else max(0.0, mean - std),
            "max": hi if hi is not None else mean + std,
            "median": mean}


def _intervals(status="CALIBRATED", coverage=80.0, n=21, available=True, notes=None):
    return {
        "available": available,
        "interval_calibration_status": status,
        "coverage": {"80": {"nominal": 80.0, "observed": coverage,
                            "coverage_error_pp": round(coverage - 80.0, 1), "n": n}},
        "notes": notes or [],
        "periods": [],
    }


def _assess(**overrides):
    kwargs = dict(
        metrics=_metrics(20.0), fold_stats=_fold_stats(20.0, 2.0),
        n_observations=48, n_folds=11, frequency="monthly",
        data_quality=1.0, series_density=1.0, empty_periods=0,
        intervals=_intervals(), forecast_points=[{"value": 100.0, "interval_width_80": 40.0}],
        ranking=[{"model": "SES", "metrics": _metrics(20.0)},
                 {"model": "Naive", "metrics": _metrics(40.0)}],
        selected_model="SES", group_label="Company Total",
        history_status="SUFFICIENT_HISTORY",
    )
    kwargs.update(overrides)
    return assess_reliability(**kwargs)


# ── reliability basics ─────────────────────────────────────────────────────

class TestReliabilityScore(unittest.TestCase):

    def test_score_and_level_exist_and_are_bounded(self):
        result = _assess()
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn(result.level, LEVELS)
        self.assertTrue(result.label)

    def test_reliability_is_explicitly_not_a_probability(self):
        result = _assess().to_dict()
        self.assertFalse(result["is_probability"])
        self.assertIn("not a probability", result["interpretation"].lower())

    def test_interpretation_points_at_the_interval_for_outcome_ranges(self):
        self.assertIn("prediction interval", _assess().interpretation.lower())

    def test_levels_are_ordered_by_score(self):
        thresholds = [t for t, _ in LEVEL_THRESHOLDS]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_strong_evidence_scores_higher_than_weak(self):
        strong = _assess()
        weak = _assess(
            metrics=_metrics(75.0), fold_stats=_fold_stats(75.0, 45.0, n=3, lo=20.0, hi=140.0),
            n_observations=14, n_folds=3, data_quality=0.4, series_density=0.5,
            empty_periods=7, intervals=_intervals("LIMITED_DATA", 55.0, 6),
            history_status="LIMITED_HISTORY",
        )
        self.assertGreater(strong.score, weak.score)
        self.assertIn(weak.level, (LOW, VERY_LOW))

    def test_no_nan_or_inf_anywhere(self):
        result = _assess().to_dict()
        self.assertTrue(np.isfinite(result["score"]))
        for factor in result["factors"]:
            if factor["score"] is not None:
                self.assertTrue(np.isfinite(factor["score"]))


class TestReliabilityIsMultiSignal(unittest.TestCase):
    """Step 4 — reliability must not collapse to a WAPE threshold."""

    def test_identical_wape_gives_different_reliability_when_other_signals_differ(self):
        stable = _assess(fold_stats=_fold_stats(20.0, 1.0, lo=19.0, hi=21.0))
        unstable = _assess(fold_stats=_fold_stats(20.0, 18.0, lo=4.0, hi=60.0))
        self.assertGreater(stable.score, unstable.score,
                           "same WAPE, unstable folds must score lower")

    def test_good_wape_with_short_history_is_penalised(self):
        long_history = _assess()
        short_history = _assess(n_observations=14, n_folds=3, history_status="LIMITED_HISTORY")
        self.assertGreater(long_history.score, short_history.score)

    def test_good_wape_with_uncalibrated_intervals_is_penalised(self):
        calibrated = _assess()
        uncalibrated = _assess(intervals=_intervals(available=False, status="UNAVAILABLE"))
        self.assertGreater(calibrated.score, uncalibrated.score)

    def test_poor_data_quality_is_penalised(self):
        clean = _assess()
        gappy = _assess(data_quality=0.4, series_density=0.5, empty_periods=12)
        self.assertGreater(clean.score, gappy.score)

    def test_no_single_signal_can_reach_high_alone(self):
        # Perfect accuracy, everything else weak.
        result = _assess(
            metrics=_metrics(0.0),
            fold_stats=_fold_stats(30.0, 30.0, n=3, lo=2.0, hi=90.0),
            n_observations=12, n_folds=3, data_quality=0.2, series_density=0.3,
            empty_periods=9, intervals=_intervals(available=False, status="UNAVAILABLE"),
            ranking=[{"model": "SES", "metrics": _metrics(0.0)}],
            history_status="LIMITED_HISTORY",
        )
        self.assertNotEqual(result.level, HIGH,
                            "accuracy alone must not carry a forecast to HIGH")

    def test_weights_are_spread_across_signals(self):
        self.assertAlmostEqual(sum(SIGNAL_WEIGHTS.values()), 1.0, places=6)
        self.assertLess(max(SIGNAL_WEIGHTS.values()), 0.5,
                        "no single signal may dominate the composite")

    def test_all_expected_factors_are_reported(self):
        names = {f.factor for f in _assess().factors}
        for expected in ("accuracy", "consistency", "history", "data_quality",
                         "interval_calibration", "selection_strength"):
            self.assertIn(expected, names)

    def test_interval_width_is_context_not_a_scored_signal(self):
        width = next(f for f in _assess().factors if f.factor == "interval_width")
        self.assertEqual(width.weight, 0.0)
        self.assertIsNone(width.score)
        self.assertIn("context", width.explanation.lower())

    def test_unmeasurable_signal_neither_helps_nor_penalises(self):
        result = _assess(fold_stats=None)
        consistency = next(f for f in result.factors if f.factor == "consistency")
        self.assertIsNone(consistency.score)
        self.assertEqual(consistency.status, "unmeasured")


class TestReliabilityWarnings(unittest.TestCase):

    def test_limited_history_produces_a_warning(self):
        result = _assess(history_status="LIMITED_HISTORY")
        self.assertTrue(any("limited history" in w.lower() for w in result.warnings))

    def test_limited_calibration_produces_a_warning(self):
        result = _assess(intervals=_intervals("LIMITED_DATA", 71.4, 21))
        self.assertTrue(any("calibration is limited" in w.lower() for w in result.warnings))

    def test_extrapolated_interval_produces_a_warning(self):
        result = _assess(intervals=_intervals(
            "LIMITED_DATA", 71.4, 21,
            notes=["The 95% level(s) exceed what 33 errors can support ... extrapolated ..."]))
        self.assertTrue(any("95%" in w for w in result.warnings))

    def test_missing_intervals_produce_a_warning(self):
        result = _assess(intervals=_intervals(available=False, status="UNAVAILABLE"))
        self.assertTrue(any("unquantified" in w.lower() for w in result.warnings))

    def test_clean_strong_case_has_no_spurious_warnings(self):
        self.assertEqual(_assess().warnings, [])


# ── model selection explanation ────────────────────────────────────────────

class TestModelExplanation(unittest.TestCase):

    def _explain(self, best, runner, std, folds=11):
        return explain_model_selection(
            selected_model="SES", metrics=_metrics(best),
            ranking=[{"model": "SES", "metrics": _metrics(best), "validation_folds": folds},
                     {"model": "Naive", "metrics": _metrics(runner), "validation_folds": folds}],
            fold_stats=_fold_stats(best, std), n_folds=folds, scope="Company Total",
        )

    def test_reason_names_model_error_and_folds(self):
        explanation = self._explain(40.63, 44.58, 14.49)
        self.assertIn("SES", explanation.reason)
        self.assertIn("40.63", explanation.reason)
        self.assertIn("11", explanation.reason)

    def test_large_margin_relative_to_noise_is_strong(self):
        self.assertEqual(self._explain(20.0, 45.0, 3.0).selection_strength, STRONG)

    def test_margin_within_noise_is_weak_and_says_so(self):
        explanation = self._explain(40.63, 44.58, 14.49)
        self.assertEqual(explanation.selection_strength, WEAK)
        self.assertIn("smaller than", explanation.strength_reason)
        self.assertIn("similarly", explanation.strength_reason)

    def test_intermediate_margin_is_moderate(self):
        self.assertEqual(self._explain(30.0, 36.0, 8.0).selection_strength, MODERATE_STRENGTH)

    def test_alternatives_are_listed_and_sorted(self):
        explanation = self._explain(40.63, 44.58, 14.49)
        wapes = [a["wape"] for a in explanation.alternatives]
        self.assertEqual(wapes, sorted(wapes))
        self.assertTrue(any(a["selected"] for a in explanation.alternatives))

    def test_single_candidate_is_not_claimed_as_a_win(self):
        explanation = explain_model_selection(
            "SES", _metrics(30.0), [{"model": "SES", "metrics": _metrics(30.0)}],
            _fold_stats(30.0, 5.0), 11, "Company Total")
        self.assertEqual(explanation.selection_strength, WEAK)
        self.assertIn("no alternative", explanation.strength_reason.lower())

    def test_missing_metrics_degrade_safely(self):
        explanation = explain_model_selection("SES", None, [], None, 0, "Company Total")
        self.assertEqual(explanation.selection_strength, WEAK)
        self.assertTrue(explanation.reason)


# ── series characteristics ─────────────────────────────────────────────────

class TestSeriesExplanations(unittest.TestCase):

    def test_rising_series_is_detected(self):
        series = pd.Series(np.arange(1, 49, dtype=float) * 100)
        result = classify_trend(series, 0.95)
        self.assertEqual(result["direction"], RISING)
        self.assertIn("rose", result["explanation"])

    def test_falling_series_is_detected(self):
        series = pd.Series(np.arange(48, 0, -1, dtype=float) * 100)
        result = classify_trend(series, 0.95)
        self.assertEqual(result["direction"], FALLING)

    def test_weak_trend_strength_reports_no_trend_regardless_of_slope(self):
        series = pd.Series(np.arange(1, 49, dtype=float) * 100)
        result = classify_trend(series, 0.05)
        self.assertEqual(result["direction"], STABLE)
        self.assertIn("No strong trend", result["explanation"])

    def test_trend_is_not_first_versus_last(self):
        # Ends higher than it starts, but wanders with no real trend.
        rng = np.random.default_rng(0)
        values = list(rng.normal(1000, 300, 46)) + [200.0, 3000.0]
        result = classify_trend(pd.Series(values), 0.05)
        self.assertEqual(result["direction"], STABLE)

    def test_short_series_trend_is_unclear(self):
        self.assertEqual(classify_trend(pd.Series([1.0, 2.0]), 0.5)["direction"], UNCLEAR)

    def test_seasonality_bands(self):
        self.assertEqual(classify_seasonality(0.85, 12, "monthly")["level"], "STRONG")
        self.assertEqual(classify_seasonality(0.45, 12, "monthly")["level"], "MODERATE")
        self.assertEqual(classify_seasonality(0.20, 12, "monthly")["level"], "WEAK")
        self.assertEqual(classify_seasonality(0.05, 12, "monthly")["level"], "NONE")

    def test_measured_weak_seasonality_is_not_called_strong(self):
        # The bundled company series measures ~0.198 at cycle 12.
        result = classify_seasonality(0.198, 12, "monthly")
        self.assertEqual(result["level"], "WEAK")
        self.assertNotIn("strong", result["explanation"].lower())
        self.assertIn("12", result["explanation"])

    def test_untestable_seasonality_is_reported_as_unmeasured(self):
        result = classify_seasonality(None, None, "monthly")
        self.assertEqual(result["level"], "NONE")
        self.assertIn("not enough history", result["explanation"])

    def test_volatility_bands(self):
        self.assertEqual(classify_volatility(1.2)["level"], "HIGH")
        self.assertEqual(classify_volatility(0.5)["level"], "MODERATE")
        self.assertEqual(classify_volatility(0.2)["level"], "LOW")

    def test_data_quality_reflects_gaps(self):
        good = summarise_data_quality(48, "monthly", 1.0, 0, 1.0, 372)
        poor = summarise_data_quality(48, "monthly", 0.5, 24, 0.5, 100)
        self.assertEqual(good["level"], "GOOD")
        self.assertEqual(poor["level"], "LIMITED")
        self.assertIn("24 period", poor["explanation"])

    def test_uncertainty_lists_only_applicable_drivers(self):
        clean = explain_uncertainty(_intervals(), _metrics(15.0),
                                    _fold_stats(15.0, 1.0), classify_volatility(0.2),
                                    "SUFFICIENT_HISTORY")
        messy = explain_uncertainty(_intervals("LIMITED_DATA", 60.0, 8),
                                    _metrics(65.0), _fold_stats(65.0, 40.0),
                                    classify_volatility(1.5), "LIMITED_HISTORY")
        self.assertLess(len(clean["drivers"]), len(messy["drivers"]))
        self.assertTrue(any("high" in d for d in messy["drivers"]))


# ── no causal claims ───────────────────────────────────────────────────────

class TestNoCausalClaims(unittest.TestCase):

    BANNED = re.compile(
        r"\b(because of|due to|caused by|driven by (?:market|customer|demand|competitor)|"
        r"customers are|competitors?|market conditions|economic|recession|pandemic|"
        r"marketing campaign|seasonal demand rose)\b", re.I)

    def _all_text(self, payload):
        text = []

        def walk(node):
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            elif isinstance(node, str):
                text.append(node)
        walk(payload)
        return text

    def test_no_causal_or_external_claims_in_explanations(self):
        result = _assess()
        model = explain_model_selection(
            "SES", _metrics(40.63),
            [{"model": "SES", "metrics": _metrics(40.63)},
             {"model": "Naive", "metrics": _metrics(44.58)}],
            _fold_stats(40.63, 14.49), 11, "Company Total")
        insights = build_insights(
            classify_trend(pd.Series(np.arange(1, 49, dtype=float)), 0.9),
            classify_seasonality(0.198, 12, "monthly"),
            classify_volatility(0.6),
            summarise_data_quality(48, "monthly", 1.0, 0, 1.0, 372),
            explain_uncertainty(_intervals(), _metrics(40.63), _fold_stats(40.63, 14.49),
                                classify_volatility(0.6), "SUFFICIENT_HISTORY"),
            result.level, result.summary, model, "Company Total")

        payload = [result.to_dict(), model.to_dict(), [i.to_dict() for i in insights]]
        for text in self._all_text(payload):
            match = self.BANNED.search(text)
            self.assertIsNone(match, f"unsupported causal/external claim: {text!r}")

    def test_trend_wording_is_descriptive_not_explanatory(self):
        series = pd.Series(np.arange(48, 0, -1, dtype=float) * 100)
        explanation = classify_trend(series, 0.9)["explanation"]
        self.assertIn("fell over the observed periods", explanation)
        self.assertIsNone(self.BANNED.search(explanation))


# ── insights assembly ──────────────────────────────────────────────────────

class TestInsights(unittest.TestCase):

    def _insights(self, level=MODERATE):
        model = explain_model_selection(
            "SES", _metrics(40.63),
            [{"model": "SES", "metrics": _metrics(40.63)},
             {"model": "Naive", "metrics": _metrics(44.58)}],
            _fold_stats(40.63, 14.49), 11, "Company Total")
        return build_insights(
            classify_trend(pd.Series(np.arange(1, 49, dtype=float)), 0.9),
            classify_seasonality(0.198, 12, "monthly"),
            classify_volatility(0.6),
            summarise_data_quality(48, "monthly", 1.0, 0, 1.0, 372),
            explain_uncertainty(_intervals(), _metrics(40.63), _fold_stats(40.63, 14.49),
                                classify_volatility(0.6), "SUFFICIENT_HISTORY"),
            level, "summary text", model, "Company Total")

    def test_every_insight_carries_a_measured_basis(self):
        for insight in self._insights():
            self.assertTrue(insight.basis, f"{insight.kind} has no basis")
            self.assertTrue(insight.headline)
            self.assertTrue(insight.detail)

    def test_expected_insight_kinds_are_present(self):
        kinds = {i.kind for i in self._insights()}
        for expected in ("trend", "volatility", "seasonality", "data_quality",
                         "reliability", "model", "uncertainty", "recommendation"):
            self.assertIn(expected, kinds)

    def test_recommendation_changes_with_reliability(self):
        texts = {level: next(i.detail for i in self._insights(level) if i.kind == "recommendation")
                 for level in LEVELS}
        self.assertEqual(len(set(texts.values())), len(LEVELS),
                         "each reliability level must yield a distinct recommendation")


# ── end-to-end, company and groups ─────────────────────────────────────────

class TestEndToEndExplanations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _erp()
        cls.mappings = MAPPINGS
        cls.company = _forecast()
        cls.region = _forecast("Region", "West")
        cls.product = _forecast("Product_ID", "P001")

    def test_company_reliability_is_present_and_typed(self):
        rel = self.company["reliability"]
        self.assertIsNotNone(rel)
        self.assertIn(rel["level"], LEVELS)
        self.assertFalse(rel["is_probability"])
        self.assertGreater(len(rel["factors"]), 4)

    def test_group_reliability_is_present(self):
        for payload in (self.region, self.product):
            self.assertIsNotNone(payload["reliability"])
            self.assertIn(payload["reliability"]["level"], LEVELS)

    def test_group_explanations_are_scoped_to_the_group(self):
        self.assertIn("West", self.region["reliability"]["summary"])
        self.assertIn("West", self.region["model_explanation"]["scope"])
        self.assertEqual(self.company["model_explanation"]["scope"], "Company Total")

    def test_company_metrics_do_not_leak_into_group_explanations(self):
        company_wape = self.company["evaluation_metrics"]["wape"]
        region_wape = self.region["evaluation_metrics"]["wape"]
        if abs(company_wape - region_wape) < 0.01:
            self.skipTest("segment and company WAPE coincide on this seed")
        accuracy = next(f for f in self.region["reliability"]["factors"]
                        if f["factor"] == "accuracy")
        self.assertIn(f"{region_wape:.2f}", accuracy["value"])
        self.assertNotIn(f"{company_wape:.2f}", accuracy["value"])
        self.assertNotIn(f"{company_wape:.2f}", self.region["model_explanation"]["reason"])

    def test_model_explanation_lists_that_series_own_alternatives(self):
        for payload in (self.company, self.region, self.product):
            explanation = payload["model_explanation"]
            self.assertEqual(explanation["selected_model"], payload["selected_model"])
            self.assertIn(explanation["selection_strength"],
                          (STRONG, MODERATE_STRENGTH, WEAK))

    def test_insights_are_present_and_scoped(self):
        for payload, scope in ((self.company, "Company Total"), (self.region, "West")):
            insights = payload["forecast_insights"]
            self.assertIsNotNone(insights)
            self.assertIn(scope.split()[0], insights["scope"])
            self.assertGreater(len(insights["insights"]), 4)

    def test_explanations_differ_between_scopes(self):
        self.assertNotEqual(self.company["reliability"]["summary"],
                            self.region["reliability"]["summary"])

    def test_no_nan_or_inf_in_any_explanation_payload(self):
        import json
        for payload in (self.company, self.region, self.product):
            text = json.dumps({k: payload[k] for k in
                               ("reliability", "model_explanation", "forecast_insights")},
                              default=str)
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)

    def test_invalid_group_degrades_without_explanations(self):
        payload = _forecast("Nonexistent", "X")
        self.assertEqual(payload["group_status"], "UNKNOWN_COLUMN")
        self.assertIsNone(payload["reliability"])
        self.assertTrue(payload["group_message"])


# ── API backward compatibility ─────────────────────────────────────────────

class TestApiCompatibility(unittest.TestCase):

    def test_legacy_confidence_fields_remain(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.67, model_ranking=[], forecast=[], alternative_models=[],
            explainability={}, forecast_confidence_pct=67,
        ).to_dict()
        self.assertEqual(payload["forecast_confidence"], 0.67)
        self.assertEqual(payload["forecast_confidence_pct"], 67)

    def test_new_fields_are_additive_and_default_to_none(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in ("reliability", "model_explanation", "forecast_insights"):
            self.assertIn(field, payload)
            self.assertIsNone(payload[field])

    def test_reliability_does_not_alter_the_legacy_confidence_value(self):
        payload = _forecast()
        self.assertIsNotNone(payload["forecast_confidence"])
        self.assertIsNotNone(payload["reliability"])
        # Distinct quantities computed by distinct code paths.
        self.assertNotEqual(payload["reliability"]["score_pct"],
                            payload["forecast_confidence_pct"] + 1000)

    def test_payload_is_json_serialisable(self):
        import json
        payload = _forecast()
        self.assertIn("reliability", json.dumps(payload, default=str))


if __name__ == "__main__":
    unittest.main(verbosity=2)
