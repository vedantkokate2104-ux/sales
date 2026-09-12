"""
test_phase3_features.py
------------------------
Phase 3 test suite: frequency-aware, leakage-safe feature engineering and
the availability contract.

Covers the fourteen areas Phase 3 specified: frequency-aware lags,
leakage-safe rolling features, monthly / weekly / daily / quarterly
generation, missing optional business columns, future-feature
availability, no target leakage, short history, constant features, ML
overfitting detection, baseline comparison, and ensemble rejection.

Stdlib `unittest`, no new dependency:

    python tests/test_phase3_features.py
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

from forecast.backtest import backtest_model, evaluate_ensemble, plan_validation
from forecast.experiments import (
    MIN_MEANINGFUL_IMPROVEMENT_PP,
    OVERFIT_RATIO_THRESHOLD,
    run_experiments,
)
from forecast.features import (
    FORECAST_REQUIRED,
    HISTORICAL_ONLY,
    KNOWN_FUTURE,
    LAGS_BY_FREQUENCY,
    SCENARIO_INPUT,
    UNKNOWN,
    USABLE_AVAILABILITY,
    FeatureConfig,
    assert_no_unavailable_features,
    audit_business_columns,
    build_feature_frame,
    classify_business_column,
    resolve_lags,
    resolve_rolling_windows,
    select_features,
)
from forecast.grain import build_forecast_series
from forecast.models.tabular import make_tabular_model, ridge_factory, xgboost_factory


def _series(n, freq="M", start="2021-01", seed=0, base=1000.0, noise=0.1):
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, periods=n, freq=freq)
    return pd.Series(np.abs(base * (1.0 + rng.normal(0, noise, n))), index=index)


def _construction(series):
    return build_forecast_series(
        pd.Series(series.index.to_timestamp()),
        pd.Series(series.to_numpy()),
        {"M": "monthly", "W-SUN": "weekly", "Q-DEC": "quarterly", "D": "daily"}.get(
            series.index.freqstr, "monthly"
        ),
    )


# ── 1: frequency-aware lags ────────────────────────────────────────────────

class TestFrequencyAwareLags(unittest.TestCase):

    def test_each_frequency_has_its_own_lag_vocabulary(self):
        self.assertEqual(resolve_lags("monthly", 48), [1, 2, 3, 6, 12])
        self.assertEqual(resolve_lags("weekly", 209), [1, 2, 4, 13, 52])
        self.assertEqual(resolve_lags("daily", 400), [1, 7, 14, 28, 365])
        self.assertEqual(resolve_lags("quarterly", 32), [1, 2, 4])

    def test_monthly_lags_are_periods_not_days(self):
        lags = resolve_lags("monthly", 48)
        for day_lag in (30, 90, 365):
            self.assertNotIn(day_lag, lags,
                             f"lag_{day_lag} is a day count and is meaningless at monthly grain")

    def test_lags_are_dropped_when_history_cannot_support_them(self):
        short = resolve_lags("monthly", 14)
        self.assertNotIn(12, short, "lag_12 would leave too few training rows in a 14-period series")
        self.assertIn(1, short)

    def test_lag_never_empties_the_training_set(self):
        for n in (10, 12, 14, 20, 48):
            lags = resolve_lags("monthly", n)
            self.assertTrue(lags, "at least one lag must always be produced")
            X, y, _ = build_feature_frame(_series(n), "monthly", FeatureConfig(use_lags=True))
            self.assertGreater(len(X), 0, f"n={n} produced an empty training set")
            self.assertEqual(len(X), len(y))

    def test_rolling_windows_are_also_frequency_aware(self):
        self.assertEqual(resolve_rolling_windows("monthly", 48), [3, 6])
        self.assertEqual(resolve_rolling_windows("weekly", 209), [4, 13])
        self.assertEqual(resolve_rolling_windows("quarterly", 32), [2, 4])


# ── 2 & 9: leakage safety ──────────────────────────────────────────────────

class TestLeakageSafety(unittest.TestCase):

    def test_no_feature_ever_equals_its_own_target(self):
        series = _series(48, seed=1)
        X, y, _ = build_feature_frame(
            series, "monthly", FeatureConfig(True, True, True, True),
        )
        for period in X.index:
            actual = float(series.loc[period])
            row = X.loc[period]
            matches = [c for c in X.columns if abs(float(row[c]) - actual) < 1e-9]
            self.assertEqual(matches, [], f"{period}: feature(s) {matches} leak the target")

    def test_rolling_features_use_shifted_history(self):
        # rolling mean at t must equal mean of t-3..t-1, never include t.
        series = pd.Series(
            np.arange(1.0, 21.0), index=pd.period_range("2021-01", periods=20, freq="M"),
        )
        X, _y, _ = build_feature_frame(series, "monthly", FeatureConfig(use_rolling=True))
        period = X.index[5]
        position = list(series.index).index(period)
        expected = series.iloc[position - 3:position].mean()
        self.assertAlmostEqual(float(X.loc[period, "roll_mean_3"]), float(expected), places=6)
        self.assertNotAlmostEqual(
            float(X.loc[period, "roll_mean_3"]),
            float(series.iloc[position - 2:position + 1].mean()), places=6,
        )

    def test_expanding_features_exclude_the_current_period(self):
        series = pd.Series(
            np.arange(1.0, 21.0), index=pd.period_range("2021-01", periods=20, freq="M"),
        )
        X, _y, _ = build_feature_frame(series, "monthly", FeatureConfig(use_rolling=True))
        period = X.index[4]
        position = list(series.index).index(period)
        self.assertAlmostEqual(
            float(X.loc[period, "expanding_mean"]),
            float(series.iloc[:position].mean()), places=6,
        )

    def test_changing_the_future_cannot_change_past_feature_rows(self):
        series = _series(48, seed=2)
        tampered = series.copy()
        tampered.iloc[30:] = tampered.iloc[30:] * 1000.0

        config = FeatureConfig(True, True, True, True)
        X_base, _y, _ = build_feature_frame(series, "monthly", config)
        X_tampered, _y2, _ = build_feature_frame(tampered, "monthly", config)

        shared = [p for p in X_base.index if p in X_tampered.index and list(series.index).index(p) < 30]
        self.assertGreater(len(shared), 0)
        np.testing.assert_allclose(
            X_base.loc[shared].to_numpy(), X_tampered.loc[shared].to_numpy(),
            err_msg="a feature row before the tampered region changed",
        )

    def test_lag_features_match_their_definition_exactly(self):
        series = pd.Series(
            np.arange(1.0, 31.0), index=pd.period_range("2021-01", periods=30, freq="M"),
        )
        X, _y, _ = build_feature_frame(series, "monthly", FeatureConfig(use_lags=True))
        period = X.index[3]
        position = list(series.index).index(period)
        for lag in resolve_lags("monthly", 30):
            self.assertAlmostEqual(
                float(X.loc[period, f"lag_{lag}"]),
                float(series.iloc[position - lag]), places=6,
            )


# ── 3-6: per-frequency generation ──────────────────────────────────────────

class TestPerFrequencyGeneration(unittest.TestCase):

    def _build(self, n, freq, frequency, start):
        series = _series(n, freq=freq, start=start)
        return build_feature_frame(series, frequency, FeatureConfig(True, True, True, True))

    def test_monthly_feature_generation(self):
        X, y, specs = self._build(48, "M", "monthly", "2021-01")
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertIn("cal_month", X.columns)
        self.assertNotIn("cal_dayofweek", X.columns, "day-of-week is constant after monthly aggregation")
        self.assertLess(X.shape[1] / len(X), 1.0, "feature count must stay below the row count")

    def test_weekly_feature_generation(self):
        X, _y, _ = self._build(160, "W", "weekly", "2021-01-03")
        self.assertIn("lag_52", X.columns)
        self.assertIn("cal_weekofyear", X.columns)

    def test_daily_feature_generation(self):
        X, _y, _ = self._build(500, "D", "daily", "2021-01-01")
        self.assertIn("lag_7", X.columns)
        self.assertIn("cal_dayofweek", X.columns)

    def test_quarterly_feature_generation(self):
        X, _y, _ = self._build(32, "Q", "quarterly", "2016Q1")
        self.assertIn("lag_4", X.columns)
        self.assertIn("cal_quarter", X.columns)
        self.assertNotIn("cal_month", X.columns, "month is redundant once aggregated to quarters")

    def test_calendar_features_are_grain_appropriate(self):
        from forecast.features import CALENDAR_BY_FREQUENCY
        for banned in ("dayofweek", "dayofmonth"):
            self.assertNotIn(banned, CALENDAR_BY_FREQUENCY["monthly"])
            self.assertNotIn(banned, CALENDAR_BY_FREQUENCY["quarterly"])


# ── 7 & 8: business columns and availability contract ──────────────────────

class TestBusinessFeatureAvailability(unittest.TestCase):

    def test_target_restatements_are_refused(self):
        for name in ("Net Sales Value", "Gross Amount", "Gst Amount", "Total Invoice Value"):
            availability, reason = classify_business_column(name)
            self.assertEqual(availability, HISTORICAL_ONLY, f"{name} must not be usable")
            self.assertNotIn(availability, USABLE_AVAILABILITY)
            self.assertIn("leakage", reason)

    def test_outcome_drivers_require_their_own_forecast(self):
        for name in ("Quantity Sold", "Unit Price", "Margin Percentage"):
            availability, _reason = classify_business_column(name)
            self.assertEqual(availability, FORECAST_REQUIRED)
            self.assertNotIn(availability, USABLE_AVAILABILITY)

    def test_planned_levers_are_scenario_inputs(self):
        availability, _ = classify_business_column("Discount Percentage")
        self.assertEqual(availability, SCENARIO_INPUT)

    def test_calendar_is_known_future(self):
        availability, _ = classify_business_column("Invoice Date")
        self.assertEqual(availability, KNOWN_FUTURE)
        self.assertIn(availability, USABLE_AVAILABILITY)

    def test_unrecognised_columns_default_to_unusable(self):
        availability, _ = classify_business_column("Some Column Nobody Mapped")
        self.assertEqual(availability, UNKNOWN)
        self.assertNotIn(availability, USABLE_AVAILABILITY)

    def test_missing_optional_business_columns_are_handled(self):
        # No mappings at all, and mappings lacking the optional drivers.
        self.assertEqual(audit_business_columns([]), [])
        self.assertEqual(audit_business_columns(None), [])
        minimal = [
            {"original_column": "d", "business_name": "Invoice Date"},
            {"original_column": "v", "business_name": "Net Sales Value"},
        ]
        audited = audit_business_columns(minimal)
        self.assertEqual(len(audited), 2)
        self.assertTrue(any(a["usable"] for a in audited))

    def test_pipeline_refuses_unavailable_features(self):
        from forecast.features import FeatureSpec
        bad = [FeatureSpec("future_marketing_spend", "business_driver",
                           FORECAST_REQUIRED, "monthly sum", True)]
        with self.assertRaises(ValueError) as ctx:
            assert_no_unavailable_features(bad)
        self.assertIn("future_marketing_spend", str(ctx.exception))

    def test_generated_features_all_satisfy_the_contract(self):
        _X, _y, specs = build_feature_frame(
            _series(48), "monthly", FeatureConfig(True, True, True, True),
        )
        assert_no_unavailable_features(specs)  # must not raise
        self.assertTrue(all(s.leakage_safe for s in specs))


# ── 10 & 11: short history and constant features ───────────────────────────

class TestShortHistoryAndConstantFeatures(unittest.TestCase):

    def test_short_series_still_produces_usable_features(self):
        X, y, _ = build_feature_frame(_series(12), "monthly", FeatureConfig(use_lags=True))
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertLessEqual(X.shape[1], 4, "a 12-period series must not get long lags")

    def test_very_short_series_degrades_without_crashing(self):
        X, y, specs = build_feature_frame(_series(4), "monthly", FeatureConfig(True, True, True, True))
        self.assertEqual(len(X), len(y))
        self.assertIsInstance(specs, list)

    def test_constant_features_are_dropped(self):
        X = pd.DataFrame({
            "useful": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [7.0] * 5,
        })
        reduced, _specs, dropped = select_features(X, [])
        self.assertNotIn("constant", reduced.columns)
        self.assertIn("constant", dropped)
        self.assertIn("constant", dropped["constant"])

    def test_duplicate_features_are_dropped(self):
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        independent = np.array([4.0, 1.0, 6.0, 2.0, 9.0, 3.0])
        X = pd.DataFrame({"a": base, "b": base * 2.0, "c": independent})
        reduced, _specs, dropped = select_features(X, [])
        self.assertEqual(len(reduced.columns), 2, "one of the perfectly correlated pair must go")
        self.assertIn("b", dropped)
        self.assertIn("c", reduced.columns, "an independent feature must survive")
        self.assertTrue(any("correlated" in reason for reason in dropped.values()))

    def test_perfectly_anticorrelated_features_are_also_redundant(self):
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        X = pd.DataFrame({"a": base, "mirror": base[::-1]})
        reduced, _specs, dropped = select_features(X, [])
        self.assertEqual(len(reduced.columns), 1,
                         "a perfectly anti-correlated column carries no additional information")
        self.assertIn("mirror", dropped)

    def test_feature_selection_keeps_informative_columns(self):
        X = pd.DataFrame({
            "a": [1.0, 5.0, 2.0, 8.0, 3.0],
            "b": [9.0, 1.0, 7.0, 2.0, 6.0],
        })
        reduced, _specs, dropped = select_features(X, [])
        self.assertEqual(set(reduced.columns), {"a", "b"})
        self.assertEqual(dropped, {})


# ── 12: overfitting detection ──────────────────────────────────────────────

class TestOverfittingDetection(unittest.TestCase):

    def test_tabular_model_reports_in_sample_training_error(self):
        series = _series(48, seed=3)
        cls = make_tabular_model("XGB test", xgboost_factory, FeatureConfig(use_rolling=True))
        model = cls().fit(series, frequency="monthly")
        self.assertIsNotNone(model.training_error_)
        self.assertGreaterEqual(model.training_error_, 0.0)

    def test_high_capacity_model_shows_a_train_validation_gap(self):
        series = _series(48, seed=4)
        construction = _construction(series)
        report = run_experiments(
            construction, horizon=3, include_classical=True,
            tabular_specs=[{
                "experiment": "overfit probe",
                "name": "XGBoost (deep)",
                "estimator": "XGBoost",
                "config": FeatureConfig(use_lags=True, use_rolling=True),
                "hyperparams": {"max_depth": 6, "n_estimators": 400,
                                "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 0.0},
            }],
        )
        row = next(r for r in report.results if r.model == "XGBoost (deep)")
        self.assertIsNotNone(row.training_error)
        self.assertIsNotNone(row.validation_error)
        self.assertGreater(row.validation_error, row.training_error,
                           "an unregularised tree on 48 periods must fit training better than validation")
        self.assertIsNotNone(row.overfit_ratio)

    def test_overfit_threshold_is_applied_to_notes(self):
        self.assertGreater(OVERFIT_RATIO_THRESHOLD, 1.0)


# ── 13 & 14: baseline comparison and ensemble rejection ────────────────────

class TestBaselineComparison(unittest.TestCase):

    def test_baseline_is_identified_and_protocol_is_shared(self):
        series = _series(48, seed=5)
        report = run_experiments(_construction(series), horizon=3, tabular_specs=[])
        baseline = next(r for r in report.results if r.status == "BASELINE")
        self.assertEqual(baseline.model, "Simple Exponential Smoothing")
        self.assertIsNotNone(report.baseline_wape)
        validated = [r for r in report.results if r.n_folds]
        for r in validated:
            self.assertEqual(r.n_folds, report.plan.n_folds,
                             "every candidate must be scored on the same fold plan")

    def test_candidates_worse_than_baseline_are_rejected(self):
        series = _series(48, seed=6)
        report = run_experiments(_construction(series), horizon=3, tabular_specs=[])
        for r in report.results:
            if r.status in ("BASELINE", "SKIPPED", "FAILED") or r.wape is None:
                continue
            if r.wape > report.baseline_wape:
                self.assertEqual(r.status, "REJECTED")

    def test_marginal_improvement_is_not_accepted(self):
        series = _series(48, seed=7)
        report = run_experiments(_construction(series), horizon=3, tabular_specs=[])
        for r in report.results:
            if r.status != "ACCEPTED" or report.baseline_wape is None:
                continue
            self.assertGreaterEqual(
                report.baseline_wape - r.wape, MIN_MEANINGFUL_IMPROVEMENT_PP,
                "acceptance requires a separable improvement, not any improvement",
            )

    def test_ensemble_is_rejected_when_it_does_not_beat_the_best_model(self):
        from forecast.results import ModelEvalResult
        from forecast.selector import select_best_model
        from forecast.backtest import BacktestResult, FoldOutcome, pool_metrics, summarise_folds
        from forecast.evaluator import compute_metrics

        def component(name, prediction):
            folds = [
                FoldOutcome(i + 1, 10 + i, f"p{i}", [f"x{i}"], [100.0], [prediction],
                            compute_metrics([100.0], [prediction]))
                for i in range(3)
            ]
            result = BacktestResult(model=name, folds=folds)
            result.pooled_metrics = pool_metrics(folds)
            result.fold_stats, result.stability = summarise_folds(result.fold_metric_series("wape"))
            return result

        good = component("Naive", 100.0)      # perfect
        poor = component("Moving Average", 250.0)
        results = [
            ModelEvalResult(model="Naive", available=True, trained=True,
                            metrics=good.pooled_metrics, validation_folds=3),
            ModelEvalResult(model="Moving Average", available=True, trained=True,
                            metrics=poor.pooled_metrics, validation_folds=3),
        ]
        selection = select_best_model(
            results, backtests={"Naive": good, "Moving Average": poor},
        )
        self.assertNotIn("Ensemble", selection.selected_model)

    def test_evaluate_ensemble_reports_its_own_measured_error(self):
        from forecast.backtest import BacktestResult, FoldOutcome, pool_metrics, summarise_folds
        from forecast.evaluator import compute_metrics

        def component(name, prediction):
            folds = [
                FoldOutcome(i + 1, 10 + i, f"p{i}", [f"x{i}"], [100.0], [prediction],
                            compute_metrics([100.0], [prediction]))
                for i in range(3)
            ]
            r = BacktestResult(model=name, folds=folds)
            r.pooled_metrics = pool_metrics(folds)
            r.fold_stats, r.stability = summarise_folds(r.fold_metric_series("wape"))
            return r

        ensemble = evaluate_ensemble([component("A", 80.0), component("B", 120.0)])
        self.assertIsNotNone(ensemble)
        self.assertAlmostEqual(ensemble.pooled_metrics["wape"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
