"""
test_phase2_validation.py
--------------------------
Phase 2 test suite: honest time-series validation and model benchmarking.

Covers the fourteen areas Phase 2 specified: temporal leakage, fold
ordering, train-precedes-validation, validation horizon matching the
production horizon, per-fold preprocessing, Seasonal Naive, real ensemble
metrics, rankings built from all folds, and short / monthly / weekly /
quarterly / missing-value / zero-heavy series.

Stdlib `unittest`, no new dependency:

    python tests/test_phase2_validation.py
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

from forecast.backtest import (
    MIN_ABSOLUTE_TRAIN,
    MIN_FOLDS,
    BacktestResult,
    FoldOutcome,
    FoldPreprocessor,
    ValidationPlan,
    backtest_model,
    evaluate_ensemble,
    plan_validation,
    pool_metrics,
    prepare_training_window,
    summarise_folds,
)
from forecast.benchmark import benchmark_series
from forecast.evaluator import compute_metrics
from forecast.grain import build_forecast_series
from forecast.models import MODEL_REGISTRY
from forecast.models.base import BaseForecastModel
from forecast.models.seasonal_naive import SeasonalNaiveModel
from forecast.selector import select_best_model
from forecast.trainer import train_candidates


# ── helpers ────────────────────────────────────────────────────────────────

def _period_series(n, freq="M", start="2021-01", base=1000.0, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, periods=n, freq=freq)
    values = np.abs(base * (1.0 + rng.normal(0, noise, n)))
    return pd.Series(values, index=index)


def _all_observed(series):
    return pd.Series(True, index=series.index)


class RecordingModel(BaseForecastModel):
    """Captures exactly what each fold handed it, so leakage can be
    asserted on directly rather than inferred."""
    name = "Recording"
    seen = []

    def fit(self, series, frequency="monthly", **hyperparams):
        type(self).seen.append(series.copy())
        self.series_ = series
        self.frequency_ = frequency
        self.fitted = True
        return self

    def predict(self, steps):
        return np.full(steps, float(self.series_.iloc[-1]), dtype=float)


class ConstantModel(BaseForecastModel):
    """Predicts a fixed value — used to build ensembles with known,
    hand-checkable arithmetic."""
    name = "Constant"
    value = 0.0

    def fit(self, series, frequency="monthly", **hyperparams):
        self.series_ = series
        self.fitted = True
        return self

    def predict(self, steps):
        return np.full(steps, float(type(self).value), dtype=float)


# ── 1-3: leakage and fold ordering ─────────────────────────────────────────

class TestNoTemporalLeakage(unittest.TestCase):

    def setUp(self):
        RecordingModel.seen = []

    def test_training_data_always_precedes_validation_data(self):
        series = _period_series(48, seed=1)
        plan = plan_validation(48, 3, 12)
        backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")

        self.assertEqual(len(RecordingModel.seen), plan.n_folds)
        for spec, train in zip(plan.folds, RecordingModel.seen):
            last_train_period = train.index[-1]
            first_val_period = series.index[spec.val_start]
            self.assertLess(last_train_period, first_val_period,
                            "a fold saw data at or after its own validation origin")
            self.assertEqual(len(train), spec.train_end)

    def test_folds_are_chronological_and_expanding(self):
        plan = plan_validation(48, 3, 12)
        ends = [f.train_end for f in plan.folds]
        self.assertEqual(ends, sorted(ends), "fold origins must be chronological")
        self.assertEqual(len(ends), len(set(ends)), "origins must not repeat")
        for earlier, later in zip(ends, ends[1:]):
            self.assertLess(earlier, later, "each fold must train on strictly more history")

    def test_train_end_equals_val_start_for_every_fold(self):
        for n in (24, 36, 48, 120):
            plan = plan_validation(n, 3, 12)
            for spec in plan.folds:
                self.assertEqual(spec.train_end, spec.val_start,
                                 "validation must begin exactly where training ends — no overlap, no gap")
                self.assertLessEqual(spec.val_end, n)

    def test_future_values_cannot_influence_earlier_folds(self):
        # Two series identical up to index 38, wildly different after.
        # Every fold whose validation window ends at or before 38 must
        # receive byte-identical training input from both.
        base = _period_series(48, seed=2)
        tampered = base.copy()
        tampered.iloc[38:] = tampered.iloc[38:] * 1000.0
        plan = plan_validation(48, 3, 12)

        RecordingModel.seen = []
        backtest_model(base, _all_observed(base), RecordingModel, plan, "monthly")
        seen_base = list(RecordingModel.seen)

        RecordingModel.seen = []
        backtest_model(tampered, _all_observed(tampered), RecordingModel, plan, "monthly")
        seen_tampered = list(RecordingModel.seen)

        compared = 0
        for spec, a, b in zip(plan.folds, seen_base, seen_tampered):
            if spec.val_end <= 38:
                np.testing.assert_allclose(
                    a.to_numpy(), b.to_numpy(),
                    err_msg=f"fold ending at {spec.val_end} was influenced by data after index 38",
                )
                compared += 1
        self.assertGreater(compared, 0, "precondition: some folds should end before the tampered region")


# ── 4: validation horizon matches production horizon ───────────────────────

class TestValidationHorizon(unittest.TestCase):

    def test_validation_horizon_equals_requested_horizon(self):
        for n in (36, 48, 120):
            plan = plan_validation(n, 3, 12)
            self.assertEqual(plan.horizon, 3)
            self.assertFalse(plan.degraded)
            for spec in plan.folds:
                self.assertEqual(spec.val_end - spec.val_start, 3)

    def test_every_fold_is_scored_at_the_full_horizon(self):
        series = _period_series(48, seed=3)
        plan = plan_validation(48, 3, 12)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")
        for fold in result.folds:
            self.assertEqual(len(fold.actuals), 3, "t+1, t+2 and t+3 must all be evaluated")
            self.assertEqual(len(fold.predictions), 3)

    def test_short_series_degrades_horizon_and_reports_it(self):
        plan = plan_validation(12, 3, 12)
        self.assertTrue(plan.degraded)
        self.assertIsNotNone(plan.degradation_reason)
        self.assertLess(plan.horizon, 3)
        self.assertIn("reduced", plan.degradation_reason)

    def test_engine_validation_horizon_matches_requested(self):
        dates = pd.Series(pd.date_range("2021-01-15", periods=48, freq="MS"))
        values = pd.Series(np.abs(np.random.default_rng(4).normal(1000, 100, 48)))
        outcome = train_candidates(dates, values, ["Naive"], frequency="monthly", horizon=3)
        self.assertEqual(outcome.plan.horizon, 3)
        self.assertEqual(outcome.plan.requested_horizon, 3)


# ── 5: preprocessing fitted inside each fold ───────────────────────────────

class TestPerFoldPreprocessing(unittest.TestCase):

    def test_outlier_bounds_come_from_training_data_only(self):
        # Varied training values so the IQR is non-degenerate, then future
        # spikes that would widen it if they were (wrongly) included.
        train_only = np.arange(1.0, 21.0)
        with_future_spike = np.append(train_only, [10_000.0] * 5)

        pre_train = FoldPreprocessor.fit(train_only)
        pre_full = FoldPreprocessor.fit(with_future_spike)

        self.assertIsNotNone(pre_train.upper)
        self.assertIsNotNone(pre_full.upper)
        self.assertLess(pre_train.upper, pre_full.upper,
                        "bounds fitted on the training window must not reflect later observations")

        # And a constant window legitimately yields no bounds at all.
        self.assertIsNone(FoldPreprocessor.fit(np.array([100.0] * 20)).upper)

    def test_a_late_spike_does_not_change_early_fold_training_values(self):
        values = np.concatenate([np.full(30, 100.0), np.full(5, 5_000_000.0)])
        observed = np.ones(len(values), dtype=bool)

        early, _ = prepare_training_window(values[:30], observed[:30])
        full, _ = prepare_training_window(values, observed)

        np.testing.assert_allclose(early, np.full(30, 100.0),
                                   err_msg="early fold training data was altered by a later spike")
        self.assertLess(full[:30].max(), 5_000_000.0)

    def test_missing_periods_are_bridged_using_training_data_only(self):
        values = np.array([10.0, 20.0, 0.0, 40.0, 50.0])
        observed = np.array([True, True, False, True, True])
        # Training window stops before the gap's right-hand neighbour, so
        # there is nothing legitimate to interpolate towards.
        bridged, _ = prepare_training_window(values[:3], observed[:3])
        self.assertEqual(bridged[2], 0.0, "a gap must not be bridged using a value the fold cannot see")

        full, _ = prepare_training_window(values, observed)
        self.assertAlmostEqual(full[2], 30.0, places=6, msg="within one window the gap should be bridged")

    def test_validation_actuals_are_never_transformed(self):
        # A spike inside the validation window must be scored as-is.
        series = _period_series(40, seed=5, noise=0.01)
        series.iloc[-2] = series.iloc[-2] * 50
        plan = plan_validation(len(series), 3, 12)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")

        spike_value = float(series.iloc[-2])
        scored = [a for fold in result.folds for a in fold.actuals]
        self.assertTrue(any(abs(a - spike_value) < 1e-6 for a in scored),
                        "the raw spike should appear among the validation actuals, uncapped")


# ── 6: Seasonal Naive ──────────────────────────────────────────────────────

class TestSeasonalNaive(unittest.TestCase):

    def test_monthly_repeats_the_value_twelve_periods_earlier(self):
        series = _period_series(36, seed=6)
        model = SeasonalNaiveModel().fit(series, frequency="monthly")
        predictions = model.predict(3)
        expected = series.iloc[-12:-9].to_numpy()
        np.testing.assert_allclose(predictions, expected)

    def test_quarterly_uses_a_four_period_cycle(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0] * 5,
                           index=pd.period_range("2019Q1", periods=20, freq="Q"))
        model = SeasonalNaiveModel().fit(series, frequency="quarterly")
        np.testing.assert_allclose(model.predict(4), [10.0, 20.0, 30.0, 40.0])

    def test_horizon_longer_than_the_cycle_wraps_around(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0] * 5,
                           index=pd.period_range("2019Q1", periods=20, freq="Q"))
        model = SeasonalNaiveModel().fit(series, frequency="quarterly")
        np.testing.assert_allclose(model.predict(6), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0])

    def test_blocked_without_two_full_cycles(self):
        series = _period_series(18, seed=7)
        ok, message = SeasonalNaiveModel().check_requirements(series, "monthly")
        self.assertFalse(ok)
        self.assertIn("2 full seasonal cycles", message)

    def test_registered_and_treated_as_a_guaranteed_candidate(self):
        from forecast.trainer import GUARANTEED_CANDIDATES
        self.assertIn("Seasonal Naive", MODEL_REGISTRY)
        self.assertIn("Seasonal Naive", GUARANTEED_CANDIDATES)

    def test_seasonal_naive_is_benchmarked_and_ranked(self):
        dates = pd.Series(pd.date_range("2021-01-15", periods=48, freq="MS"))
        values = pd.Series(np.abs(np.random.default_rng(8).normal(1000, 100, 48)))
        outcome = train_candidates(dates, values, [], frequency="monthly", horizon=3)
        names = [r.model for r in outcome.results]
        self.assertIn("Seasonal Naive", names)
        entry = next(r for r in outcome.results if r.model == "Seasonal Naive")
        self.assertTrue(entry.trained)
        self.assertGreater(entry.validation_folds, 0)


# ── 7: per-fold metrics and stability ──────────────────────────────────────

class TestFoldMetricsAndStability(unittest.TestCase):

    def test_per_fold_metrics_are_exposed(self):
        series = _period_series(48, seed=9)
        plan = plan_validation(48, 3, 12)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")
        self.assertEqual(len(result.folds), plan.n_folds)
        for fold in result.folds:
            for key in ("mae", "rmse", "mape", "smape", "wape"):
                self.assertIn(key, fold.metrics)

    def test_fold_statistics_capture_spread_not_just_the_mean(self):
        steady, steady_stability = summarise_folds([38.0, 39.0, 40.0, 41.0])
        erratic, erratic_stability = summarise_folds([30.0, 31.0, 32.0, 90.0])

        self.assertLess(steady["std"], erratic["std"])
        self.assertGreater(steady_stability, erratic_stability,
                           "a model with one catastrophic fold must score less stable")
        for key in ("mean", "median", "std", "min", "max"):
            self.assertIn(key, steady)
        self.assertEqual(steady["min"], 38.0)
        self.assertEqual(steady["max"], 41.0)

    def test_stability_is_none_with_a_single_fold(self):
        _stats, stability = summarise_folds([42.0])
        self.assertIsNone(stability)

    def test_pooled_metrics_are_computed_once_over_pooled_pairs(self):
        folds = [
            FoldOutcome(1, 10, "p", ["a"], [100.0], [90.0], compute_metrics([100.0], [90.0])),
            FoldOutcome(2, 11, "p", ["b"], [200.0], [260.0], compute_metrics([200.0], [260.0])),
        ]
        pooled = pool_metrics(folds)
        expected = compute_metrics([100.0, 200.0], [90.0, 260.0])
        self.assertEqual(pooled["rmse"], expected["rmse"])
        self.assertNotEqual(pooled["rmse"], pooled["mae"],
                            "pooling must preserve RMSE's definition rather than collapsing it to MAE")

    def test_model_eval_result_carries_fold_detail(self):
        dates = pd.Series(pd.date_range("2021-01-15", periods=48, freq="MS"))
        values = pd.Series(np.abs(np.random.default_rng(10).normal(1000, 100, 48)))
        outcome = train_candidates(dates, values, ["Naive"], frequency="monthly", horizon=3)
        naive = next(r for r in outcome.results if r.model == "Naive")
        self.assertIsNotNone(naive.fold_metrics)
        self.assertEqual(len(naive.fold_metrics), naive.validation_folds)
        self.assertIsNotNone(naive.fold_stats)


# ── 8: ensembles are actually evaluated ────────────────────────────────────

class TestEnsembleEvaluation(unittest.TestCase):

    def _component(self, name, prediction):
        folds = []
        for i in range(3):
            actuals = [100.0, 100.0]
            preds = [prediction, prediction]
            folds.append(FoldOutcome(i + 1, 10 + i, f"p{i}", [f"x{i}", f"y{i}"],
                                     actuals, preds, compute_metrics(actuals, preds)))
        result = BacktestResult(model=name, folds=folds)
        result.pooled_metrics = pool_metrics(folds)
        result.fold_stats, result.stability = summarise_folds(result.fold_metric_series("wape"))
        return result

    def test_ensemble_metrics_come_from_blended_predictions(self):
        low = self._component("Low", 80.0)     # 20% under
        high = self._component("High", 120.0)  # 20% over
        ensemble = evaluate_ensemble([low, high])

        self.assertIsNotNone(ensemble)
        # Equal weights on fold 1; inverse-error weights thereafter are
        # equal too since both components err identically. The blend lands
        # exactly on the actuals.
        self.assertAlmostEqual(ensemble.pooled_metrics["wape"], 0.0, places=6)
        self.assertAlmostEqual(low.pooled_metrics["wape"], 20.0, places=6)
        self.assertAlmostEqual(high.pooled_metrics["wape"], 20.0, places=6)
        self.assertNotEqual(ensemble.pooled_metrics["wape"], low.pooled_metrics["wape"],
                            "the ensemble must not inherit a component's metrics")

    def test_ensemble_is_scored_on_the_same_folds_as_its_components(self):
        a, b = self._component("A", 90.0), self._component("B", 110.0)
        ensemble = evaluate_ensemble([a, b])
        self.assertEqual(len(ensemble.folds), len(a.folds))
        self.assertEqual([f.fold for f in ensemble.folds], [f.fold for f in a.folds])

    def test_first_fold_uses_equal_weights_and_never_its_own_outcome(self):
        a, b = self._component("A", 60.0), self._component("B", 140.0)
        ensemble = evaluate_ensemble([a, b])
        first = ensemble.folds[0]
        # Equal weights => (60 + 140) / 2 = 100
        np.testing.assert_allclose(first.predictions, [100.0, 100.0])
        self.assertIn("only from the folds before it", ensemble.weight_basis)

    def test_worse_ensemble_is_rejected_by_the_selector(self):
        from forecast.results import ModelEvalResult

        good = self._component("Naive", 100.0)          # perfect
        bad = self._component("Moving Average", 200.0)  # 100% off
        results = [
            ModelEvalResult(model="Naive", available=True, trained=True,
                            metrics=good.pooled_metrics, validation_folds=3, stability=good.stability),
            ModelEvalResult(model="Moving Average", available=True, trained=True,
                            metrics=bad.pooled_metrics, validation_folds=3, stability=bad.stability),
        ]
        selection = select_best_model(
            results, backtests={"Naive": good, "Moving Average": bad},
        )
        self.assertNotIn("Ensemble", selection.selected_model,
                         "an ensemble that does not beat the best single model must not be selected")

    def test_no_unconditional_ensemble_confidence_bonus(self):
        from forecast.results import ModelEvalResult

        a, b = self._component("Naive", 95.0), self._component("Moving Average", 105.0)
        results = [
            ModelEvalResult(model="Naive", available=True, trained=True,
                            metrics=a.pooled_metrics, validation_folds=3, stability=a.stability),
            ModelEvalResult(model="Moving Average", available=True, trained=True,
                            metrics=b.pooled_metrics, validation_folds=3, stability=b.stability),
        ]
        with_ensemble = select_best_model(results, backtests={"Naive": a, "Moving Average": b})
        without_ensemble = select_best_model(results)

        if "Ensemble" in with_ensemble.selected_model:
            self.assertEqual(
                with_ensemble.confidence_pct, without_ensemble.confidence_pct,
                "selecting an ensemble must not by itself change the confidence score",
            )
            self.assertIsNotNone(with_ensemble.ensemble_evaluation)
            self.assertEqual(with_ensemble.metrics, with_ensemble.ensemble_evaluation["pooled_metrics"],
                             "reported metrics must be the ensemble's own")


# ── 9: rankings use all folds ──────────────────────────────────────────────

class TestRankingUsesAllFolds(unittest.TestCase):

    def test_every_model_is_scored_on_the_same_fold_plan(self):
        dates = pd.Series(pd.date_range("2021-01-15", periods=48, freq="MS"))
        values = pd.Series(np.abs(np.random.default_rng(11).normal(1000, 100, 48)))
        outcome = train_candidates(dates, values, [], frequency="monthly", horizon=3)

        validated = [r for r in outcome.results if r.trained and r.metrics]
        self.assertGreaterEqual(len(validated), 3)
        for r in validated:
            self.assertEqual(r.validation_folds, outcome.plan.n_folds,
                             f"{r.model} was scored on a different number of folds")

    def test_pooled_metrics_span_more_than_a_single_window(self):
        dates = pd.Series(pd.date_range("2021-01-15", periods=48, freq="MS"))
        values = pd.Series(np.abs(np.random.default_rng(12).normal(1000, 100, 48)))
        outcome = train_candidates(dates, values, [], frequency="monthly", horizon=3)
        naive = next(r for r in outcome.results if r.model == "Naive")
        self.assertGreaterEqual(naive.metrics["n_points"], 3 * MIN_FOLDS)

    def test_simpler_model_wins_an_empirical_tie(self):
        from forecast.results import ModelEvalResult
        tied = {"wape": 40.0, "smape": 40.0, "rmse": 100.0, "mae": 90.0}
        results = [
            ModelEvalResult(model="SARIMA", available=True, trained=True,
                            metrics=dict(tied), validation_folds=8, stability=0.7),
            ModelEvalResult(model="Naive", available=True, trained=True,
                            metrics=dict(tied), validation_folds=8, stability=0.7),
        ]
        selection = select_best_model(results)
        self.assertEqual(selection.selected_model, "Naive",
                         "with identical evidence the simpler model should win")


# ── 9-14: series shapes ────────────────────────────────────────────────────

class TestSeriesShapes(unittest.TestCase):

    def _benchmark(self, series, horizon=3):
        construction = build_forecast_series(
            pd.Series(series.index.to_timestamp()), pd.Series(series.to_numpy()), series.index.freqstr and
            {"M": "monthly", "W-SUN": "weekly", "Q-DEC": "quarterly", "D": "daily"}.get(series.index.freqstr, "monthly"),
        )
        return benchmark_series(construction, horizon=horizon)

    def test_monthly_series(self):
        series = _period_series(48, freq="M", seed=13)
        plan = plan_validation(len(series), 3, 12)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")
        self.assertGreaterEqual(result.n_folds, MIN_FOLDS)
        self.assertEqual(plan.horizon, 3)

    def test_weekly_series(self):
        series = _period_series(160, freq="W", start="2021-01-04", seed=14)
        plan = plan_validation(len(series), 3, 52)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "weekly")
        self.assertGreaterEqual(result.n_folds, MIN_FOLDS)
        self.assertGreaterEqual(plan.min_train_size, 104, "weekly plans should hold back two annual cycles")

    def test_quarterly_series(self):
        series = _period_series(32, freq="Q", start="2016Q1", seed=15)
        plan = plan_validation(len(series), 3, 4)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "quarterly")
        self.assertGreaterEqual(result.n_folds, MIN_FOLDS)

    def test_short_history_still_produces_a_usable_or_honest_result(self):
        series = _period_series(14, freq="M", seed=16)
        plan = plan_validation(len(series), 3, 12)
        if plan.n_folds:
            self.assertTrue(plan.degraded or plan.min_train_size >= MIN_ABSOLUTE_TRAIN)
        else:
            self.assertIsNotNone(plan.degradation_reason)

    def test_series_with_missing_periods_scores_only_observed_ones(self):
        series = _period_series(48, seed=17)
        observed = _all_observed(series)
        observed.iloc[30:34] = False
        plan = plan_validation(len(series), 3, 12)
        result = backtest_model(series, observed, RecordingModel, plan, "monthly")

        for fold in result.folds:
            for period in fold.periods:
                position = list(series.index.astype(str)).index(period)
                self.assertTrue(bool(observed.iloc[position]),
                                "an unobserved period must never be scored as though it were measured")

    def test_zero_heavy_series_does_not_crash_and_reports_honestly(self):
        values = np.zeros(48)
        values[::4] = 500.0
        series = pd.Series(values, index=pd.period_range("2021-01", periods=48, freq="M"))
        plan = plan_validation(len(series), 3, 12)
        result = backtest_model(series, _all_observed(series), RecordingModel, plan, "monthly")

        self.assertGreater(result.n_folds, 0)
        self.assertIsNotNone(result.pooled_metrics)
        # WAPE stays finite because it is a ratio of sums; per-fold values
        # may legitimately be None where a whole window was zero.
        self.assertIsNotNone(result.pooled_metrics["wape"])
        self.assertTrue(np.isfinite(result.pooled_metrics["wape"]))

    def test_benchmark_reports_unavailable_models_separately(self):
        series = _period_series(48, seed=18)
        construction = build_forecast_series(
            pd.Series(series.index.to_timestamp()), pd.Series(series.to_numpy()), "monthly",
        )
        report = benchmark_series(construction, horizon=3)
        self.assertGreater(len(report.rows), 0)
        self.assertIsInstance(report.unavailable, dict)
        for name in report.unavailable:
            self.assertNotIn(name, [r.model for r in report.rows],
                             "a model cannot be both benchmarked and unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
