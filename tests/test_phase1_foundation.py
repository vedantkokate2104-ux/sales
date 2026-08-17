"""
test_phase1_foundation.py
--------------------------
Phase 1 test suite: data integrity + forecasting-series foundation.

Covers the sixteen areas Phase 1 specified:
  duplicate handling (repeated invoice numbers, true duplicate rows,
  legitimate repeated identifiers), date+revenue-only datasets, sparse
  data, daily/weekly/monthly/quarterly grains, missing dates, explicit
  zero revenue, short and long history, frequency selection, leakage
  prevention, and forecast-series statistics — plus API backward
  compatibility.

Uses stdlib `unittest` rather than pytest so it runs with no new
dependency:

    python tests/test_phase1_foundation.py
    python -m unittest discover -s tests
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

from data_cleaner import clean
from transaction_key import analyze_duplicates, deduplicate_transactions, resolve_business_key
from forecast.analyzer import analyze_forecast_series, prepare_forecast_series
from forecast.grain import (
    MIN_DENSITY,
    MIN_PERIODS,
    build_forecast_series,
    evaluate_grain,
    resolve_fill_strategy,
    select_forecast_grain,
)
from forecast.results import AutoForecastResult
from forecast.types import SeriesStats
from forecast.utils import detect_seasonality, infer_frequency


# ── helpers ────────────────────────────────────────────────────────────────

def _mapping(col, name, category):
    return {"original_column": col, "business_name": name, "category": category}


ERP_MAPPINGS = [
    _mapping("inv_no",   "Invoice Number",  "identifier"),
    _mapping("inv_dt",   "Invoice Date",    "date"),
    _mapping("cust_cd",  "Customer Code",   "identifier"),
    _mapping("prod_id",  "Product Id",      "identifier"),
    _mapping("qty_sold", "Quantity Sold",   "quantity"),
    _mapping("nsv",      "Net Sales Value", "amount"),
]

MINIMAL_MAPPINGS = [
    _mapping("inv_dt", "Invoice Date",    "date"),
    _mapping("nsv",    "Net Sales Value", "amount"),
]


def _erp_row(inv_no, date, cust="C001", prod="P001", qty=10, nsv=1000.0):
    return {
        "inv_no": inv_no, "inv_dt": date, "cust_cd": cust,
        "prod_id": prod, "qty_sold": qty, "nsv": nsv,
    }


def _dated_series(start, periods, freq, seed=0, base=1000.0, noise=0.08):
    """One transaction per period at `freq`, with mild multiplicative noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq=freq)
    values = base * (1.0 + rng.normal(0, noise, periods))
    return pd.Series(dates), pd.Series(np.abs(values))


# ── 1-3, 11: duplicate handling ────────────────────────────────────────────

class TestDuplicateHandling(unittest.TestCase):
    """The bug this phase exists to fix: an identifier that repeats is not
    a duplicate transaction."""

    def test_repeated_invoice_numbers_are_preserved(self):
        # Same invoice number, genuinely different transactions — exactly
        # the shape of the bundled dataset that lost 54% of its rows.
        df = pd.DataFrame([
            _erp_row("INV1000", "2024-10-16", prod="P002", qty=24, nsv=6785.28),
            _erp_row("INV1000", "2022-03-14", prod="P002", qty=19, nsv=5440.31),
            _erp_row("INV1001", "2023-01-05", prod="P003", qty=5,  nsv=999.0),
        ])
        deduped, report = deduplicate_transactions(df, ERP_MAPPINGS)
        self.assertEqual(len(deduped), 3, "no row should be dropped for a repeated identifier")
        self.assertEqual(report["rows_removed"], 0)
        self.assertEqual(report["repeated_identifier_rows"], 1)
        self.assertEqual(report["preserved_repeated_identifier_rows"], 1)

    def test_exact_duplicate_rows_are_removed(self):
        row = _erp_row("INV2000", "2024-01-01")
        df = pd.DataFrame([row, dict(row), _erp_row("INV2001", "2024-01-02")])
        deduped, report = deduplicate_transactions(df, ERP_MAPPINGS)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertEqual(report["rows_removed"], 1)

    def test_business_key_duplicate_differing_on_incidental_column_is_removed(self):
        # Identical on the business key, differing only on a field that
        # does not identify the event.
        a = _erp_row("INV3000", "2024-05-05")
        b = dict(a)
        df = pd.DataFrame([a, b])
        df["pmt_stat"] = ["PAID", "OVERDUE"]
        deduped, report = deduplicate_transactions(df, ERP_MAPPINGS)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(report["business_key_duplicate_rows"], 1)
        self.assertEqual(report["exact_duplicate_rows"], 0,
                         "these rows are not exact duplicates — the business key is what catches them")

    def test_same_invoice_same_day_different_product_is_kept(self):
        # Two line items on one invoice: same number, same date, different
        # product. Distinct transactions.
        df = pd.DataFrame([
            _erp_row("INV4000", "2024-06-01", prod="P001", nsv=100.0),
            _erp_row("INV4000", "2024-06-01", prod="P002", nsv=250.0),
        ])
        deduped, _ = deduplicate_transactions(df, ERP_MAPPINGS)
        self.assertEqual(len(deduped), 2)

    def test_date_and_revenue_only_dataset_falls_back_to_exact_rows(self):
        # Not enough identifying columns for a business key — must degrade
        # to exact-row detection AND say so, never guess at uniqueness.
        df = pd.DataFrame({
            "inv_dt": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "nsv":    [100.0, 100.0, 200.0],
        })
        key_cols, basis, limitation = resolve_business_key(df, MINIMAL_MAPPINGS)
        self.assertEqual(key_cols, [])
        self.assertIsNotNone(limitation)
        self.assertIn("EXACT duplicate rows only", limitation)

        deduped, report = deduplicate_transactions(df, MINIMAL_MAPPINGS)
        self.assertEqual(report["strategy"], "exact_row")
        self.assertEqual(len(deduped), 2)

    def test_duplicate_report_logs_every_required_count(self):
        df = pd.DataFrame([
            _erp_row("INV5000", "2024-01-01"),
            _erp_row("INV5000", "2024-01-01"),
            _erp_row("INV5000", "2024-07-07", nsv=42.0),
        ])
        report = analyze_duplicates(df, ERP_MAPPINGS)
        for field in ("total_rows", "exact_duplicate_rows", "business_key_duplicate_rows",
                      "rows_retained", "rows_removed", "removal_reason",
                      "repeated_identifier_rows", "preserved_repeated_identifier_rows"):
            self.assertIn(field, report)
        self.assertEqual(report["total_rows"], 3)
        self.assertTrue(report["removal_reason"])

    def test_clean_pipeline_preserves_recycled_invoice_numbers(self):
        # End-to-end through data_cleaner.clean(), which is where the
        # 392 -> 180 collapse used to happen.
        rows = [
            _erp_row(f"INV{1000 + (i % 20)}", f"2024-{(i % 12) + 1:02d}-15",
                     prod=f"P{i % 7:03d}", qty=i + 1, nsv=100.0 * (i + 1))
            for i in range(60)
        ]
        df = pd.DataFrame(rows)
        cleaned, report = clean(df, ERP_MAPPINGS)
        self.assertEqual(report["rows_before"], 60)
        self.assertEqual(len(cleaned), 60, "distinct transactions must survive cleaning")
        self.assertEqual(report["duplicates_removed"], 0)
        self.assertIn("duplicate_detail", report)


# ── 4, 6-9, 13-14: candidate grains and frequency selection ────────────────

class TestGrainSelection(unittest.TestCase):

    def test_daily_dataset_selects_daily(self):
        dates, values = _dated_series("2022-01-01", 400, "D", seed=1)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertEqual(selection.selected_frequency, "daily")
        self.assertFalse(selection.fallback)

    def test_weekly_dataset_selects_weekly(self):
        dates, values = _dated_series("2021-01-04", 156, "W-MON", seed=2, noise=0.05)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertEqual(selection.selected_frequency, "weekly")

    def test_monthly_dataset_selects_monthly(self):
        dates, values = _dated_series("2021-01-01", 48, "MS", seed=3)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertEqual(selection.selected_frequency, "monthly")

    def test_quarterly_dataset_selects_quarterly(self):
        dates, values = _dated_series("2016-01-01", 32, "QS", seed=4)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertEqual(selection.selected_frequency, "quarterly")

    def test_every_candidate_grain_is_evaluated_and_reported(self):
        dates, values = _dated_series("2021-01-01", 48, "MS", seed=5)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertEqual(set(selection.candidate_frequencies), {"daily", "weekly", "monthly", "quarterly"})
        for freq, diag in selection.candidate_frequencies.items():
            self.assertEqual(diag.frequency, freq)
            for field in ("n_periods", "observed_periods", "empty_periods", "zero_value_periods",
                          "zero_pct", "density", "mean", "std", "coefficient_of_variation",
                          "trend_strength", "seasonality_strength", "history_length_days"):
                self.assertIn(field, diag.to_dict())
            if not diag.usable:
                self.assertIsNotNone(diag.rejection_reason, f"{freq} rejected without a reason")

    def test_sparse_transactions_reject_fine_grains(self):
        # One transaction a month for three years: daily and weekly are
        # mostly empty and must be rejected on density.
        dates, values = _dated_series("2021-01-15", 36, "MS", seed=6)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertFalse(selection.candidate_frequencies["daily"].usable)
        self.assertLess(selection.candidate_frequencies["daily"].density, MIN_DENSITY)
        self.assertIn(selection.selected_frequency, ("monthly", "quarterly"))

    def test_frequency_is_not_inferred_from_transaction_arrival_gaps(self):
        # Transactions arrive in tight bursts a few days apart, so the
        # median inter-arrival gap reads "daily"/"weekly" — but most
        # daily and weekly periods are empty, so neither is forecastable.
        rng = np.random.default_rng(7)
        stamps = []
        for month_start in pd.date_range("2021-01-01", periods=40, freq="MS"):
            stamps.extend(month_start + pd.to_timedelta(rng.integers(0, 4, 6), unit="D"))
        dates = pd.Series(sorted(stamps))
        values = pd.Series(np.abs(rng.normal(1000, 80, len(dates))))

        arrival_frequency = infer_frequency(dates)
        selection = select_forecast_grain(dates, values, horizon=3)

        self.assertIn(arrival_frequency, ("daily", "weekly"),
                      "precondition: arrival gaps should look fine-grained here")
        self.assertNotEqual(selection.selected_frequency, arrival_frequency,
                            "grain selection must not simply echo transaction arrival cadence")
        self.assertEqual(selection.selected_frequency, "monthly")

    def test_short_history_falls_back_and_says_so(self):
        dates, values = _dated_series("2024-01-01", 6, "MS", seed=8)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertTrue(selection.fallback)
        self.assertIn("No candidate grain met the minimum requirements", selection.reason)

    def test_long_history_supports_seasonal_cycle(self):
        dates, values = _dated_series("2015-01-01", 120, "MS", seed=9)
        selection = select_forecast_grain(dates, values, horizon=3)
        monthly = selection.candidate_frequencies["monthly"]
        self.assertTrue(monthly.usable)
        self.assertEqual(monthly.seasonal_period, 12)
        self.assertGreaterEqual(monthly.n_periods, MIN_PERIODS)

    def test_horizon_incompatible_grain_is_rejected(self):
        dates, values = _dated_series("2021-01-01", 16, "QS", seed=10)
        diag = evaluate_grain(dates, values, "quarterly", horizon=12)
        self.assertFalse(diag.usable)
        self.assertIn("horizon", diag.rejection_reason.lower())


# ── 5, 10, 11: missing vs zero ─────────────────────────────────────────────

class TestMissingVersusZero(unittest.TestCase):

    def test_missing_period_is_not_counted_as_zero_demand(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-02-15", "2024-05-15"]))
        values = pd.Series([100.0, 200.0, 300.0])
        construction = build_forecast_series(dates, values, "monthly")

        self.assertEqual(construction.n_periods, 5)          # Jan..May
        self.assertEqual(construction.n_observed, 3)
        self.assertEqual(construction.n_missing, 2)          # Mar, Apr never observed
        self.assertEqual(construction.n_explicit_zero, 0)    # none genuinely totalled zero
        self.assertFalse(bool(construction.observed_mask.iloc[2]))
        self.assertTrue(bool(construction.observed_mask.iloc[0]))

    def test_explicit_zero_revenue_is_distinguished_from_missing(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]))
        values = pd.Series([100.0, 0.0, 300.0])
        construction = build_forecast_series(dates, values, "monthly")

        self.assertEqual(construction.n_missing, 0, "every period was observed")
        self.assertEqual(construction.n_explicit_zero, 1, "February was observed and genuinely zero")
        self.assertTrue(bool(construction.observed_mask.iloc[1]))
        self.assertEqual(float(construction.series.iloc[1]), 0.0)

    def test_short_gap_is_interpolated_long_gap_is_zero_filled(self):
        # Dense series with one 1-period gap and one 5-period gap.
        months = [f"2022-{m:02d}-15" for m in range(1, 13)] + [f"2023-{m:02d}-15" for m in range(1, 13)]
        keep = [m for i, m in enumerate(months) if i not in {5} and not (10 <= i <= 14)]
        dates = pd.Series(pd.to_datetime(keep))
        values = pd.Series([1000.0] * len(keep))
        construction = build_forecast_series(dates, values, "monthly")

        self.assertEqual(construction.fill_strategy, "interpolate_short_gaps")
        self.assertEqual(construction.n_interpolated, 1, "the 1-period gap should be bridged")
        self.assertEqual(construction.n_zero_filled, 5, "the 5-period gap must not be invented")
        self.assertTrue(any("recording gaps" in n for n in construction.notes))

    def test_fill_strategy_is_chosen_from_density(self):
        self.assertEqual(resolve_fill_strategy(1.0), "none_needed")
        self.assertEqual(resolve_fill_strategy(0.85), "interpolate_short_gaps")
        self.assertEqual(resolve_fill_strategy(0.40), "zero_filled_sparse")

    def test_sparse_series_is_zero_filled_and_flagged_not_interpolated(self):
        dates, values = _dated_series("2021-01-15", 36, "MS", seed=11)
        construction = build_forecast_series(dates, values, "weekly")
        self.assertEqual(construction.fill_strategy, "zero_filled_sparse")
        self.assertEqual(construction.n_interpolated, 0)
        self.assertTrue(any("density" in n for n in construction.notes))

    def test_constructed_series_is_gapless_and_finite(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-04-15"]))
        values = pd.Series([100.0, 300.0])
        construction = build_forecast_series(dates, values, "monthly")
        self.assertTrue(np.all(np.isfinite(construction.series.to_numpy(dtype=float))))
        self.assertEqual(len(construction.series), len(construction.observed_mask))


# ── 8: frequency-aware seasonality ─────────────────────────────────────────

class TestSeasonalityIsFrequencyAware(unittest.TestCase):

    def test_weekly_series_uses_a_52_period_cycle(self):
        values = np.tile(np.sin(np.linspace(0, 2 * np.pi, 52)) + 5, 4)
        strength, period = detect_seasonality(values, "weekly")
        self.assertEqual(period, 52, "weekly data must not be tested at period=12")
        self.assertGreater(strength, 0.5)

    def test_monthly_series_uses_a_12_period_cycle(self):
        values = np.tile(np.sin(np.linspace(0, 2 * np.pi, 12)) + 5, 5)
        strength, period = detect_seasonality(values, "monthly")
        self.assertEqual(period, 12)
        self.assertGreater(strength, 0.5)

    def test_quarterly_series_uses_a_4_period_cycle(self):
        values = np.tile([10.0, 20.0, 15.0, 25.0], 8)
        strength, period = detect_seasonality(values, "quarterly")
        self.assertEqual(period, 4)
        self.assertGreater(strength, 0.9)

    def test_daily_series_can_use_a_weekly_cycle(self):
        values = np.tile([5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 1.0], 30)
        strength, period = detect_seasonality(values, "daily")
        self.assertEqual(period, 7)
        self.assertGreater(strength, 0.9)

    def test_short_series_gets_no_fabricated_seasonality(self):
        strength, period = detect_seasonality(np.arange(10, dtype=float), "monthly")
        self.assertIsNone(period, "fewer than 2 full cycles — nothing can be measured")
        self.assertEqual(strength, 0.0)

    def test_yearly_frequency_has_no_subcycle(self):
        strength, period = detect_seasonality(np.arange(50, dtype=float), "yearly")
        self.assertIsNone(period)
        self.assertEqual(strength, 0.0)


# ── 15: leakage prevention ─────────────────────────────────────────────────

class TestLeakagePrevention(unittest.TestCase):

    def test_selection_ignores_the_holdout_tail(self):
        dates, values = _dated_series("2021-01-01", 48, "MS", seed=12)
        baseline = select_forecast_grain(dates, values, horizon=3)

        # Replace the final 20% of the calendar span with wildly different
        # values. Selection statistics come from the leading span only, so
        # the chosen grain must not move.
        tampered = values.copy()
        cutoff = int(len(values) * 0.8)
        tampered.iloc[cutoff:] = tampered.iloc[cutoff:] * 500.0
        tampered_selection = select_forecast_grain(dates, tampered, horizon=3)

        self.assertEqual(baseline.selected_frequency, tampered_selection.selected_frequency,
                         "the holdout tail must not influence grain selection")

    def test_selection_never_fits_a_forecasting_model(self):
        from forecast.models import MODEL_REGISTRY

        originals = {}
        try:
            for name, cls in MODEL_REGISTRY.items():
                originals[name] = cls.fit

                def _boom(self, *a, **kw):
                    raise AssertionError("grain selection must not fit any model")

                cls.fit = _boom

            dates, values = _dated_series("2021-01-01", 48, "MS", seed=13)
            selection = select_forecast_grain(dates, values, horizon=3)
            self.assertEqual(selection.selected_frequency, "monthly")
        finally:
            for name, cls in MODEL_REGISTRY.items():
                cls.fit = originals[name]

    def test_selection_basis_is_declared(self):
        dates, values = _dated_series("2021-01-01", 48, "MS", seed=14)
        selection = select_forecast_grain(dates, values, horizon=3)
        self.assertIn("no model was fit", selection.selection_basis)
        self.assertIn("80%", selection.selection_basis)


# ── 16: forecast-series statistics ─────────────────────────────────────────

class TestForecastSeriesStatistics(unittest.TestCase):

    def test_n_observations_counts_periods_not_transactions(self):
        # 200 transactions landing in 24 months.
        rng = np.random.default_rng(15)
        months = pd.date_range("2022-01-01", periods=24, freq="MS")
        dates = pd.Series(rng.choice(months, 200)) + pd.to_timedelta(rng.integers(0, 27, 200), unit="D")
        values = pd.Series(np.abs(rng.normal(500, 50, 200)))

        _construction, _selection, stats = prepare_forecast_series(
            dates, values, horizon=3, frequency="monthly",
        )
        self.assertEqual(stats.n_observations, 24, "must be modelled periods, not invoices")
        self.assertEqual(stats.n_transactions, 200, "raw transaction count is preserved separately")

    def test_pct_zero_sales_counts_zero_periods(self):
        dates = pd.Series(pd.to_datetime([f"2023-{m:02d}-15" for m in range(1, 13)]))
        values = pd.Series([0.0, 0.0, 0.0] + [100.0] * 9)
        construction = build_forecast_series(dates, values, "monthly")
        stats = analyze_forecast_series(construction, n_transactions=12)
        self.assertEqual(stats.n_observations, 12)
        self.assertAlmostEqual(stats.pct_zero_sales, 25.0, places=2)
        self.assertAlmostEqual(stats.zero_period_ratio, 0.25, places=4)

    def test_cv_describes_the_aggregated_series_not_individual_rows(self):
        # Individual transactions vary hugely; monthly totals are steady.
        rng = np.random.default_rng(16)
        dates, values = [], []
        for month in pd.date_range("2022-01-01", periods=24, freq="MS"):
            amounts = rng.choice([10.0, 5000.0], 10)
            amounts = amounts * (1000.0 / amounts.sum())
            for offset, amount in enumerate(amounts):
                dates.append(month + pd.Timedelta(days=offset))
                values.append(amount)
        dates, values = pd.Series(dates), pd.Series(values)

        _c, _s, stats = prepare_forecast_series(dates, values, horizon=3, frequency="monthly")
        raw_cv = float(values.std() / values.mean())
        self.assertGreater(raw_cv, 1.0, "precondition: row-level spread is large")
        self.assertLess(stats.coefficient_of_variation, 0.05,
                        "series-level CV must describe monthly totals, which are near constant")

    def test_stats_report_density_and_empty_periods(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-02-15", "2024-05-15"]))
        values = pd.Series([100.0, 200.0, 300.0])
        construction = build_forecast_series(dates, values, "monthly")
        stats = analyze_forecast_series(construction, n_transactions=3)
        self.assertEqual(stats.observed_periods, 3)
        self.assertEqual(stats.empty_periods, 2)
        self.assertAlmostEqual(stats.series_density, 0.6, places=4)
        self.assertEqual(stats.frequency, "monthly")

    def test_seasonality_uses_the_selected_frequency(self):
        dates, values = _dated_series("2018-01-01", 200, "W-MON", seed=17)
        _c, _s, stats = prepare_forecast_series(dates, values, horizon=3, frequency="weekly")
        self.assertEqual(stats.frequency, "weekly")
        self.assertEqual(stats.seasonal_period, 52, "weekly stats must not be measured at period=12")


# ── 9: API backward compatibility ──────────────────────────────────────────

class TestApiBackwardCompatibility(unittest.TestCase):

    LEGACY_RESULT_FIELDS = [
        "horizon", "selected_model", "selection_reason", "evaluation_metrics",
        "forecast_confidence", "model_ranking", "forecast", "alternative_models",
        "explainability", "warnings", "frequency", "forecast_confidence_pct",
        "data_quality_score",
    ]

    LEGACY_STATS_FIELDS = [
        "group", "n_observations", "start_date", "end_date", "frequency",
        "missing_timestamps", "missing_values", "trend_strength",
        "seasonality_strength", "variance", "coefficient_of_variation",
        "pct_zero_sales", "outlier_percentage", "history_length_days",
    ]

    def test_auto_forecast_result_keeps_every_legacy_field(self):
        result = AutoForecastResult(
            horizon=3, selected_model="Naive", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        )
        payload = result.to_dict()
        for field in self.LEGACY_RESULT_FIELDS:
            self.assertIn(field, payload, f"removing '{field}' would break existing API consumers")

    def test_new_result_fields_are_additive_and_optional(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="Naive", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in ("selected_frequency", "frequency_reason", "series_density",
                      "zero_period_ratio", "candidate_frequencies", "n_modelled_periods",
                      "n_transactions"):
            self.assertIn(field, payload)
            self.assertIsNone(payload[field], "new fields must default to None, not crash")

    def test_series_stats_keeps_every_legacy_field(self):
        stats = SeriesStats(
            group="ALL", n_observations=10, start_date="2024-01-01", end_date="2024-10-01",
            frequency="monthly", missing_timestamps=0, missing_values=0, trend_strength=0.1,
            seasonality_strength=0.2, variance=1.0, coefficient_of_variation=0.3,
            pct_zero_sales=0.0, outlier_percentage=0.0, history_length_days=273,
        )
        payload = stats.to_dict()
        for field in self.LEGACY_STATS_FIELDS:
            self.assertIn(field, payload)

    def test_frequency_and_selected_frequency_agree(self):
        dates, values = _dated_series("2021-01-01", 48, "MS", seed=18)
        _c, selection, stats = prepare_forecast_series(dates, values, horizon=3)
        self.assertEqual(stats.frequency, selection.selected_frequency,
                         "'frequency' is retained for existing consumers and must mirror the new field")


if __name__ == "__main__":
    unittest.main(verbosity=2)
