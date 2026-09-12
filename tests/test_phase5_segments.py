"""
test_phase5_segments.py
------------------------
Phase 5 test suite: grouped / segment-level forecasting.

Covers the areas Phase 5 specified: company and segment forecasting
through one pipeline, dynamic column and value discovery, invalid /
missing / empty groups, history classification, per-segment grain,
validation, model selection and intervals, high-cardinality safeguards,
API compatibility, and the company/segment mix-up that motivated the
phase.

Uses a synthetic ERP frame with a realistic dimensional structure
(products, categories, regions, channels) so nothing depends on the
bundled dataset.

    python tests/test_phase5_segments.py
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

from forecast.forecast_engine import run_auto_forecast
from forecast.results import AutoForecastResult
from forecast.segments import (
    GROUP_EMPTY,
    GROUP_MISSING_VALUE,
    GROUP_NOT_GROUPABLE,
    GROUP_OK,
    GROUP_UNKNOWN_COLUMN,
    GROUP_UNKNOWN_VALUE,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    MAX_CARDINALITY,
    SUFFICIENT_HISTORY,
    classify_history,
    discover_group_columns,
    discover_group_values,
    group_metadata,
    resolve_group,
)


# ── synthetic ERP data ─────────────────────────────────────────────────────

MAPPINGS = [
    {"original_column": "Invoice_Number", "business_name": "Invoice Number",  "category": "identifier"},
    {"original_column": "Date",           "business_name": "Invoice Date",    "category": "date"},
    {"original_column": "Product_ID",     "business_name": "Product Id",      "category": "identifier"},
    {"original_column": "Product_Name",   "business_name": "Product Name",    "category": "product"},
    {"original_column": "Category",       "business_name": "Product Category", "category": "product"},
    {"original_column": "Region",         "business_name": "Region",          "category": "geography"},
    {"original_column": "Sales_Channel",  "business_name": "Sales Channel",   "category": "product"},
    {"original_column": "Customer_ID",    "business_name": "Customer Code",   "category": "identifier"},
    {"original_column": "Quantity",       "business_name": "Quantity Sold",   "category": "quantity"},
    {"original_column": "Unit_Price",     "business_name": "Unit Price",      "category": "amount"},
    {"original_column": "Revenue",        "business_name": "Net Sales Value", "category": "amount"},
]

PRODUCTS = [("P001", "Widget", "Hardware"), ("P002", "Gadget", "Hardware"),
            ("P003", "Service", "Services"), ("P004", "Licence", "Software")]
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Direct", "Partner"]


def make_erp(months=48, seed=0, scale=1.0, short_products=True, start="2021-01-01"):
    """
    A realistic ERP extract: several products, categories, regions and
    channels across `months` months, with deliberately uneven history so
    the small-segment paths are exercised.
    """
    rng = np.random.default_rng(seed)
    periods = pd.date_range(start, periods=months, freq="MS")
    rows, invoice = [], 1000

    for i, month in enumerate(periods):
        for pid, pname, category in PRODUCTS:
            # P004 only starts three quarters in, and P003 appears only in
            # the final few months, so the segment set spans sufficient,
            # limited and insufficient history.
            if short_products and pid == "P004" and i < int(months * 0.75):
                continue
            if short_products and pid == "P003" and i < months - 4:
                continue
            for region in REGIONS:
                if rng.random() < 0.15:      # a genuinely missing period
                    continue
                invoice += 1
                qty = int(rng.integers(5, 60))
                price = float(abs(rng.normal(900, 120))) * scale
                rows.append({
                    "Invoice_Number": f"INV{invoice}",
                    "Date": month.strftime("%Y-%m-%d"),
                    "Product_ID": pid, "Product_Name": pname, "Category": category,
                    "Region": region,
                    "Sales_Channel": CHANNELS[(i + len(region)) % len(CHANNELS)],
                    "Customer_ID": f"C{rng.integers(1, 26):03d}",
                    "Quantity": qty, "Unit_Price": round(price, 2),
                    "Revenue": round(qty * price, 2),
                })
    return pd.DataFrame(rows)


def _total(result_dict):
    return sum(p["value"] for p in result_dict.get("forecast", []))


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_erp()
        cls.mappings = MAPPINGS


# ── discovery ──────────────────────────────────────────────────────────────

class TestGroupDiscovery(_Base):

    def test_meaningful_dimensions_are_discovered(self):
        names = {c.name for c in discover_group_columns(self.df, self.mappings)}
        for expected in ("Product_ID", "Category", "Region", "Sales_Channel"):
            self.assertIn(expected, names, f"{expected} should be a usable dimension")

    def test_measures_and_dates_are_excluded(self):
        names = {c.name for c in discover_group_columns(self.df, self.mappings)}
        for excluded in ("Revenue", "Quantity", "Unit_Price", "Date"):
            self.assertNotIn(excluded, names, f"{excluded} is a measure or the time axis")

    def test_high_cardinality_identifier_is_excluded(self):
        names = {c.name for c in discover_group_columns(self.df, self.mappings)}
        self.assertNotIn("Invoice_Number", names,
                         "invoice number is a key, not a business dimension")

    def test_rejected_columns_carry_a_reason(self):
        every = discover_group_columns(self.df, self.mappings, include_unusable=True)
        rejected = [c for c in every if not c.usable]
        self.assertTrue(rejected)
        for column in rejected:
            self.assertTrue(column.reason, f"{column.name} rejected without a reason")

    def test_internal_pipeline_columns_are_not_offered(self):
        df = self.df.copy()
        df["Revenue_outlier_flag"] = 0
        df.loc[df.index[:5], "Revenue_outlier_flag"] = 1
        names = {c.name for c in discover_group_columns(df, self.mappings)}
        self.assertNotIn("Revenue_outlier_flag", names,
                         "cleaner-generated flags are not fields of the source data")

    def test_single_valued_column_is_excluded(self):
        df = self.df.copy()
        df["Company"] = "ACME"
        names = {c.name for c in discover_group_columns(df, self.mappings)}
        self.assertNotIn("Company", names, "one distinct value cannot segment anything")

    def test_cardinality_limit_is_enforced(self):
        df = self.df.copy()
        df["Ref"] = [f"R{i}" for i in range(len(df))]
        usable = discover_group_columns(df, self.mappings)
        self.assertNotIn("Ref", {c.name for c in usable})
        for column in usable:
            self.assertLessEqual(column.cardinality, MAX_CARDINALITY)

    def test_mostly_missing_column_is_excluded(self):
        df = self.df.copy()
        df["Sparse"] = np.nan
        df.loc[df.index[:5], "Sparse"] = "X"
        names = {c.name for c in discover_group_columns(df, self.mappings)}
        self.assertNotIn("Sparse", names)


class TestValueDiscovery(_Base):

    def test_values_come_from_the_data(self):
        values = {v.value for v in discover_group_values(self.df, "Region", self.mappings)}
        self.assertEqual(values, set(REGIONS))

    def test_values_carry_history_context(self):
        for value in discover_group_values(self.df, "Region", self.mappings):
            self.assertGreater(value.n_records, 0)
            self.assertGreater(value.n_periods, 0)
            self.assertIsNotNone(value.first_date)
            self.assertIn(value.history_status,
                          (SUFFICIENT_HISTORY, LIMITED_HISTORY, INSUFFICIENT_HISTORY))

    def test_blank_and_missing_values_are_not_offered_as_segments(self):
        df = self.df.copy()
        df.loc[df.index[:20], "Region"] = np.nan
        df.loc[df.index[20:40], "Region"] = "   "
        values = {v.value for v in discover_group_values(df, "Region", self.mappings)}
        self.assertNotIn("nan", values)
        self.assertNotIn("", values)
        self.assertTrue(values.issubset(set(REGIONS)))

    def test_duplicate_labels_collapse_to_one_segment(self):
        values = discover_group_values(self.df, "Category", self.mappings)
        labels = [v.value for v in values]
        self.assertEqual(len(labels), len(set(labels)), "each value must appear once")

    def test_unknown_column_returns_no_values(self):
        self.assertEqual(discover_group_values(self.df, "Nope", self.mappings), [])

    def test_value_limit_is_respected(self):
        values = discover_group_values(self.df, "Customer_ID", self.mappings, limit=3)
        self.assertLessEqual(len(values), 3)

    def test_metadata_payload_shape(self):
        meta = group_metadata(self.df, self.mappings)
        self.assertIn("group_columns", meta)
        self.assertGreater(meta["n_group_columns"], 0)
        for column in meta["group_columns"]:
            for key in ("name", "label", "cardinality", "values", "usable"):
                self.assertIn(key, column)


# ── history classification ─────────────────────────────────────────────────

class TestHistoryClassification(unittest.TestCase):

    def test_long_history_is_sufficient(self):
        status, note = classify_history(48, horizon=3, frequency="monthly")
        self.assertEqual(status, SUFFICIENT_HISTORY)
        self.assertIn("fold", note)

    def test_very_short_history_is_insufficient(self):
        status, note = classify_history(4, horizon=3, frequency="monthly")
        self.assertEqual(status, INSUFFICIENT_HISTORY)
        self.assertTrue(note)

    def test_middling_history_is_limited_or_sufficient_but_never_crashes(self):
        for n in range(0, 40):
            status, note = classify_history(n, horizon=3, frequency="monthly")
            self.assertIn(status, (SUFFICIENT_HISTORY, LIMITED_HISTORY, INSUFFICIENT_HISTORY))
            self.assertTrue(note)

    def test_classification_is_monotone_in_history_length(self):
        rank = {INSUFFICIENT_HISTORY: 0, LIMITED_HISTORY: 1, SUFFICIENT_HISTORY: 2}
        seen = [rank[classify_history(n, 3, "monthly")[0]] for n in range(2, 60)]
        self.assertEqual(seen, sorted(seen), "more history must never classify as less")


# ── resolution: the two audit bugs ─────────────────────────────────────────

class TestGroupResolution(_Base):

    def test_no_group_resolves_to_company_total(self):
        resolution = resolve_group(self.df, self.mappings)
        self.assertTrue(resolution.ok)
        self.assertTrue(resolution.is_company)
        self.assertIsNone(resolution.frame, "the company path must not copy the frame")
        self.assertEqual(resolution.label, "Company Total")

    def test_unknown_column_is_rejected_not_silently_ignored(self):
        resolution = resolve_group(self.df, self.mappings, "Nonexistent", "X")
        self.assertEqual(resolution.status, GROUP_UNKNOWN_COLUMN)
        self.assertFalse(resolution.ok)
        self.assertIsNone(resolution.frame)
        self.assertIn("not a column", resolution.message)

    def test_unknown_value_is_rejected_with_known_values_listed(self):
        resolution = resolve_group(self.df, self.mappings, "Region", "Atlantis")
        self.assertEqual(resolution.status, GROUP_UNKNOWN_VALUE)
        self.assertIn("Atlantis", resolution.message)
        self.assertTrue(any(r in resolution.message for r in REGIONS))

    def test_column_without_value_is_rejected(self):
        for missing in (None, "", "   "):
            resolution = resolve_group(self.df, self.mappings, "Region", missing)
            self.assertEqual(resolution.status, GROUP_MISSING_VALUE)

    def test_non_groupable_column_is_rejected(self):
        resolution = resolve_group(self.df, self.mappings, "Revenue", "123")
        self.assertEqual(resolution.status, GROUP_NOT_GROUPABLE)
        self.assertIn("cannot be used", resolution.message)

    def test_valid_group_returns_only_its_own_rows(self):
        resolution = resolve_group(self.df, self.mappings, "Region", "West")
        self.assertEqual(resolution.status, GROUP_OK)
        self.assertIsNotNone(resolution.frame)
        self.assertEqual(set(resolution.frame["Region"].unique()), {"West"})
        self.assertEqual(resolution.n_records, len(resolution.frame))
        self.assertLess(len(resolution.frame), len(self.df))

    def test_whitespace_around_a_value_still_matches(self):
        resolution = resolve_group(self.df, self.mappings, "Region", "  West  ")
        self.assertEqual(resolution.status, GROUP_OK)

    def test_label_uses_the_mapped_business_name(self):
        resolution = resolve_group(self.df, self.mappings, "Product_ID", "P001")
        self.assertIn("Product Id", resolution.label)
        self.assertIn("P001", resolution.label)


# ── end-to-end forecasting ─────────────────────────────────────────────────

class TestGroupedForecasting(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.company = run_auto_forecast(cls.df, cls.mappings, horizon=3).to_dict()
        cls.west = run_auto_forecast(cls.df, cls.mappings, horizon=3,
                                     group_col="Region", group_value="West").to_dict()
        cls.p001 = run_auto_forecast(cls.df, cls.mappings, horizon=3,
                                     group_col="Product_ID", group_value="P001").to_dict()

    def test_company_forecast_still_works(self):
        self.assertEqual(self.company["group_status"], GROUP_OK)
        self.assertIsNone(self.company["group_col"])
        self.assertEqual(self.company["group_label"], "Company Total")
        self.assertGreater(_total(self.company), 0)

    def test_region_forecast_works(self):
        self.assertEqual(self.west["group_status"], GROUP_OK)
        self.assertEqual(self.west["group_col"], "Region")
        self.assertEqual(self.west["group_value"], "West")
        self.assertGreater(_total(self.west), 0)

    def test_product_forecast_works(self):
        self.assertEqual(self.p001["group_status"], GROUP_OK)
        self.assertEqual(self.p001["group_value"], "P001")
        self.assertGreater(_total(self.p001), 0)

    def test_segment_forecast_is_smaller_than_the_company(self):
        self.assertLess(_total(self.west), _total(self.company),
                        "one region cannot exceed the whole company")
        self.assertLess(_total(self.p001), _total(self.company))

    def test_company_and_segment_results_never_mix(self):
        # The exact bug this phase fixed: distinct series must yield
        # distinct numbers and distinct labels.
        self.assertNotAlmostEqual(_total(self.west), _total(self.company), places=2)
        self.assertNotEqual(self.west["group_label"], self.company["group_label"])
        self.assertNotEqual(self.p001["group_label"], self.west["group_label"])

    def test_invalid_group_does_not_return_company_numbers(self):
        bad = run_auto_forecast(self.df, self.mappings, horizon=3,
                                group_col="Nonexistent", group_value="X").to_dict()
        self.assertEqual(bad["group_status"], GROUP_UNKNOWN_COLUMN)
        self.assertEqual(bad["forecast"], [])
        self.assertNotAlmostEqual(_total(bad), _total(self.company), places=2)
        self.assertTrue(bad["group_message"])

    def test_empty_group_fails_safely(self):
        bad = run_auto_forecast(self.df, self.mappings, horizon=3,
                                group_col="Region", group_value="Atlantis").to_dict()
        self.assertEqual(bad["group_status"], GROUP_UNKNOWN_VALUE)
        self.assertEqual(bad["forecast"], [])
        self.assertEqual(bad["selected_model"], "None")

    def test_each_segment_validates_independently(self):
        for payload in (self.company, self.west, self.p001):
            self.assertIsNotNone(payload["evaluation_metrics"])
            self.assertGreater((payload["validation"] or {}).get("n_folds", 0), 0)
        # Different series must not coincidentally share an identical WAPE.
        wapes = {round(p["evaluation_metrics"]["wape"], 6)
                 for p in (self.company, self.west, self.p001)}
        self.assertGreater(len(wapes), 1, "segments must be validated on their own data")

    def test_each_segment_selects_its_own_grain(self):
        for payload in (self.company, self.west, self.p001):
            self.assertIn(payload["frequency"],
                          ("daily", "weekly", "monthly", "quarterly"))
            self.assertGreater(payload["n_modelled_periods"], 0)

    def test_each_segment_gets_its_own_intervals(self):
        for payload in (self.company, self.west, self.p001):
            points = payload["forecast"]
            self.assertTrue(points)
            for point in points:
                if point.get("lower_80") is None:
                    continue
                self.assertGreaterEqual(point["lower_80"], 0.0)
                self.assertLessEqual(point["lower_80"], point["value"])
                self.assertLessEqual(point["value"], point["upper_80"])
                self.assertLessEqual(point["interval_width_80"], point["interval_width_95"])

    def test_segment_intervals_are_not_the_company_intervals(self):
        company_width = self.company["forecast"][0].get("interval_width_80")
        west_width = self.west["forecast"][0].get("interval_width_80")
        if company_width and west_width:
            self.assertNotAlmostEqual(company_width, west_width, places=2,
                                      msg="a segment must use its own out-of-sample errors")

    def test_history_status_is_reported_per_segment(self):
        for payload in (self.company, self.west, self.p001):
            self.assertIn(payload["group_history_status"],
                          (SUFFICIENT_HISTORY, LIMITED_HISTORY, INSUFFICIENT_HISTORY))
            self.assertTrue(payload["group_history_note"])

    def test_short_history_segment_does_not_crash(self):
        # P003 exists only in the final few months by construction.
        payload = run_auto_forecast(self.df, self.mappings, horizon=3,
                                    group_col="Product_ID", group_value="P003").to_dict()
        self.assertEqual(payload["group_status"], GROUP_OK)
        self.assertIn(payload["group_history_status"],
                      (LIMITED_HISTORY, INSUFFICIENT_HISTORY, SUFFICIENT_HISTORY))
        for point in payload.get("forecast", []):
            self.assertTrue(np.isfinite(point["value"]))

    def test_category_and_channel_dimensions_also_forecast(self):
        for column, value in (("Category", "Hardware"), ("Sales_Channel", "Direct")):
            payload = run_auto_forecast(self.df, self.mappings, horizon=3,
                                        group_col=column, group_value=value).to_dict()
            self.assertEqual(payload["group_status"], GROUP_OK)
            self.assertGreater(_total(payload), 0)

    def test_different_revenue_scales_are_handled(self):
        big = make_erp(months=48, seed=3, scale=10000.0)
        payload = run_auto_forecast(big, self.mappings, horizon=3,
                                    group_col="Region", group_value="North").to_dict()
        self.assertEqual(payload["group_status"], GROUP_OK)
        self.assertGreater(_total(payload), 0)
        for point in payload["forecast"]:
            self.assertTrue(np.isfinite(point["value"]))

    def test_no_future_leakage_in_segment_validation(self):
        plan = self.west["validation"]
        for fold in plan["folds"]:
            self.assertEqual(fold["train_end"], fold["val_start"],
                             "training must end exactly where validation begins")
            self.assertLess(fold["val_start"], fold["val_end"])


# ── API compatibility ──────────────────────────────────────────────────────

class TestApiCompatibility(_Base):

    LEGACY_FIELDS = [
        "horizon", "selected_model", "selection_reason", "evaluation_metrics",
        "forecast_confidence", "model_ranking", "forecast", "alternative_models",
        "explainability", "warnings", "frequency", "forecast_confidence_pct",
        "data_quality_score", "selected_frequency", "n_modelled_periods",
        "requested_horizon", "forecast_periods", "forecast_unit",
        "validation", "ensemble_evaluation", "prediction_intervals",
        "interval_method", "interval_calibration_status",
    ]

    def test_every_legacy_field_survives(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in self.LEGACY_FIELDS:
            self.assertIn(field, payload)

    def test_group_fields_default_to_none(self):
        payload = AutoForecastResult(
            horizon=3, selected_model="SES", selection_reason=[], evaluation_metrics=None,
            forecast_confidence=0.5, model_ranking=[], forecast=[], alternative_models=[],
            explainability={},
        ).to_dict()
        for field in ("group_col", "group_value", "group_label", "group_status",
                      "group_records", "group_message", "group_history_status"):
            self.assertIn(field, payload)
            self.assertIsNone(payload[field])

    def test_ungrouped_request_matches_pre_phase5_shape(self):
        payload = run_auto_forecast(self.df, self.mappings, horizon=3).to_dict()
        self.assertIsNone(payload["group_col"])
        self.assertIsNone(payload["group_value"])
        for field in self.LEGACY_FIELDS:
            self.assertIn(field, payload)

    def test_payload_is_json_serialisable(self):
        import json
        payload = run_auto_forecast(self.df, self.mappings, horizon=3,
                                    group_col="Region", group_value="East").to_dict()
        self.assertIn("group_label", json.dumps(payload, default=str))

    def test_missing_group_column_in_uploaded_csv_is_reported(self):
        # A CSV that simply lacks the dimension a caller asks for.
        minimal = self.df[["Invoice_Number", "Date", "Revenue"]].copy()
        payload = run_auto_forecast(minimal, self.mappings, horizon=3,
                                    group_col="Region", group_value="West").to_dict()
        self.assertEqual(payload["group_status"], GROUP_UNKNOWN_COLUMN)
        self.assertIn("Region", payload["group_message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
