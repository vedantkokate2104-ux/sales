"""
Experiment 1 — dataset characterisation, schema mapping, duplicate identity,
and forecast-grain selection.

Everything here calls the project's own modules; nothing is re-implemented.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJ = r"C:\Users\Vedant Koakte\Downloads\Sales_ERP_claude"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, "src"))

import numpy as np
import pandas as pd

CSV = os.path.join(PROJ, "data", "erp_export.csv")
report = {}

# ── 1. Dataset characterisation ───────────────────────────────────────────
df = pd.read_csv(CSV)
dates = pd.to_datetime(df["inv_dt"], format="%d/%m/%y", errors="coerce")
report["dataset"] = {
    "n_rows": int(len(df)),
    "n_columns": int(len(df.columns)),
    "columns": list(df.columns),
    "date_min": str(dates.min().date()),
    "date_max": str(dates.max().date()),
    "span_days": int((dates.max() - dates.min()).days),
    "n_customers": int(df["cust_cd"].nunique()),
    "n_products": int(df["prod_id"].nunique()),
    "n_regions": int(df["reg_cd"].nunique()),
    "n_salespeople": int(df["slsp"].nunique()),
    "n_product_groups": int(df["prod_grp"].nunique()),
    "total_nsv": float(df["nsv"].sum()),
    "nulls_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    "n_duplicate_invoice_numbers": int(df["inv_no"].duplicated().sum()),
    "n_unique_invoice_numbers": int(df["inv_no"].nunique()),
}

monthly = df.assign(_p=dates.dt.to_period("M")).groupby("_p")["nsv"].sum()
report["dataset"]["monthly_periods_observed"] = int(len(monthly))
report["dataset"]["monthly_cv"] = float(monthly.std() / monthly.mean())
report["dataset"]["monthly_series"] = {str(k): float(v) for k, v in monthly.items()}

# ── 2. Schema mapping (cold: no persisted memory) ─────────────────────────
import tempfile, shutil
tmp = tempfile.mkdtemp()
cwd = os.getcwd()
os.chdir(tmp)          # schema_mapper persists memory to ./column_memory.json
try:
    import importlib
    import schema_mapper, data_loader
    importlib.reload(schema_mapper)
    columns = data_loader.extract_column_info(df)
    t0 = time.time()
    mappings = schema_mapper.map_all_columns(columns)
    map_seconds = time.time() - t0
finally:
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)

report["schema_mapping"] = {
    "seconds": round(map_seconds, 2),
    "n_columns": len(mappings),
    "rows": [
        {
            "column": m.get("original_column"),
            "business_name": m.get("business_name"),
            "category": m.get("category"),
            "confidence": m.get("confidence"),
            "source": m.get("source"),
        }
        for m in mappings
    ],
}
conf = [m.get("confidence", 0.0) for m in mappings]
report["schema_mapping"]["n_high_0.75"] = int(sum(1 for c in conf if c >= 0.75))
report["schema_mapping"]["n_medium"] = int(sum(1 for c in conf if 0.60 <= c < 0.75))
report["schema_mapping"]["n_low"] = int(sum(1 for c in conf if c < 0.60))
report["schema_mapping"]["mean_confidence"] = round(float(np.mean(conf)), 4)
by_source = {}
for m in mappings:
    by_source[m.get("source", "?")] = by_source.get(m.get("source", "?"), 0) + 1
report["schema_mapping"]["by_source"] = by_source

# ── 3. Transaction identity / duplicate handling ──────────────────────────
import transaction_key
key_cols, basis, limitation = transaction_key.resolve_business_key(df, mappings)
dup = transaction_key.analyze_duplicates(df, mappings)
report["duplicates"] = {
    "business_key_columns": key_cols,
    "basis": basis,
    "limitation": limitation,
    "analysis": dup,
}

# naive single-identifier deduplication (the pre-Phase-1 behaviour), measured
id_col = "inv_no"
naive_kept = df.drop_duplicates(subset=[id_col], keep="first")
report["duplicates"]["naive_identifier_dedup"] = {
    "column": id_col,
    "rows_before": int(len(df)),
    "rows_after": int(len(naive_kept)),
    "rows_destroyed": int(len(df) - len(naive_kept)),
    "pct_rows_destroyed": round(100 * (len(df) - len(naive_kept)) / len(df), 2),
    "revenue_before": float(df["nsv"].sum()),
    "revenue_after": float(naive_kept["nsv"].sum()),
    "pct_revenue_destroyed": round(
        100 * (df["nsv"].sum() - naive_kept["nsv"].sum()) / df["nsv"].sum(), 2),
}

# ── 4. Forecast-grain selection ───────────────────────────────────────────
from forecast.grain import select_forecast_grain, build_forecast_series
from forecast.utils import infer_frequency
from forecast.preprocessing import parse_dates_deterministic

parsed = parse_dates_deterministic(df["inv_dt"])
report["grain"] = {
    "inter_arrival_inferred_frequency": infer_frequency(parsed.dropna()),
}
sel = select_forecast_grain(df["inv_dt"], df["nsv"], horizon=3)
report["grain"]["selected"] = sel.selected_frequency
report["grain"]["reason"] = sel.reason
report["grain"]["selection_basis"] = sel.selection_basis
report["grain"]["fallback"] = sel.fallback
report["grain"]["candidates"] = {k: v.to_dict() for k, v in sel.candidate_frequencies.items()}

for freq in ("daily", "weekly", "monthly", "quarterly"):
    c = build_forecast_series(df["inv_dt"], df["nsv"], freq)
    report["grain"].setdefault("construction", {})[freq] = c.to_dict()

# series stats at the selected grain
from forecast.analyzer import prepare_forecast_series
from forecast.classifier import classify_series
from forecast.model_selector import recommend_models

construction, selection, stats = prepare_forecast_series(
    df["inv_dt"], df["nsv"], horizon=3, group="ALL")
cls = classify_series(stats)
recs = recommend_models(cls, stats)
report["series_stats"] = stats.to_dict()
report["classification"] = cls.to_dict()
report["recommended_candidates"] = [r.to_dict() for r in recs]

# raw-transaction-level stats (the pre-Phase-1 behaviour), for contrast
from forecast.analyzer import analyze_series
raw_stats = analyze_series(df["inv_dt"], df["nsv"])
report["series_stats_transaction_level"] = raw_stats.to_dict()
raw_cls = classify_series(raw_stats)
report["classification_transaction_level"] = raw_cls.to_dict()
report["recommended_candidates_transaction_level"] = [
    r.to_dict() for r in recommend_models(raw_cls, raw_stats)]

with open(os.path.join(OUT, "exp1.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "schema_mapping"}, indent=2, default=str)[:8000])
print("\n--- schema mapping ---")
print(json.dumps(report["schema_mapping"], indent=2, default=str))
