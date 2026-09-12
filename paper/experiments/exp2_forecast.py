"""
Experiment 2 — end-to-end auto-forecast, full-registry benchmark,
prediction-interval calibration, reliability assessment.
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
df = pd.read_csv(CSV)

# schema mappings from the persisted project memory (what the app uses)
with open(os.path.join(PROJ, "column_memory.json")) as f:
    memory = json.load(f)
mappings = [memory[c] for c in df.columns if c in memory]
print("mapped columns:", len(mappings))

report = {}

# ── A. End-to-end auto forecast (the production path) ─────────────────────
from forecast.forecast_engine import run_auto_forecast

t0 = time.time()
result = run_auto_forecast(df, mappings, horizon=3)
elapsed = time.time() - t0
payload = result.to_dict() if hasattr(result, "to_dict") else result.__dict__
report["auto_forecast"] = {"runtime_seconds": round(elapsed, 2), "result": payload}
print(f"auto forecast done in {elapsed:.1f}s -> {payload.get('selected_model')}")

# ── B. Full-registry benchmark under the shared fold plan ─────────────────
from forecast.benchmark import benchmark_dataframe, format_report

t0 = time.time()
bench = benchmark_dataframe(df, mappings, horizon=3, tune=False)
bench_seconds = time.time() - t0
report["benchmark_untuned"] = bench.to_dict()
report["benchmark_untuned"]["wall_seconds"] = round(bench_seconds, 2)
report["benchmark_untuned_text"] = format_report(bench)
print(format_report(bench))

t0 = time.time()
bench_t = benchmark_dataframe(df, mappings, horizon=3, tune=True)
report["benchmark_tuned"] = bench_t.to_dict()
report["benchmark_tuned"]["wall_seconds"] = round(time.time() - t0, 2)
report["benchmark_tuned_text"] = format_report(bench_t)
print("\n=== TUNED ===")
print(format_report(bench_t))

with open(os.path.join(OUT, "exp2.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nwritten")
