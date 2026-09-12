"""
Experiment 3 — controlled ablations, all driven through the project's own
backtest engine so the protocol is identical across every row.

  A. Forecast grain            weekly (inter-arrival) vs monthly (selected)
  B. Validation protocol       single hold-out vs walk-forward; h=1 vs h=3
  C. Preprocessing placement   whole-series capping (leaky) vs per-fold
  D. Feature ablation ladder   forecast.experiments.run_experiments()
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJ = r"C:\Users\Vedant Koakte\Downloads\Sales_ERP_claude"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, "src"))

import numpy as np
import pandas as pd

from forecast.backtest import (
    FoldSpec, ValidationPlan, backtest_model, plan_validation, prepare_training_window,
)
from forecast.grain import build_forecast_series
from forecast.models import MODEL_REGISTRY
from forecast.utils import seasonal_period_for_freq

df = pd.read_csv(os.path.join(PROJ, "data", "erp_export.csv"))
report = {}

CLASSICAL = ["Simple Exponential Smoothing", "Naive", "Moving Average", "Holt Linear",
             "Seasonal Naive", "Holt-Winters", "SARIMA", "XGBoost"]


def run_all(construction, plan, freq, names=CLASSICAL):
    rows = {}
    for name in names:
        cls = MODEL_REGISTRY.get(name)
        if cls is None or not cls.is_available():
            continue
        try:
            ok, msg = cls().check_requirements(construction.series, freq)
        except Exception as exc:
            ok, msg = False, str(exc)
        if not ok:
            rows[name] = {"skipped": msg}
            continue
        res = backtest_model(construction.series, construction.observed_mask, cls,
                             plan, freq, model_name=name)
        rows[name] = {
            "wape": (res.pooled_metrics or {}).get("wape"),
            "mae": (res.pooled_metrics or {}).get("mae"),
            "rmse": (res.pooled_metrics or {}).get("rmse"),
            "smape": (res.pooled_metrics or {}).get("smape"),
            "stability": res.stability,
            "n_folds": res.n_folds,
            "n_points": res.n_points,
            "fold_stats": res.fold_stats,
        }
    return rows


# ── A. Grain ablation ─────────────────────────────────────────────────────
grain_rows = {}
for freq in ("weekly", "monthly", "quarterly"):
    c = build_forecast_series(df["inv_dt"], df["nsv"], freq)
    p = plan_validation(len(c.series), 3, seasonal_period_for_freq(freq))
    grain_rows[freq] = {
        "n_periods": len(c.series),
        "density": c.density,
        "fill_strategy": c.fill_strategy,
        "plan": {"n_folds": p.n_folds, "horizon": p.horizon,
                 "min_train": p.min_train_size, "step": p.step,
                 "n_validation_points": p.n_validation_points,
                 "degraded": p.degraded},
        "models": run_all(c, p, freq),
    }
    ok = {k: v["wape"] for k, v in grain_rows[freq]["models"].items()
          if isinstance(v.get("wape"), (int, float))}
    grain_rows[freq]["best_model"] = min(ok, key=ok.get) if ok else None
    grain_rows[freq]["best_wape"] = min(ok.values()) if ok else None
    grain_rows[freq]["median_wape"] = float(np.median(list(ok.values()))) if ok else None
    print(freq, grain_rows[freq]["best_model"], grain_rows[freq]["best_wape"])
report["A_grain"] = grain_rows

# monthly construction reused below
monthly = build_forecast_series(df["inv_dt"], df["nsv"], "monthly")
n = len(monthly.series)
SP = seasonal_period_for_freq("monthly")

# ── B. Validation protocol ablation ───────────────────────────────────────
protocols = {}

# B1 — single trailing hold-out of 3 periods (one origin)
single = ValidationPlan(
    horizon=3, requested_horizon=3, min_train_size=n - 3, step=1,
    folds=[FoldSpec(index=1, train_end=n - 3, val_start=n - 3, val_end=n)],
    rule="single trailing hold-out of 3 periods (one origin)",
)
protocols["single_holdout_h3"] = {"plan": single.to_dict(), "models": run_all(monthly, single, "monthly")}

# B2 — walk-forward, one step ahead at every origin
wf1 = plan_validation(n, 1, SP)
protocols["walk_forward_h1"] = {"plan": wf1.to_dict(), "models": run_all(monthly, wf1, "monthly")}

# B3 — walk-forward at the production horizon (the deployed protocol)
wf3 = plan_validation(n, 3, SP)
protocols["walk_forward_h3"] = {"plan": wf3.to_dict(), "models": run_all(monthly, wf3, "monthly")}

for k, v in protocols.items():
    ok = {m: r["wape"] for m, r in v["models"].items() if isinstance(r.get("wape"), (int, float))}
    v["best_model"] = min(ok, key=ok.get) if ok else None
    v["best_wape"] = min(ok.values()) if ok else None
    v["ranking"] = sorted(ok.items(), key=lambda kv: kv[1])
    print(k, v["best_model"], v["best_wape"], v["plan"]["n_folds"], "folds")
report["B_protocol"] = protocols

# ── C. Preprocessing-placement ablation ───────────────────────────────────
# LEAKY: IQR capping bounds fitted on the WHOLE series (including the periods
# each fold is later scored against), exactly the pre-Phase-2 behaviour.
# HONEST: bounds refitted inside every fold on that fold's training window.
leaky_values, leaky_pre = prepare_training_window(
    monthly.series.to_numpy(dtype=float), monthly.observed_mask.to_numpy(dtype=bool))
leaky_series = pd.Series(leaky_values, index=monthly.series.index)


class _Wrapped:
    def __init__(self, series, mask):
        self.series, self.observed_mask = series, mask


leaky_construction = _Wrapped(leaky_series, monthly.observed_mask)
report["C_preprocessing"] = {
    "n_capped_whole_series": leaky_pre.n_capped,
    "cap_bounds_whole_series": [leaky_pre.lower, leaky_pre.upper],
    "honest_per_fold": {m: r for m, r in protocols["walk_forward_h3"]["models"].items()},
    "leaky_whole_series": run_all(leaky_construction, wf3, "monthly"),
}
h = {m: v["wape"] for m, v in report["C_preprocessing"]["honest_per_fold"].items()
     if isinstance(v.get("wape"), (int, float))}
l = {m: v["wape"] for m, v in report["C_preprocessing"]["leaky_whole_series"].items()
     if isinstance(v.get("wape"), (int, float))}
report["C_preprocessing"]["delta_pp"] = {m: round(h[m] - l[m], 3) for m in h if m in l}
print("C delta:", report["C_preprocessing"]["delta_pp"])

# ── D. Feature ablation ladder ────────────────────────────────────────────
from forecast.experiments import run_experiments, format_report

t0 = time.time()
exp = run_experiments(monthly, horizon=3)
report["D_features"] = exp.to_dict()
report["D_features_text"] = format_report(exp)
report["D_features"]["wall_seconds"] = round(time.time() - t0, 2)
print(format_report(exp))

with open(os.path.join(OUT, "exp3.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print("written")
