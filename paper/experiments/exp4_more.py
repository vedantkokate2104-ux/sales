"""
Experiment 4 —
  E. Rank agreement between validation protocols (Spearman/Kendall).
  F. Leakage ablation where preprocessing is actually active
     (weekly grain: 32 unobserved periods; monthly grain with a tightened
     IQR multiplier so capping fires).
  G. Metric behaviour on a sparse grain: epsilon-guarded MAPE vs the
     project's zero-excluded MAPE vs WAPE.
  H. Segment-level forecasting across every discovered business dimension.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJ = r"C:\Users\Vedant Koakte\Downloads\Sales_ERP_claude"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, "src"))

import numpy as np
import pandas as pd
from scipy import stats as sps

from forecast.backtest import (
    FoldPreprocessor, backtest_model, plan_validation, prepare_training_window,
)
from forecast.grain import build_forecast_series
from forecast.models import MODEL_REGISTRY
from forecast.utils import seasonal_period_for_freq

df = pd.read_csv(os.path.join(PROJ, "data", "erp_export.csv"))
with open(os.path.join(PROJ, "column_memory.json")) as f:
    memory = json.load(f)
mappings = [memory[c] for c in df.columns if c in memory]

report = {}
prev = json.load(open(os.path.join(OUT, "exp3.json")))

MODELS = ["Simple Exponential Smoothing", "Naive", "Moving Average", "Holt Linear",
          "Seasonal Naive", "Holt-Winters", "SARIMA", "XGBoost"]


# ── E. Rank agreement between protocols ───────────────────────────────────
def order(d):
    return [m for m, _ in d]


ranks = {k: dict((m, i + 1) for i, (m, _) in enumerate(v["ranking"]))
         for k, v in prev["B_protocol"].items()}
names = MODELS
E = {}
for a, b in (("single_holdout_h3", "walk_forward_h3"), ("walk_forward_h1", "walk_forward_h3")):
    ra = [ranks[a][m] for m in names]
    rb = [ranks[b][m] for m in names]
    rho, p_rho = sps.spearmanr(ra, rb)
    tau, p_tau = sps.kendalltau(ra, rb)
    E[f"{a}_vs_{b}"] = {
        "spearman_rho": round(float(rho), 4), "spearman_p": round(float(p_rho), 4),
        "kendall_tau": round(float(tau), 4), "kendall_p": round(float(p_tau), 4),
        "top1_a": order(prev["B_protocol"][a]["ranking"])[0],
        "top1_b": order(prev["B_protocol"][b]["ranking"])[0],
        "ranks_a": dict(zip(names, ra)), "ranks_b": dict(zip(names, rb)),
    }
report["E_rank_agreement"] = E
print(json.dumps(E, indent=1))


# ── F. Leakage ablation where preprocessing is active ─────────────────────
class _W:
    def __init__(self, s, m):
        self.series, self.observed_mask = s, m


def run(construction, plan, freq, names=MODELS):
    out = {}
    for name in names:
        cls = MODEL_REGISTRY.get(name)
        if cls is None or not cls.is_available():
            continue
        try:
            ok, _ = cls().check_requirements(construction.series, freq)
        except Exception:
            ok = False
        if not ok:
            continue
        res = backtest_model(construction.series, construction.observed_mask, cls,
                             plan, freq, model_name=name)
        out[name] = (res.pooled_metrics or {}).get("wape")
    return out


F = {}

# F1 — weekly grain: 32 unobserved periods, so gap bridging is active.
wk = build_forecast_series(df["inv_dt"], df["nsv"], "weekly")
wk_plan = plan_validation(len(wk.series), 3, seasonal_period_for_freq("weekly"))
honest_wk = run(wk, wk_plan, "weekly")
leaky_vals, leaky_pre = prepare_training_window(
    wk.series.to_numpy(dtype=float), wk.observed_mask.to_numpy(dtype=bool))
leaky_wk = run(_W(pd.Series(leaky_vals, index=wk.series.index), wk.observed_mask),
               wk_plan, "weekly")
F["weekly"] = {
    "n_periods": len(wk.series), "n_unobserved": int((~wk.observed_mask).sum()),
    "n_capped_whole_series": leaky_pre.n_capped,
    "honest_per_fold": honest_wk, "leaky_whole_series": leaky_wk,
    "delta_pp": {m: round(honest_wk[m] - leaky_wk[m], 3) for m in honest_wk if m in leaky_wk},
}
print("F weekly delta:", F["weekly"]["delta_pp"], "capped:", leaky_pre.n_capped)

# F2 — monthly grain with the IQR multiplier tightened to 1.5 so that
# capping actually fires. This is a SENSITIVITY ANALYSIS, not the deployed
# setting (the engine uses 3.0), stated as such.
mo = build_forecast_series(df["inv_dt"], df["nsv"], "monthly")
mo_plan = plan_validation(len(mo.series), 3, seasonal_period_for_freq("monthly"))
vals = mo.series.to_numpy(dtype=float)
mask = mo.observed_mask.to_numpy(dtype=bool)
pre15 = FoldPreprocessor.fit(vals, multiplier=1.5)
capped15 = pre15.transform(vals)
F["monthly_iqr1.5"] = {
    "n_capped_whole_series": int(pre15.n_capped),
    "bounds": [pre15.lower, pre15.upper],
    "honest_per_fold_iqr3": {m: v["wape"] for m, v in prev["B_protocol"]["walk_forward_h3"]["models"].items()},
    "leaky_whole_series_iqr1.5": run(_W(pd.Series(capped15, index=mo.series.index), mo.observed_mask),
                                     mo_plan, "monthly"),
}
# honest counterpart at 1.5x: cap inside each fold using that fold's own bounds
import forecast.backtest as bt
_orig = bt.OUTLIER_IQR_MULTIPLIER
bt.OUTLIER_IQR_MULTIPLIER = 1.5
F["monthly_iqr1.5"]["honest_per_fold_iqr1.5"] = run(mo, mo_plan, "monthly")
bt.OUTLIER_IQR_MULTIPLIER = _orig
h = F["monthly_iqr1.5"]["honest_per_fold_iqr1.5"]
l = F["monthly_iqr1.5"]["leaky_whole_series_iqr1.5"]
F["monthly_iqr1.5"]["delta_pp"] = {m: round(h[m] - l[m], 3) for m in h if m in l}
print("F monthly1.5 capped:", pre15.n_capped, "delta:", F["monthly_iqr1.5"]["delta_pp"])
report["F_leakage"] = F


# ── G. Metric behaviour on a sparse grain ─────────────────────────────────
from forecast.evaluator import compute_metrics, wape as wape_fn, mape as mape_fn

daily = build_forecast_series(df["inv_dt"], df["nsv"], "daily")
d_plan = plan_validation(len(daily.series), 3, seasonal_period_for_freq("daily"))


def epsilon_mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


G = {"n_periods": len(daily.series),
     "zero_period_share_pct": round(float((daily.series.to_numpy() == 0).mean() * 100), 2),
     "plan": {"n_folds": d_plan.n_folds, "horizon": d_plan.horizon,
              "min_train": d_plan.min_train_size, "step": d_plan.step},
     "models": {}}
for name in ["Naive", "Moving Average", "Simple Exponential Smoothing"]:
    cls = MODEL_REGISTRY[name]
    res = backtest_model(daily.series, daily.observed_mask, cls, d_plan, "daily", model_name=name)
    if not res.folds:
        continue
    a = np.concatenate([np.array(f.actuals) for f in res.folds])
    p = np.concatenate([np.array(f.predictions) for f in res.folds])
    G["models"][name] = {
        "n_points": int(len(a)),
        "n_zero_actuals": int((a == 0).sum()),
        "epsilon_guarded_mape_pct": epsilon_mape(a, p),
        "project_mape_pct": mape_fn(a, p),
        "wape_pct": wape_fn(a, p),
        "smape_pct": compute_metrics(a, p)["smape"],
    }
report["G_metrics"] = G
print("G:", json.dumps(G["models"], indent=1, default=str))


# ── H. Segment-level forecasting ──────────────────────────────────────────
from forecast.segments import discover_group_columns, discover_group_values
from forecast.forecast_engine import run_auto_forecast

cols = discover_group_columns(df, mappings)
report["H_segments"] = {"discovered": [c.to_dict() for c in cols], "forecasts": []}
print("\ndiscovered dimensions:", [(c.name, c.cardinality, c.usable) for c in cols])

for col_name, values in (("reg_cd", None), ("prod_grp", None), ("cust_seg", None)):
    if col_name not in df.columns:
        continue
    for value in sorted(df[col_name].dropna().unique()):
        t0 = time.time()
        try:
            res = run_auto_forecast(df, mappings, horizon=3,
                                    group_col=col_name, group_value=str(value))
            p = res.to_dict()
            report["H_segments"]["forecasts"].append({
                "dimension": col_name, "value": str(value),
                "n_records": p.get("group_records"),
                "history_status": p.get("group_history_status"),
                "frequency": p.get("selected_frequency"),
                "n_periods": p.get("n_modelled_periods"),
                "selected_model": p.get("selected_model"),
                "wape": (p.get("evaluation_metrics") or {}).get("wape"),
                "confidence_pct": p.get("forecast_confidence_pct"),
                "reliability_level": (p.get("reliability") or {}).get("level"),
                "reliability_score": (p.get("reliability") or {}).get("score"),
                "coverage_80": p.get("coverage_80"),
                "interval_status": p.get("interval_calibration_status"),
                "seconds": round(time.time() - t0, 1),
            })
            print(" ", col_name, value, p.get("selected_model"),
                  (p.get("evaluation_metrics") or {}).get("wape"),
                  p.get("group_history_status"))
        except Exception as exc:
            report["H_segments"]["forecasts"].append(
                {"dimension": col_name, "value": str(value), "error": str(exc)})
            print("  ERROR", col_name, value, exc)

with open(os.path.join(OUT, "exp4.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print("written")
