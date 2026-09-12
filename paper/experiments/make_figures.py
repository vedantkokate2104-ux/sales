"""Generate every figure used by the IEEE paper, as PDF, from the measured
experiment outputs in results/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

e1 = json.load(open(os.path.join(RES, "exp1.json")))
e2 = json.load(open(os.path.join(RES, "exp2.json")))
e3 = json.load(open(os.path.join(RES, "exp3.json")))

COL = "#1f4e79"
COL2 = "#c00000"
COL3 = "#7f7f7f"
COL4 = "#2e7d32"

W1, W2 = 3.4, 7.1

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=190)
   # IEEE single- and double-column widths (inches)


# ── Fig 1 — system architecture ───────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#eef3f9", ec=COL, tc=COL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.8))
        for i, t in enumerate(title):
            ax.text(x + w / 2, y + h - 2.2 - i * 3.4, t, ha="center", va="top",
                    fontsize=7.0, fontweight="bold", color=tc)
        top = y + h - 2.2 - len(title) * 3.4 - 1.4
        for i, t in enumerate(lines):
            ax.text(x + w / 2, top - i * 3.1, t, ha="center", va="top", fontsize=6.0,
                    color="#333333")

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=7, lw=0.9, color="#555555",
                                     shrinkA=0, shrinkB=0))

    W, GAP = 17.0, 3.75
    top_specs = [
        (["A. Schema", "resolution"], ["rules + E5", "embeddings", "column memory"]),
        (["B. Identity", "and cleaning"], ["composite key", "fixed date format", "quality score"]),
        (["C. Grain", "selection"], ["D/W/M/Q scored", "missing is not zero", "leading span only"]),
        (["D. Fold", "planning"], ["one shared plan", "$h$ = production $h$", "stepped origins"]),
        (["E. Per-fold", "backtesting"], ["refit per origin", "in-fold cleaning", "raw actuals"]),
    ]
    for i, (title, lines) in enumerate(top_specs):
        box(i * (W + GAP), 25, W, 20, title, lines)
        if i < 4:
            arrow(i * (W + GAP) + W + 0.4, 35, (i + 1) * (W + GAP) - 0.4, 35)

    bot_specs = [
        (["F. Model selection"], ["WAPE-lexicographic rank", "banded tie-break",
                                  "ensemble adopted only if measured better"]),
        (["G. Conformal intervals"], ["split-conformal relative-error quantiles",
                                      "prequential coverage, Winkler score"]),
        (["H. Reliability report"], ["six weighted evidence factors",
                                     "stated to be not a probability"]),
    ]
    xs = [1.0, 35.0, 69.0]
    ws = [30.0, 30.0, 30.0]
    for (title, lines), x, w in zip(bot_specs, xs, ws):
        box(x, 2, w, 16, title, lines, fc="#f8f2e8", ec="#8a6d3b", tc="#8a6d3b")
    arrow(91.5, 25, 91.5, 18.6)
    arrow(69.0, 10, 65.4, 10, style="<|-")
    arrow(35.0, 10, 31.4, 10, style="<|-")
    ax.text(50, 22.2, "every candidate model is scored on one identical, leakage-free fold plan",
            ha="center", fontsize=6.2, style="italic", color="#666666")
    save(fig, "fig_architecture")
    plt.close(fig)


# ── Fig 2 — series + forecast + intervals ─────────────────────────────────
def fig_forecast():
    ms = e1["dataset"]["monthly_series"]
    periods = list(ms.keys()); values = np.array([ms[k] for k in periods]) / 1e6
    res = e2["auto_forecast"]["result"]
    fc = res["forecast"]
    n = len(values)
    x = np.arange(n)
    fx = np.arange(n, n + len(fc))
    fv = np.array([p["value"] for p in fc]) / 1e6
    lo80 = np.array([p["lower_80"] for p in fc]) / 1e6
    hi80 = np.array([p["upper_80"] for p in fc]) / 1e6
    lo95 = np.array([p["lower_95"] for p in fc]) / 1e6
    hi95 = np.array([p["upper_95"] for p in fc]) / 1e6

    fig, ax = plt.subplots(figsize=(W2, 2.2))
    ax.plot(x, values, color=COL, marker="o", ms=2.2, label="observed monthly net sales")
    ax.axvline(n - 0.5, color=COL3, lw=0.7, ls=":")
    ax.fill_between(fx, lo95, hi95, color=COL2, alpha=0.13, label="95% conformal interval")
    ax.fill_between(fx, lo80, hi80, color=COL2, alpha=0.26, label="80% conformal interval")
    ax.plot(fx, fv, color=COL2, marker="s", ms=2.6, label="forecast (SES, $h=3$)")
    # validation region shading
    ax.axvspan(23.5, n - 0.5, color=COL4, alpha=0.07)
    ax.set_ylim(0, max(values.max(), hi95.max()) * 1.28)
    ax.text(35.5, ax.get_ylim()[1] * 0.055, "walk-forward validation span (origins 24–45)",
            fontsize=6.2, color=COL4, ha="center")
    ticks = list(range(0, n + len(fc), 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([(periods + [p["period"] for p in fc])[t] for t in ticks], rotation=0)
    ax.set_ylabel("net sales (million INR)")
    ax.set_xlabel("period (monthly grain)")
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16),
              handlelength=1.4, columnspacing=1.2)
    save(fig, "fig_forecast")
    plt.close(fig)


# ── Fig 3 — grain diagnostics ─────────────────────────────────────────────
def fig_grain():
    cands = e1["grain"]["candidates"]
    order = ["daily", "weekly", "monthly", "quarterly"]
    dens = [cands[f]["density"] * 100 for f in order]
    cv = [cands[f]["coefficient_of_variation"] for f in order]
    score = [cands[f]["score"] if cands[f]["score"] is not None else 0 for f in order]
    nper = [cands[f]["n_periods"] for f in order]
    best = {"daily": None, "weekly": e3["A_grain"]["weekly"]["best_wape"],
            "monthly": e3["A_grain"]["monthly"]["best_wape"],
            "quarterly": e3["A_grain"]["quarterly"]["best_wape"]}

    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.95))
    idx = np.arange(4)

    ax = axes[0]
    b = ax.bar(idx, dens, color=[COL3, COL3, COL, COL3], width=0.62)
    ax.axhline(60, color=COL2, ls="--", lw=0.8)
    ax.text(3.45, 62, "60% floor", color=COL2, fontsize=6, ha="right")
    ax.set_ylabel("observed-period density (%)"); ax.set_ylim(0, 112)
    for i, v in enumerate(dens):
        ax.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=6)
    ax.set_xticks(idx); ax.set_xticklabels([f"{f[:1].upper()}\n({n})" for f, n in zip(order, nper)])

    ax = axes[1]
    ax.bar(idx, cv, color=[COL3, COL3, COL, COL3], width=0.62)
    ax.set_ylabel("coefficient of variation")
    for i, v in enumerate(cv):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(idx); ax.set_xticklabels([f[:1].upper() for f in order])
    ax.set_ylim(0, max(cv) * 1.22)

    ax = axes[2]
    ax.bar(idx - 0.19, score, width=0.36, color=COL, label="structural score")
    bw = [best[f] / 100 if best[f] else 0 for f in order]
    ax.bar(idx + 0.19, bw, width=0.36, color=COL2, label="best model WAPE")
    ax.set_ylabel("score  /  WAPE (fraction)")
    ax.set_xticks(idx); ax.set_xticklabels([f[:1].upper() for f in order])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0, 0.03, "rejected", rotation=90, fontsize=6, color=COL2, ha="center")
    ax.set_ylim(0, 1.05)
    save(fig, "fig_grain")
    plt.close(fig)


# ── Fig 4 — fold layout ───────────────────────────────────────────────────
def fig_folds():
    plan = e2["benchmark_untuned"]["validation"]
    folds = plan["folds"]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for f in folds:
        y = f["fold"]
        ax.add_patch(Rectangle((0, y - 0.32), f["train_end"], 0.64,
                               fc="#dce6f2", ec=COL, lw=0.4))
        ax.add_patch(Rectangle((f["val_start"], y - 0.32), f["val_end"] - f["val_start"], 0.64,
                               fc="#f6c6c6", ec=COL2, lw=0.4))
    ax.set_xlim(0, 48); ax.set_ylim(0.3, len(folds) + 0.7)
    ax.invert_yaxis()
    ax.set_xlabel("monthly period index"); ax.set_ylabel("validation fold")
    ax.set_yticks(range(1, len(folds) + 1))
    ax.grid(axis="x", alpha=0.25)
    ax.text(11, len(folds) + 0.5, "expanding training window", fontsize=6.2, color=COL, ha="center")
    ax.text(40, len(folds) + 0.5, "scored horizon ($h{=}3$)", fontsize=6.2, color=COL2, ha="center")
    save(fig, "fig_folds")
    plt.close(fig)


# ── Fig 5 — model comparison at monthly grain ─────────────────────────────
def fig_models():
    m = e3["A_grain"]["monthly"]["models"]
    names = sorted(m, key=lambda k: m[k]["wape"])
    w = [m[k]["wape"] for k in names]
    lo = [m[k]["wape"] - m[k]["fold_stats"]["min"] for k in names]
    hi = [m[k]["fold_stats"]["max"] - m[k]["wape"] for k in names]
    stab = [m[k]["stability"] for k in names]
    short = {"Simple Exponential Smoothing": "SES", "Moving Average": "Mov. Avg.",
             "Seasonal Naive": "Seas. Naive", "Holt Linear": "Holt", "Holt-Winters": "Holt-Winters"}
    labels = [short.get(k, k) for k in names]

    fig, ax = plt.subplots(figsize=(W1, 2.05))
    y = np.arange(len(names))
    ax.barh(y, w, xerr=[lo, hi], color=[COL] + [COL3] * (len(names) - 1),
            error_kw=dict(elinewidth=0.6, capsize=1.6, ecolor="#555555"), height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("pooled WAPE (%)  [bars: per-fold min–max]")
    xmax = max(m[k]["fold_stats"]["max"] for k in names)
    for i, (k, v, sv) in enumerate(zip(names, w, stab)):
        ax.text(m[k]["fold_stats"]["max"] + xmax * 0.025, i,
                f"{v:.1f}  ($S$={sv:.2f})", va="center", fontsize=6)
    ax.set_xlim(0, xmax * 1.38)
    save(fig, "fig_models")
    plt.close(fig)


# ── Fig 6 — validation protocol comparison ────────────────────────────────
def fig_protocol():
    P = e3["B_protocol"]
    labels = ["single hold-out\n(1 origin, 3 points)", "walk-forward\n$h{=}1$ (12 origins)",
              "walk-forward\n$h{=}3$ (11 origins, 33 points)"]
    keys = ["single_holdout_h3", "walk_forward_h1", "walk_forward_h3"]
    models = [m for m, _ in P["walk_forward_h3"]["ranking"]]
    short = {"Simple Exponential Smoothing": "SES", "Moving Average": "MA",
             "Seasonal Naive": "SNaive", "Holt Linear": "Holt",
             "Holt-Winters": "HW", "Naive": "Naive", "SARIMA": "SARIMA", "XGBoost": "XGB"}

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.15), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    xw = np.arange(len(models))
    for j, k in enumerate(keys):
        d = dict(P[k]["ranking"])
        ax.bar(xw + (j - 1) * 0.27, [d[m] for m in models], width=0.26,
               color=[COL3, "#9dc3e6", COL][j], label=labels[j].replace("\n", " "))
    ax.set_xticks(xw); ax.set_xticklabels([short[m] for m in models], rotation=30, ha="right")
    ax.set_ylabel("WAPE (%)"); ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    ranks = {k: {m: i + 1 for i, (m, _) in enumerate(P[k]["ranking"])} for k in keys}
    for m in models:
        ys = [ranks[k][m] for k in keys]
        c = COL2 if m == "Seasonal Naive" else (COL if m == "Simple Exponential Smoothing" else COL3)
        lw = 1.5 if c != COL3 else 0.7
        ax.plot([0, 1, 2], ys, marker="o", ms=3, color=c, lw=lw)
        ax.text(2.06, ranks[keys[2]][m], short[m], fontsize=6, va="center", color=c)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["hold-out", "WF $h{=}1$", "WF $h{=}3$"])
    ax.set_ylabel("rank (1 = best)"); ax.invert_yaxis()
    ax.set_yticks(range(1, len(models) + 1)); ax.set_xlim(-0.15, 2.6)
    save(fig, "fig_protocol")
    plt.close(fig)


# ── Fig 7 — feature ablation: training vs validation error ────────────────
def fig_ablation():
    rows = [r for r in e3["D_features"]["results"]
            if r.get("wape") is not None and r.get("training_error") is not None]
    fig, ax = plt.subplots(figsize=(W1, 2.05))
    base = e3["D_features"]["baseline_wape"]
    for r in rows:
        univ = r["features"].startswith("none")
        c = COL if univ else COL2
        mk = "s" if univ else "o"
        ax.scatter(r["training_error"], r["wape"], s=16, facecolor="none",
                   edgecolor=c, marker=mk, lw=0.8)
    lim = [0, max(max(r["training_error"] for r in rows), max(r["wape"] for r in rows)) * 1.08]
    ax.plot(lim, lim, color=COL3, lw=0.6, ls="--")
    ax.axhline(base, color=COL4, lw=0.8, ls=":")
    ax.text(1.5, base - 4.5, f"SES baseline {base:.2f}%", fontsize=6,
            color=COL4, ha="left")
    ax.text(lim[1] * 0.70, lim[1] * 0.755, "training err. = validation err.", fontsize=5.8,
            color=COL3, rotation=36, ha="center")
    ax.set_xlabel("in-sample (training) WAPE (%)")
    ax.set_ylabel("walk-forward validation WAPE (%)")
    ax.set_xlim(0, lim[1]); ax.set_ylim(0, lim[1] * 1.02)
    ax.scatter([], [], s=16, facecolor="none", edgecolor=COL, marker="s",
               label="univariate statistical")
    ax.scatter([], [], s=16, facecolor="none", edgecolor=COL2, marker="o",
               label="feature-based ML")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_ablation")
    plt.close(fig)


# ── Fig 8 — interval calibration and sharpness ────────────────────────────
def fig_intervals():
    pi = e2["auto_forecast"]["result"]["prediction_intervals"]
    cov = pi["coverage"]; sh = pi["sharpness"]
    levels = ["80", "95"]
    nominal = [float(cov[l]["nominal"]) for l in levels]
    observed = [float(cov[l]["observed"]) for l in levels]
    n = [cov[l]["n"] for l in levels]
    se = [100 * np.sqrt(max(o / 100 * (1 - o / 100), 1e-9) / nn) for o, nn in zip(observed, n)]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.95))
    ax = axes[0]
    xw = np.arange(2)
    ax.bar(xw - 0.19, nominal, width=0.36, color=COL3, label="nominal")
    ax.bar(xw + 0.19, observed, width=0.36, color=COL2, yerr=se,
           error_kw=dict(elinewidth=0.7, capsize=2, ecolor="#333333"), label="observed (prequential)")
    ax.set_xticks(xw); ax.set_xticklabels([f"{l}% level\n($n$={nn})" for l, nn in zip(levels, n)])
    ax.set_ylabel("coverage (%)"); ax.set_ylim(0, 105)
    for i, (a, b) in enumerate(zip(nominal, observed)):
        ax.text(i + 0.19, b + se[i] + 2, f"{b:.1f}", ha="center", fontsize=6)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.19),
              handlelength=1.2, columnspacing=1.0)

    ax = axes[1]
    mn = [sh[l]["mean_normalized_width"] for l in levels]
    isc = [sh[l]["interval_score"] / 1e6 for l in levels]
    ax.bar(xw - 0.19, mn, width=0.36, color=COL, label="mean width / point forecast")
    ax2 = ax.twinx()
    ax2.bar(xw + 0.19, isc, width=0.36, color="#b58900", label="Winkler score")
    ax.set_xticks(xw); ax.set_xticklabels([f"{l}%" for l in levels])
    ax.set_ylabel("normalised interval width")
    ax2.set_ylabel("Winkler score (million INR)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.set_ylim(0, max(mn) * 1.35)
    ax2.set_ylim(0, max(isc) * 1.35)
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.19), fontsize=6, handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_intervals")
    plt.close(fig)


for fn in (fig_architecture, fig_forecast, fig_grain, fig_folds, fig_models,
           fig_protocol, fig_ablation, fig_intervals):
    fn()
    print("ok", fn.__name__)
print("figures in", FIG)
