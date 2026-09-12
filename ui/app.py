"""
app.py — Flask backend for the Sales Analysis Agent UI
Run with: python app.py
Opens on: http://localhost:5000
"""

import os

# Prevent OpenMP / XGBoost thread crashes on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, request, jsonify
import pandas as pd

from data_loader   import load_erp_data, extract_column_info
from schema_mapper import map_all_columns, correct_mapping
from data_cleaner  import clean
from analytics     import calculate_kpis, detect_anomalies, run_forecast
from query_engine  import answer
from forecast.analyzer       import analyze_primary, analyze_primary_with_grain
from forecast.classifier     import classify_series
from forecast.model_selector import recommend_models
from forecast.forecast_engine import run_auto_forecast
from forecast.segments      import group_metadata

app = Flask(__name__)

# ── global state (held in memory for the POC session) ─────────────────────────
STATE = {
    "df":       None,
    "df_clean": None,
    "mappings": None,
    "kpis":     None,
    "anomalies":None,
    "forecast": None,
    "report":   None,
    "loaded":   False,
    "auto_forecast": None,   # Phase 2: cached AutoForecastResult (dict), set by /api/forecast/auto
    # Phase 5: one cached result PER (horizon, group_col, group_value) so a
    # segment forecast can never be served under another segment's label.
    "auto_forecast_cache": {},
}

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'erp_export.csv')


def _run_pipeline():
    """Run the full backend pipeline and populate STATE."""
    df       = load_erp_data(DATA_PATH)
    columns  = extract_column_info(df)
    mappings = map_all_columns(columns)
    df_clean, report = clean(df, mappings)
    kpis     = calculate_kpis(df_clean, mappings)
    anomalies= detect_anomalies(df_clean, mappings)
    forecast = run_forecast(df_clean, mappings, months=3)

    STATE.update({
        "df": df, "df_clean": df_clean, "mappings": mappings,
        "kpis": kpis, "anomalies": anomalies, "forecast": forecast,
        "report": report, "loaded": True,
        "auto_forecast": None, "auto_forecast_cache": {},
    })


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load", methods=["POST"])
def api_load():
    try:
        _run_pipeline()
        m = STATE["mappings"]
        r = STATE["report"]
        return jsonify({
            "ok": True,
            "rows":     len(STATE["df"]),
            "columns":  len(STATE["df"].columns),
            "rows_clean": r["rows_after"],
            "dupes":    r["duplicates_removed"],
            "nulls":    sum(v["null_count"] for v in extract_column_info(STATE["df"])),
            "mappings": [
                {
                    "col":        mp.get("original_column", ""),
                    "name":       mp.get("business_name", ""),
                    "category":   mp.get("category", ""),
                    "confidence": mp.get("confidence", 0.0),
                    "source":     mp.get("source", ""),
                    "reasons":    mp.get("reasons", [])
                }
                for mp in m
            ],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kpis")
def api_kpis():
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    k = STATE["kpis"]
    return jsonify({
        "ok": True,
        "total_revenue":    k.get("total_revenue", 0),
        "avg_order_value":  k.get("avg_order_value", 0),
        "unique_customers": k.get("unique_customers", 0),
        "total_units":      k.get("total_units_sold", 0),
        "total_discount":   k.get("total_discount_given", 0),
        "disc_pct":         k.get("discount_as_pct_of_revenue", 0),
        "overdue_amount":   k.get("overdue_amount", 0),
        "revenue_by_region":   k.get("revenue_by_region", {}),
        "revenue_by_category": k.get("revenue_by_category", {}),
        "monthly_revenue":     k.get("monthly_revenue", {}),
        "top_customers":       k.get("top_5_customers", {}),
        "payment_status":      k.get("payment_status_breakdown", {}),
    })


@app.route("/api/anomalies")
def api_anomalies():
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    return jsonify({"ok": True, "anomalies": STATE["anomalies"]})


@app.route("/api/forecast", methods=["GET", "POST"])
def api_forecast():
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400

    if request.method == "POST":
        data = request.json or {}
        months = int(data.get("months", 3))
        # Fast path: re-run forecast only without mapping/cleaning data again
        STATE["forecast"] = run_forecast(STATE["df_clean"], STATE["mappings"], months=months)

    fc = STATE["forecast"]
    if "error" in fc:
        return jsonify({"ok": False, "error": fc.get("error")}), 400

    return jsonify({
        "ok": True,
        "forecast": [{"month": m, "value": v} for m, v in fc.items()]
    })


@app.route("/api/forecast/analysis")
def api_forecast_analysis():
    """
    Forecast Intelligence Engine (Phase 1): dataset analysis, series
    classification, and candidate model recommendations for the Forecast
    page's "Forecast Analysis" panel. Advisory only — this endpoint does
    not affect /api/forecast's existing output or the active forecaster
    (analytics.run_forecast), which are unchanged.
    """
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    try:
        # analyze_primary_with_grain() returns the same SeriesStats
        # analyze_primary() does, plus the grain-selection record — so the
        # existing "stats"/"classification"/"recommendations" keys are
        # unchanged and "grain" is purely additive (Phase 1, Step 9).
        _construction, grain_selection, stats = analyze_primary_with_grain(
            STATE["df_clean"], STATE["mappings"]
        )
        classification = classify_series(stats)
        recommendations = recommend_models(classification, stats)
        return jsonify({
            "ok": True,
            "stats": stats.to_dict(),
            "classification": classification.to_dict(),
            "recommendations": [r.to_dict() for r in recommendations],
            "grain": grain_selection.to_dict(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/forecast/auto", methods=["GET", "POST"])
def api_forecast_auto():
    """
    Auto Forecasting Engine (Phase 2): trains every candidate model Phase 1
    recommends (plus guaranteed baselines), evaluates them on a
    time-ordered validation split, automatically selects the
    best-performing model, and generates the forecast. Cached in
    STATE["auto_forecast"] so GET can re-serve the last result (e.g. after
    a page navigation) without retraining. Does not affect /api/forecast
    or /api/forecast/analysis, both unchanged from Phase 1.
    """
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400

    # Phase 5: group_col/group_value are optional and additive. Omitting
    # them runs the company total exactly as before, so every existing
    # client is unaffected.
    if request.method == "POST":
        data = request.json or {}
        months = int(data.get("months", 3))
        group_col = (data.get("group_col") or None)
        group_value = (data.get("group_value") or None)
    else:
        months = int(request.args.get("months", 3))
        group_col = request.args.get("group_col") or None
        group_value = request.args.get("group_value") or None

    cache_key = _auto_cache_key(months, group_col, group_value)

    if request.method == "POST":
        try:
            result = run_auto_forecast(
                STATE["df_clean"], STATE["mappings"], horizon=months,
                group_col=group_col, group_value=group_value,
            )
            payload = result.to_dict()
            # Cached PER SEGMENT. A single shared slot would let a GET
            # after a segment run serve one series under another's label,
            # which is precisely the company/segment mix-up Phase 5 has to
            # rule out.
            STATE["auto_forecast_cache"][cache_key] = payload
            STATE["auto_forecast"] = payload
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    cached = STATE["auto_forecast_cache"].get(cache_key)
    if cached is None:
        # Fall back to the single-slot cache only for an ungrouped request,
        # preserving the pre-Phase-5 GET behaviour exactly.
        if group_col is None and STATE["auto_forecast"] is not None:
            cached = STATE["auto_forecast"]
        else:
            return jsonify({
                "ok": False,
                "error": "Auto forecast has not been run yet for this selection",
            }), 400

    return jsonify({"ok": True, **cached})


def _auto_cache_key(months, group_col, group_value):
    return f"{months}|{group_col or ''}|{group_value or ''}"


@app.route("/api/forecast/groups")
def api_forecast_groups():
    """
    Phase 5, Step 13 — the grouping dimensions available in the CURRENTLY
    LOADED dataset, and the values of each, so the UI can offer a Forecast
    Level selector without anything being hard-coded.

    Deliberately lightweight: value counts and date ranges only. No model
    is trained here, so a dataset with many segments costs one pass rather
    than one forecast per segment (Step 17).
    """
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    try:
        frequency = "monthly"
        auto = STATE.get("auto_forecast")
        if auto and auto.get("frequency"):
            frequency = auto["frequency"]
        meta = group_metadata(STATE["df_clean"], STATE["mappings"], frequency=frequency)
        return jsonify({"ok": True, **meta})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def api_query():
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    q = request.json.get("question", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Empty question"}), 400
    # strip color codes for web output
    import re
    raw = answer(q, STATE["df_clean"], STATE["mappings"], STATE["kpis"])
    clean_ans = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    return jsonify({"ok": True, "answer": clean_ans})


@app.route("/api/correct", methods=["POST"])
def api_correct():
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    col  = request.json.get("column", "").strip()
    name = request.json.get("name", "").strip()
    if not col or not name:
        return jsonify({"ok": False, "error": "Missing column or name"}), 400
    correct_mapping(col, name)
    # refresh mappings
    columns  = extract_column_info(STATE["df"])
    STATE["mappings"] = map_all_columns(columns)
    return jsonify({"ok": True, "message": f"'{col}' updated to '{name}'"})

@app.route("/api/upload", methods=["POST"])
def api_upload():

    if "file" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "ok": False,
            "error": "No file selected"
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "ok": False,
            "error": "Only CSV files are allowed"
        }), 400

    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "erp_export.csv"
    )

    file.save(save_path)

    # Invalidate every cached forecast. Without this a GET after an
    # upload re-serves the PREVIOUS dataset's forecast, since
    # STATE["loaded"] = False alone only forces a reload of the frame.
    STATE["loaded"] = False
    STATE["auto_forecast"] = None
    STATE["auto_forecast_cache"] = {}

    return jsonify({
        "ok": True,
        "message": "CSV uploaded successfully"
    })

@app.route("/api/duplicates")
def api_duplicates():
    """Return duplicate rows from the raw dataframe before cleaning."""
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400
    df = STATE["df"]
    m  = STATE["mappings"]

    # Find the identifier column (invoice number)
    id_col = None
    for mp in m:
        if mp["category"] == "identifier":
            id_col = mp["original_column"]
            break

    if id_col is None or id_col not in df.columns:
        return jsonify({"ok": True, "duplicates": [], "id_col": None,
                        "columns": list(df.columns[:8])})

    # Rows where the identifier appears more than once
    dup_mask = df.duplicated(subset=[id_col], keep=False)
    dup_df   = df[dup_mask].copy()

    # Return first 8 columns + id_col to keep payload small
    show_cols = [id_col] + [c for c in df.columns if c != id_col][:7]
    dup_df = dup_df[show_cols].fillna("—")

    return jsonify({
        "ok":         True,
        "id_col":     id_col,
        "columns":    show_cols,
        "duplicates": dup_df.to_dict(orient="records"),
        "total":      int(dup_mask.sum()),
    })


@app.route("/api/duplicates/delete", methods=["POST"])
def api_duplicates_delete():
    """
    Delete a specific invoice number from the raw dataframe.
    Keeps the first occurrence, removes all others.
    User can also delete all occurrences by passing delete_all=true.
    """
    if not STATE["loaded"]:
        return jsonify({"ok": False, "error": "Load data first"}), 400

    inv_no     = request.json.get("inv_no", "").strip()
    delete_all = request.json.get("delete_all", False)

    m  = STATE["mappings"]
    id_col = None
    for mp in m:
        if mp["category"] == "identifier":
            id_col = mp["original_column"]
            break

    if not id_col or not inv_no:
        return jsonify({"ok": False, "error": "Cannot identify invoice column"}), 400

    df = STATE["df"]
    if delete_all:
        STATE["df"] = df[df[id_col] != inv_no].copy().reset_index(drop=True)
        removed = int((df[id_col] == inv_no).sum())
    else:
        # Keep only the first occurrence, drop subsequent ones
        first_idx = df[df[id_col] == inv_no].index[0]
        drop_idx  = df[(df[id_col] == inv_no) & (df.index != first_idx)].index
        STATE["df"] = df.drop(drop_idx).reset_index(drop=True)
        removed = len(drop_idx)

        return jsonify({
        "ok": True,
        "removed": removed,
        "inv_no": inv_no
    })


if __name__ == "__main__":
    print("\n  Sales Analysis Agent — UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5001)