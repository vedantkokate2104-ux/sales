# TEST REPORT — Runtime Concurrency Stability Audit (Phase I)

## 1. Scope verification

Diffed against `sales_agent_poc_phaseH_runtime_stability.zip`: only
`src/forecast/trainer.py` and `src/forecast/process_isolation.py`
changed. Every model, `selector.py`, `results.py`, `forecast_engine.py`,
`evaluator.py`, the Flask backend, and every frontend file are
byte-identical to the prior build.

## 2. Concurrency inventory (Step 1)

`grep`-based search across `src/` and `ui/` for `ThreadPoolExecutor`,
`ProcessPoolExecutor`, `multiprocessing`, `concurrent.futures`,
`threading`, `joblib`, `n_jobs`: exactly 3 files matched, all inside
`src/forecast/` (`trainer.py`, `models/prophet.py`,
`process_isolation.py`). Separately confirmed no `n_jobs`/`nthread`/
`n_threads` is set anywhere in the project (XGBoost's `XGBRegressor`
therefore always defaults to all-cores via its internal OpenMP pool).

## 3. Mode-equivalence test (sequential vs. parallel)

Same 42-observation synthetic monthly series, 6 candidates
(XGBoost/SARIMA/Holt-Winters/Holt Linear/Moving Average/Naive), run once
with `ENABLE_PARALLEL_FORECASTING = False` (the new default) and once with
`True`:

| Mode | Elapsed | Result |
|---|---|---|
| Sequential (default) | 2.51s | 7/7 candidates trained |
| Parallel (flag=True) | 3.33s | 7/7 candidates trained |

Verified programmatically: identical trained-model set, identical WAPE
per model, identical chosen hyperparameters (SARIMA order `(2,1,2)`,
Holt-Winters config, XGBoost tuning) in both modes — confirms the flag
changes only execution strategy, never forecasting logic or output. (In
this sandboxed environment sequential was also marginally faster,
consistent with thread-pool overhead not paying off at this scale /
possible thread contention — not a claim about every environment, just an
observation supporting the conservative default.)

## 4. Resource-leak fix verification (Step 5)

Ran a short-lived script performing 4 isolated Prophet calls in sequence
(1 `fit()` + 3 `predict()`, mirroring the call pattern a real walk-forward
validation loop produces) and inspected its complete stderr output on
exit: zero occurrences of `resource_tracker` or "leaked semaphore".

## 5. Full live Flask regression

Started the real `ui/app.py`, loaded the bundled sample dataset, and hit
every endpoint the bug report listed plus `/api/forecast/auto`:

| Route | Result |
|---|---|
| `POST /api/load` | 200 |
| `POST /api/forecast/auto` | 200 — `Weighted Ensemble (Naive + Simple Exponential Smoothing + Prophet)`, `forecast_confidence_pct: 53` |
| `GET /api/kpis` | 200 |
| `GET /api/anomalies` | 200 |
| `GET /api/duplicates` | 200 |
| `GET /api/forecast/analysis` | 200 |
| `GET /` (health check, AFTER all of the above) | 200 |

Server's complete stderr/stdout log for this run contains **zero**
occurrences of "segmentation", "Fatal Python error", "resource_tracker",
or "leaked semaphore". The process remained alive and responsive to a
final request issued after every listed endpoint had already completed —
directly reproducing, and passing, the exact scenario described in the
bug report ("only after these complete does Python terminate").

## 6. Step 7/8 findings (no code change needed, verified by inspection)

- No in-place mutation of the shared `series`/`tuning_prefix` arguments
  anywhere in `forecast/models/` (grepped for `series[...] =`,
  `.values[...] =` — no matches other than an unrelated docstring
  comment).
- `trained_models`/`results_by_name`/`warnings_by_name` are written only
  by the single thread running `train_candidates()`'s result-collection
  loop, never by the worker threads themselves.
- `werkzeug.serving.run_simple`'s `threaded` parameter confirmed (via
  `inspect.signature`, not documentation lookup) to default to `False` in
  the installed version; `ui/app.py` does not override it — the dev
  server used here processes one request at a time, so the global
  `STATE` dict is not subject to concurrent-request races in this
  deployment.

## 7. Conclusion

Two real, independent bugs identified and fixed:
1. `trainer.py`'s `ThreadPoolExecutor` could run multiple native libraries
   (statsmodels/scipy via Accelerate BLAS, XGBoost via `libomp`)
   concurrently — a documented macOS crash pattern — now gated behind
   `ENABLE_PARALLEL_FORECASTING` (default `False`), with the parallel path
   fully preserved and verified output-identical to sequential.
2. `process_isolation.py` was leaking `multiprocessing.Queue`/`Process`
   resources — now explicitly cleaned up in every code path, verified to
   produce zero `resource_tracker` warnings across a multi-call sequence.

No model was removed, no architecture was changed, and every existing
API/response contract is unchanged.
