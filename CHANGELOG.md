# CHANGELOG — Runtime Concurrency Stability Audit (Phase I)

Full project-wide concurrency audit in response to a reported segmentation
fault with `concurrent.futures.thread._worker` frames on the stack, plus
`resource_tracker: leaked semaphore objects` warnings at shutdown, on
macOS (Apple Silicon). Two real, fixable issues were found and fixed. No
architecture was changed, no model was removed, and parallel execution
was made configurable, not deleted. Diffed against the prior delivered
build (`sales_agent_poc_phaseH_runtime_stability.zip`): only `trainer.py`
and `process_isolation.py` changed. Every other file — every model,
`selector.py`, `results.py`, `forecast_engine.py`, `evaluator.py`, the
Flask backend, and the frontend — is byte-identical.

## Step 1 — project-wide concurrency inventory

Searched the entire project (`src/`, `ui/`, excluding the stray `venv/`
directory) for `ThreadPoolExecutor`, `ProcessPoolExecutor`,
`multiprocessing`, `concurrent.futures`, `threading`, and `joblib`/`n_jobs`
usage. **Result: exactly three files reference concurrency, all inside
`src/forecast/`, and there is exactly ONE thread/process pool in the
entire codebase:**

| File | What it uses | What runs inside it |
|---|---|---|
| `forecast/trainer.py` | `ThreadPoolExecutor(max_workers=4)` | `_train_one_candidate()` per model — hyperparameter tuning, walk-forward validation (repeated fit/predict), and the final full-series fit, for whichever of SARIMA/Holt-Winters/XGBoost/Prophet/Holt Linear/SES/Moving Average/Naive were recommended |
| `forecast/models/prophet.py` | Imports and calls `run_isolated()` | No thread/process pool of its own — delegates to `process_isolation.py` |
| `forecast/process_isolation.py` | `multiprocessing.get_context("spawn")` — `Queue` + `Process` | One isolated child process per Prophet fit/predict/tune call (see Phase H) |

No other file — `ui/app.py`, `data_loader.py`, `schema_mapper.py`,
`embedding_mapper.py` (sentence-transformers/PyTorch), `data_cleaner.py`,
`analytics.py`, `query_engine.py`, `main.py` — creates a thread pool,
process pool, or explicit background thread anywhere in this project.
This directly narrows the investigation: the `concurrent.futures.thread
._worker` frames in the crash report can only be `trainer.py`'s
`ThreadPoolExecutor` — there is nothing else in the codebase that could
have produced that stack frame.

## Step 2 & 6 — which native libraries actually run inside that pool, and are they thread-safe on macOS

Reviewed every candidate model's `suggest_hyperparameters()`/`fit()`/
`predict()` to determine which ones execute native, non-Python code when
called from inside `trainer.py`'s `ThreadPoolExecutor` workers:

- **SARIMA, Holt-Winters** (`statsmodels`): every `fit()` call —
  including the ones inside `suggest_hyperparameters()`'s AIC grid search
  and every walk-forward fold — runs `scipy.optimize`-based maximum-
  likelihood estimation, which calls into LAPACK/BLAS (on macOS, typically
  Apple's Accelerate framework). With up to 4 candidates training
  concurrently, this means multiple threads can be inside BLAS/LAPACK
  native code AT THE SAME TIME. Apple's Accelerate framework has
  documented thread-safety edge cases under exactly this kind of
  concurrent, multi-threaded invocation from a process that did not
  itself initialise Accelerate's threading state up front (numerous
  numpy/scipy issue reports of intermittent segfaults on Apple Silicon
  under concurrent BLAS calls from a `ThreadPoolExecutor`).
- **XGBoost**: confirmed via a project-wide grep for `n_jobs`/`nthread`
  that no `XGBRegressor(...)` construction anywhere in `models/xgboost.py`
  sets an explicit thread count — it defaults to using ALL available CPU
  cores via its own internal OpenMP thread pool for every single fit
  (there are several per candidate: the tuning grid, every walk-forward
  fold, and the final fit). XGBoost's macOS wheels bundle `libomp`
  (LLVM's OpenMP runtime). Initialising and tearing down an OpenMP thread
  pool from a NON-main, transient thread (a `ThreadPoolExecutor` worker,
  which Python may create and destroy across calls) — and doing so
  CONCURRENTLY from multiple such worker threads, each spinning up its own
  OpenMP pool — is a documented source of instability for `libomp` on
  macOS, including reports of segfaults specifically at process exit when
  OpenMP thread-pool teardown races against Python interpreter
  finalisation.
- **Prophet** (`cmdstanpy`): already isolated into its own spawned
  subprocess since Phase H — never runs inside `trainer.py`'s
  `ThreadPoolExecutor` threads directly (only the lightweight act of
  *starting* that subprocess happens there, which `multiprocessing`'s own
  documentation confirms is safe to do from a non-main thread when using
  the `"spawn"` start method).
- **PyTorch / sentence-transformers**: confirmed via the Step 1 grep that
  `embedding_mapper.py` (the only file that imports `sentence_transformers`
  /`torch`) is never called from inside `trainer.py`'s `ThreadPoolExecutor`
  or any other pool — it runs once, synchronously, during schema mapping
  (`/api/load`), on whichever thread handles that Flask request. It is
  **not a contributor to this crash** (no evidence it runs concurrently
  with anything in this codebase), though it is worth recording that
  PyTorch also maintains its own internal native thread pool
  (MKL/Accelerate-backed), and the same class of risk would apply if it
  were ever invoked concurrently with other native code in the future.

**Conclusion**: the segfault's proximate cause is native, C-level
thread-pool/BLAS code (Accelerate via statsmodels/scipy, and/or `libomp`
via XGBoost) being entered concurrently from multiple
`ThreadPoolExecutor` worker threads — exactly the pattern the crash
report's own stack trace (`concurrent.futures.thread._worker`) points to,
now confirmed rather than assumed, since the project-wide search in Step 1
proves there is no other pool it could have come from.

## Step 5 — resource/shutdown audit, second real bug found

Reviewing every `Queue`/`Process` lifecycle in `process_isolation.py`
(the only file in the project that creates `multiprocessing` primitives)
found a genuine, independent bug: `_CTX.Queue()` and `_CTX.Process(...)`
were created and used, but neither was ever explicitly `.close()`d. A
`multiprocessing.Queue` holds an OS pipe plus (on most platforms,
including macOS) a semaphore registered with `multiprocessing
.resource_tracker`; leaving it for garbage collection instead of
explicitly closing it is precisely what produces the reported
`UserWarning: resource_tracker: There appear to be leaked semaphore
objects` — and because `run_isolated()` is called once per Prophet
tuning candidate, once per walk-forward fold, and once for the final fit,
a single forecast request can leak several of these before the warning is
ever noticed at process exit. `trainer.py`'s `ThreadPoolExecutor` itself
was already correctly used as a context manager (`with ThreadPoolExecutor
(...) as executor:`), which calls `shutdown(wait=True)` on exit — so
`trainer.py` was not independently leaking threads; the leak was isolated
to the multiprocessing resources in `process_isolation.py`.

## Files Modified

| File | Root cause | Why the crash/warning happened | Why the fix is correct |
|---|---|---|---|
| `src/forecast/trainer.py` | The project's only `ThreadPoolExecutor` runs multiple native, non-thread-safe-on-macOS libraries (statsmodels/scipy via Accelerate BLAS, XGBoost via its own `libomp` pool) concurrently across up to 4 worker threads. | Concurrent entry into Accelerate/`libomp` native thread-pool code from several Python threads at once is a documented macOS crash pattern; the reported segfault's `concurrent.futures.thread._worker` stack frames are consistent with this and, per the Step 1 inventory, cannot have come from anywhere else in this codebase. | Added `ENABLE_PARALLEL_FORECASTING` (default `False`) and factored the existing sequential fallback logic into a reusable `_train_sequential()` helper, called either when the flag is off or if the thread pool itself fails to start. Both the parallel and sequential paths call the exact same `_train_one_candidate()` per model, so **output is verified byte-identical between modes** (same metrics, same chosen hyperparameters, same trained-model set) — only whether candidates' native library calls can overlap in time changes. Parallel execution is fully preserved (`ENABLE_PARALLEL_FORECASTING = True` re-enables the exact prior behaviour), matching the explicit instruction not to remove it. |
| `src/forecast/process_isolation.py` | Every `multiprocessing.Queue`/`Process` created by `run_isolated()` was left for garbage collection instead of being explicitly closed. | This is the direct, independent cause of the reported `resource_tracker: leaked semaphore objects` warning — a `Queue`'s underlying semaphore is only guaranteed to be released promptly via an explicit `.close()`/`.join_thread()` call, not by relying on `__del__`/garbage collection timing, especially inside a long-running Flask process making many such calls over its lifetime. | Added an explicit cleanup block: `Process.close()` after confirming the process is no longer alive (terminating it first if a timeout occurred), and `Queue.close()` + `Queue.join_thread()` in every case, all inside the existing `finally` block so cleanup runs on every return path (success, ordinary failure, timeout, and even a failure of the isolation machinery itself). Verified directly: running several isolated Prophet calls (one fit + three predicts, matching a real walk-forward-validation-style call pattern) in a single short-lived process and checking its full stderr output on exit shows zero `resource_tracker`/leaked-semaphore warnings. |

## Step 4 — per-model isolation re-verified (no change needed)

`_train_one_candidate()`'s own try/except (hyperparameter tuning wrapped
separately from validation+fit, both catching `Exception` and returning a
`ModelEvalResult` rather than raising) is called identically by both the
parallel and the new sequential path — switching modes does not weaken
this guarantee in either direction. Confirmed via the same test that
verified output-identity between modes: every candidate's `trained`/
`status`/`error` fields matched exactly regardless of which mode trained
it.

## Step 7 — shared mutable state audit (no issue found)

Grepped every model file in `forecast/models/` for any in-place mutation
of the shared `series`/`tuning_prefix` arguments (e.g. `series[i] = ...`,
`.values[...] = ...`) — none exists anywhere in the package; every model
only ever slices/reads them. `trained_models`, `results_by_name`, and
`warnings_by_name` are written to exclusively by the single thread running
`train_candidates()` itself, in the result-collection loop AFTER
`future.result()`/`_train_sequential()` returns — never concurrently by
the worker threads. The only process-wide mutable singleton found in the
whole project is `embedding_mapper.py`'s lazily-loaded `SentenceTransformer`
instance, which is outside the forecasting pipeline this audit is scoped
to and is not invoked from inside any parallel worker.

## Step 8 — Flask integration audit (no issue found)

`ui/app.py`'s `app.run(debug=False, host="0.0.0.0", port=5001)` does not
pass `threaded=True`; verified directly against the installed Werkzeug
version that `threaded` defaults to `False` (confirmed via
`inspect.signature` on `werkzeug.serving.run_simple`, not assumed) — the
dev server used here handles one request at a time, so the global `STATE`
dict is not subject to concurrent-request races in this deployment.
Every `ThreadPoolExecutor`/`Process` created during a request is fully
joined/shut down (via `with ThreadPoolExecutor(...)` and, after this
pass's fix, explicit `Process.close()`) before `train_candidates()`
returns and the Flask response is sent — nothing outlives the request.

## Assumptions

- `ENABLE_PARALLEL_FORECASTING` defaults to `False`. The task's own
  framing ("keep parallel execution available behind a configuration flag
  **once stability is confirmed**") implies stability is not yet
  confirmed — that is the reason for this audit — so sequential is the
  safe default until real-world (not just sandboxed) runs on the affected
  macOS environment confirm the parallel path no longer crashes. Flipping
  the flag back to `True` requires no code change, only that one
  assignment.
- No native library's own configuration (e.g. forcing `nthread=1` on
  XGBoost, or setting `OMP_NUM_THREADS`) was changed. The fix is scoped to
  where this project invokes those libraries (never do so concurrently by
  default), not to reconfiguring the libraries themselves — a narrower,
  more conservative change that doesn't risk altering any model's
  numerical behaviour.
- PyTorch/sentence-transformers were investigated and ruled out as a
  contributor to this specific crash (never invoked concurrently with
  anything else in this codebase) rather than being preemptively modified.

## Verification

Full regression against the live Flask app: `/api/load`, `/api/kpis`,
`/api/anomalies`, `/api/forecast`, `/api/forecast/analysis`,
`/api/duplicates`, and `/api/forecast/auto` all return HTTP 200 in
sequence, the server remains alive and responsive to a final health check
afterward, and the server's own stderr log contains zero occurrences of
"segmentation", "Fatal Python error", "resource_tracker", or "leaked
semaphore". Sequential- and parallel-mode training verified to produce
byte-identical model rankings, metrics, and hyperparameters on the same
synthetic dataset — this pass changes execution strategy and resource
cleanup only, never forecasting logic or output.
