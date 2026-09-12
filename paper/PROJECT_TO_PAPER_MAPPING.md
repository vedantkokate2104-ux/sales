# Project → Paper Mapping

Every claim in `main.tex` traced to the source file that implements it and the
experiment output that measured it. Use this to verify that the paper
describes your implementation and nothing else.

Paths are relative to the repository root. Experiment outputs are in
`paper/experiments/results/*.json`; the scripts that produced them are in
`paper/experiments/`.

---

## A. Architecture and pipeline

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/forecast_engine.py` → `run_auto_forecast()` | §IV (whole), Fig. 1 | Orchestrator; the eight stages A–H in Fig. 1 are its call sequence |
| `src/forecast/__init__.py`, `types.py`, `results.py` | §IV data structures | `SeriesStats`, `SeriesClassification`, `ModelEvalResult`, `SelectionResult`, `AutoForecastResult` |
| `ui/app.py` routes (`/api/forecast/auto`, `/api/forecast/analysis`, `/api/forecast/groups`) | §IV, §VI-I | The deployment surface; not itself evaluated in the paper |

## B. Stage A — semantic schema resolution

| Project component | Paper section | Evidence |
|---|---|---|
| `src/schema_mapper.py` — `SYNONYM_MAP`, `_rule_match()`, `map_all_columns()` | §IV-A ("A rule stage matches the normalised name…") | `exp1.json → schema_mapping` |
| `src/embedding_mapper.py` — `ERP_ABBREVIATIONS`, `_expand_column_name()`, `_detect_patterns()`, E5 similarity ranking | §IV-A ("A semantic stage expands ERP abbreviations…"), ref. [26] | `exp1.json → schema_mapping.by_source` |
| `column_memory.json` + `correct_mapping()` | §IV-A ("Human corrections are persisted…") | Not exercised in E1: the experiment runs cold in a temp directory with no memory file |
| **Result:** 21/21 columns correct, mean confidence 0.943, 13/21 hybrid agreement, 8/21 rule-over-embedding, 31.6 s cold | §VI-A | `exp1.json → schema_mapping` (`n_high_0.75`, `mean_confidence`, `by_source`, `seconds`) |

> Correctness of the 21 mappings was assessed by manual inspection of the
> column semantics, as stated in §VI-A and §VIII. There is no independent
> labelled ground truth in the repository.

## C. Stage B — transaction identity and cleaning

| Project component | Paper section | Evidence |
|---|---|---|
| `src/transaction_key.py` — `_KEY_COMPONENTS`, `MIN_KEY_COMPONENTS = 3`, `resolve_business_key()`, `analyze_duplicates()` | §IV-B, §VI-A | `exp1.json → duplicates` |
| `src/duplicate_detector.py` — `_normalize()` | §IV-B (value normalisation) | Reused by `transaction_key` for key comparison |
| `src/forecast/preprocessing.py` — `parse_dates_deterministic()`, `_CANDIDATE_DATE_FORMATS`, `_MIN_FORMAT_MATCH_SHARE = 0.8` | §IV-B ("Dates are parsed deterministically…") | Implementation; not separately benchmarked |
| `src/forecast/preprocessing.py` — `clean_transactions()`, `compute_data_quality_score()` | §IV-B, §VI-H (Q = 0.913) | `exp2.json → auto_forecast.result.data_quality_score` |
| `src/forecast/validation.py` — `validate_dataset()` | §IV-B (reported counts) | `exp2.json → auto_forecast.result.warnings` |
| **Result:** business key = 6 columns; 7 exact-row and 20 business-key duplicates; 212 repeated-identifier rows of which 192 preserved | §VI-A | `exp1.json → duplicates.analysis` |
| **Result:** naive `inv_no` dedup destroys 54.08 % of rows / 56.65 % of revenue | §VI-A, §VII-A | `exp1.json → duplicates.naive_identifier_dedup` |

## D. Stage C — forecast-grain selection

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/grain.py` — `CANDIDATE_GRAINS`, `MIN_PERIODS = 12`, `MIN_PERIODS_PER_HORIZON = 4`, `MIN_DENSITY = 0.60` | §IV-C rejection rules | `exp1.json → grain.candidates` |
| `src/forecast/grain.py` — `_score_grain()` weights 0.35 / 0.25 / 0.15 / 0.15 / 0.10, `_RESOLUTION_PREFERENCE` | §IV-C **eq. (1)** | Weights in eq. (1) are copied verbatim from `_score_grain()` |
| `src/forecast/grain.py` — `SELECTION_HOLDOUT_FRACTION = 0.2`, `_training_span_mask()` | §IV-C ("computed on the leading 80 % of the calendar span") | `exp1.json → grain.selection_basis` |
| `src/forecast/grain.py` — `resolve_fill_strategy()`, `_interpolate_short_runs()`, `MAX_INTERPOLATED_RUN = 2` | §IV-C fill strategy | `exp1.json → grain.construction` |
| `src/forecast/grain.py` — `aggregate_to_period()` returning `observed_mask` | §IV-C ("missing is not zero") | `exp1.json → grain.construction.*.n_missing / n_explicit_zero` |
| `src/forecast/utils.py` — `infer_frequency()` | §VI-B ("inter-arrival inference returns *weekly*") | `exp1.json → grain.inter_arrival_inferred_frequency` |
| **Result:** Table III (4 grains, density/CV/score) | §VI-B | `exp1.json → grain.candidates` |
| **Result:** best WAPE per grain (weekly 69.03, monthly 47.48, quarterly 12.87) | §VI-B, Table III, Fig. 3 | `exp3.json → A_grain.*.best_wape` |
| **Result:** transaction-level vs period-level classification (CV 1.7242 → *Highly Volatile* vs 0.6087 → *Long History*) | §VI-B | `exp1.json → classification_transaction_level`, `classification` |

## E. Stages D & E — fold planning and per-fold backtesting

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/backtest.py` — `plan_validation()`, `MIN_ABSOLUTE_TRAIN = 8`, `MIN_TRAIN_FRACTION = 0.5`, `TARGET_FOLDS = 12`, `MIN_FOLDS = 3` | §IV-D **eqs. (2), (3)** | `exp2.json → benchmark_untuned.validation` |
| `src/forecast/backtest.py` — `backtest_model()`, `FoldSpec`, `FoldOutcome` | §IV-E, Fig. 4 | `exp2.json → benchmark_untuned.validation.folds` |
| `src/forecast/backtest.py` — `FoldPreprocessor.fit()` / `prepare_training_window()` / `_bridge_missing()` | §IV-E ("fitted on the raw training prefix only") | E5 measures the alternative: `exp3.json → C_preprocessing`, `exp4.json → F_leakage` |
| `src/forecast/backtest.py` — `pool_metrics()` | §IV-E ("pooled … computed once") | Implementation |
| `src/forecast/backtest.py` — `summarise_folds()` | §IV-E **eq. (5)** stability | `exp2.json → benchmark_untuned.rows[*].stability` |
| `src/forecast/evaluator.py` — `wape()`, `mape()` (zero-excluded, `_MAPE_POINT_CAP = 500`), `smape()`, `compute_metrics()` | §IV-E **eq. (4)**, §VI-F | `exp4.json → G_metrics` |
| `src/forecast/trainer.py` — `train_candidates()`, `GUARANTEED_CANDIDATES`, `check_requirements()` pre-flight | §IV-E, §V-B | `exp2.json → auto_forecast.result.model_ranking` |
| `src/forecast/trainer.py` — `suggest_hyperparameters()` on the first fold's prefix | §V-B, §VI-C (tuning helped 1, hurt 2) | `exp2.json → benchmark_untuned` vs `benchmark_tuned` |
| `src/forecast/process_isolation.py` | §V-C (Prophet failure isolated) | `exp2.json → benchmark_*.unavailable`; `auto_forecast.result.warnings` |
| **Result:** plan = 11 folds, 33 points, t₀ = 24, δ = 2 | §V-C, §VI-C, Fig. 4 | `exp2.json → benchmark_untuned.validation` |
| **Result:** Table V (8 models) | §VI-C | `exp2.json → benchmark_untuned.rows` |
| **Result:** protocol comparison + rank correlations | §VI-D, Table VI, Fig. 6 | `exp3.json → B_protocol`; `exp4.json → E_rank_agreement` |
| **Result:** leakage deltas | §VI-E, Table VII | `exp3.json → C_preprocessing` (null result at deployed setting); `exp4.json → F_leakage` |

## F. Stage F — model selection

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/selector.py` — `_rank_key()` (WAPE → SMAPE → RMSE → MAE, MAPE excluded) | §IV-F | `exp2.json → auto_forecast.result.model_ranking` |
| `src/forecast/selector.py` — `_TIE_BREAK_RELATIVE_BAND = 0.05`, `_apply_business_aware_tie_break()`, `_COMPLEXITY_RANK` | §IV-F ("within 5 % … empirical tie") | Not triggered on this dataset: SES leads the runner-up by 9.1 % of its own WAPE (47.48 vs 51.80), outside the 5 % tie band |
| `src/forecast/selector.py` — `_ENSEMBLE_RELATIVE_BAND = 0.25`, `_ENSEMBLE_MAX_MODELS = 4`, `_ENSEMBLE_MIN_IMPROVEMENT = 0.01` | §IV-F **eq. (6)** | `exp2.json → auto_forecast.result.selection_reason` (ensemble measured and rejected) |
| `src/forecast/backtest.py` — `evaluate_ensemble()` with pre-fold-k weights | §IV-F ("weights derived at fold k only from folds strictly before k") | `exp2.json → benchmark_untuned.ensemble.weight_basis` |
| `src/forecast/selector.py` — `_compute_confidence()` (8 signals, clipped to [5, 97]) | §VI-H ("legacy scalar confidence is 65 %") | `exp2.json → auto_forecast.result.forecast_confidence_pct` |
| `src/forecast/model_selector.py` + `classifier.py` | §IV (candidate shortlisting, mentioned in Fig. 1 flow) | `exp1.json → recommended_candidates` |
| **Result:** SES selected; ensemble 50.05 % measured and rejected | §VI-C | `exp2.json → benchmark_untuned.ensemble.pooled_metrics` |

## G. Stage G — conformal prediction intervals

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/intervals.py` — `collect_relative_errors()` | §IV-G ("r = (y − ŷ)/ŷ") | `exp2.json → …prediction_intervals.n_errors_by_horizon` |
| `src/forecast/intervals.py` — `conformal_indices()`, `error_quantiles()` | §IV-G **eq. (7)** | Index formula in eq. (7) matches the implementation, tolerance term included |
| `src/forecast/intervals.py` — `choose_basis()`, `MIN_ERRORS_PER_HORIZON_FLOOR = 8` | §VI-H ("below the 19 per step … so the system pools") | `exp2.json → …prediction_intervals.notes[0]` |
| `src/forecast/intervals.py` — `MIN_ERRORS_FOR_CALIBRATED = 40`, `CONFORMAL` / `CONFORMAL_RANGE` / `EXTRAPOLATED` | §IV-G, §VI-H | `exp2.json → …interval_calibration_status` = `LIMITED_DATA` |
| `src/forecast/intervals.py` — `evaluate_coverage()` (prequential, `min_training_folds = 4`) | §IV-G, §VI-H | `exp2.json → …prediction_intervals.coverage` |
| `src/forecast/intervals.py` — `_interval_score()` | §IV-G **eq. (8)** | `exp2.json → …prediction_intervals.sharpness.*.interval_score` |
| `src/forecast/intervals.py` — `_enforce_nesting_bounds()`, `compute_level_bounds()` | §IV-G (single bound implementation) | Implementation |
| **Result:** 80 % → 66.7 % (n = 21), 95 % → 66.7 %, Winkler 2.839 / 3.305 MINR | §VI-H, Table IX, Fig. 8 | `exp2.json → …prediction_intervals` |

## H. Stage H — reliability reporting

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/reliability.py` — `SIGNAL_WEIGHTS`, `LEVEL_THRESHOLDS`, `assess_reliability()` | §IV-H **eq. (9)** | Weights in eq. (9) copied verbatim from `SIGNAL_WEIGHTS` |
| `src/forecast/reliability.py` — `is_probability = False`, `interpretation` text | §IV-H ("states in text that R is an evidence rating") | `exp2.json → …reliability.is_probability` |
| `src/forecast/insights.py` — `explain_model_selection()`, `build_insights()` | §IV-H (per-factor breakdown) | `exp2.json → …forecast_insights` |
| **Result:** R = 0.602 MODERATE, six-factor decomposition | §VI-H, Table X | `exp2.json → …reliability.factors` |

## I. Segment-level forecasting

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/segments.py` — `discover_group_columns()`, `MAX_CARDINALITY = 50`, `MAX_UNIQUE_RATIO = 0.5`, `_NON_GROUPABLE_CATEGORIES` | §VI-I ("nine of the 21 columns qualified") | `exp4.json → H_segments.discovered` |
| `src/forecast/segments.py` — `resolve_group()` typed statuses | §IV (Fig. 1), §VI-I | Not triggered: all 12 requested segments resolved OK |
| `src/forecast/segments.py` — `classify_history()` | §VI-I ("all twelve … sufficient history") | `exp4.json → H_segments.forecasts[*].history_status` |
| **Result:** Table XI (12 segments) | §VI-I | `exp4.json → H_segments.forecasts` |

## J. Feature engineering and the ablation study

| Project component | Paper section | Evidence |
|---|---|---|
| `src/forecast/features.py` — `build_feature_frame()` with `series.shift(1)` before every window | §IV-E leakage guarantee, §VI-G | Implementation |
| `src/forecast/features.py` — `LAGS_BY_FREQUENCY`, `ROLLING_WINDOWS_BY_FREQUENCY`, `CALENDAR_BY_FREQUENCY`, `resolve_lags()` | §V-B, §VI-G feature families | `exp3.json → D_features.results[*].n_features` |
| `src/forecast/features.py` — `select_features()` (constant / correlation ≥ 0.98 removal) | §VI-G ("18 selected features") | `exp3.json → D_features` |
| `src/forecast/features.py` — `classify_business_column()`, `USABLE_AVAILABILITY`, `assert_no_unavailable_features()` | §VII-B ("availability contract … no column qualified") | Implementation; the four target-restatement columns are named in the module |
| `src/forecast/models/tabular.py` — `TabularForecastModel`, `make_tabular_model()`, `ESTIMATOR_FACTORIES`, `training_error_` | §V-B, §VI-G | `exp3.json → D_features.results[*].training_error` |
| `src/forecast/experiments.py` — `run_experiments()`, `default_tabular_specs()`, `MIN_MEANINGFUL_IMPROVEMENT_PP = 1.0`, `OVERFIT_RATIO_THRESHOLD = 2.0` | §V-D(E7), §VI-G, Table VIII, Fig. 7 | `exp3.json → D_features` |
| **Result:** 22 configurations (15 ML, 7 statistical); none accepted; overfit ratios up to 5.69× | §VI-G | `exp3.json → D_features.results` |

## K. Model library

| Project component | Paper section |
|---|---|
| `src/forecast/models/naive.py`, `seasonal_naive.py`, `moving_average.py`, `simple_exponential.py`, `holt.py` | §V-B (pure numpy/pandas models, always available) |
| `src/forecast/models/holt_winters.py`, `sarima.py` | §V-B (statsmodels-backed) |
| `src/forecast/models/prophet.py` (+ `process_isolation.py`) | §V-B, §V-C (backend non-functional here) |
| `src/forecast/models/xgboost.py` | §V-B, §VI-C, §VI-E |
| `src/forecast/models/base.py` — `BaseForecastModel`, `check_requirements()`, `suggest_hyperparameters()` | §IV-E, §V-B |
| `src/forecast/benchmark.py` — `benchmark_series()`, `benchmark_dataframe()`, `format_report()` | §V-D(E3), Table V |

## L. Components deliberately NOT claimed in the paper

These exist in the repository and are **not** presented as contributions or
evaluated, so the paper does not overstate its scope:

- `src/analytics.py` — KPI engine and anomaly detection (`calculate_kpis`,
  `detect_anomalies`, `run_forecast`). Only `get_col()` and `_clean_numeric()`
  are used by the forecasting path; the KPI and anomaly features are not
  discussed.
- `src/query_engine.py` — natural-language query handling. Not mentioned.
- `src/main.py` — terminal demo. Not mentioned.
- `ui/` frontend (`app.js`, `style.css`, `index.html`). Not mentioned.
- `src/data_cleaner.py` — the general cleaning report used by the UI, distinct
  from `forecast/preprocessing.py` used by the forecasting path.
- `tests/test_phase*.py` — the project's own test suite. Not run or reported
  in the paper.
- `ENABLE_PARALLEL_FORECASTING` and the concurrency audit documented in
  `CHANGELOG.md` / `TEST_REPORT.md`. The paper states only that training ran
  sequentially (§V-C); it makes no claim about the parallel path.

## M. Claims in the paper that required a check beyond the code

| Claim | How it was established |
|---|---|
| "21/21 columns correct" | Manual inspection of each mapping against the column's values and name; stated as such in §VI-A and §VIII |
| Rank correlations ρ, τ and their p-values | `scipy.stats.spearmanr` / `kendalltau` in `exp4_more.py`; values in `exp4.json → E_rank_agreement` |
| Leakage deltas at 1.5×IQR | A **sensitivity analysis**, explicitly labelled as not the deployed setting (`OUTLIER_IQR_MULTIPLIER` is 3.0 in `backtest.py`); the experiment monkey-patches the constant and restores it |
| ε-guarded MAPE blow-up | `exp4_more.py → epsilon_mape()`, a deliberate reimplementation of the *rejected* formulation, compared against the project's own `evaluator.mape()` |
| "eight models consume 2.9 s of fitting" | Sum of `exp2.json → benchmark_untuned.rows[*].runtime_seconds` |
| Prophet backend failure | Reproduced directly: `ProphetModel.fit()` raises `AttributeError: 'Prophet' object has no attribute 'stan_backend'` inside the isolated subprocess |
