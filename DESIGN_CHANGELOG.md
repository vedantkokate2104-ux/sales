# DESIGN CHANGELOG — Forecast Page Redesign

Purely a frontend/UX rework of the Forecast page. **No backend file was
touched** — verified by SHA-256 hashing every `.py` file before and after
this change; all 29 files are byte-identical (see Verification below). No
forecasting algorithm, model selection rule, evaluation metric, or data
processing step was modified. No other frontend page (Data Overview,
Schema AI, KPIs, Anomalies, Chat) was touched — verified by confirming
every other page's JavaScript function is byte-identical to before this
change, and every other page's HTML `<section>` is unchanged.

## What Changed, and Why

### Section 1 — Forecast Summary (new)
Four large hero cards — Selected Model, Forecast Horizon, Forecast
Confidence, Expected Revenue — plus the horizon selector and "Run Auto
Forecast" button, now sit at the very top of the page. **Why:** the old
page opened with a horizon dropdown and a "Forecast Analysis" technical
panel; a business user had to scroll and interpret raw statistics before
reaching an answer to "what is the forecast and how sure are we?" The
four numbers a manager actually needs are now the first thing on screen,
in large type, with everything else demoted below them.

### Section 2 — Historical vs Forecast Chart (enlarged, enhanced)
The chart grew from a 180px-tall utility chart to the visually dominant
element on the page (340px, full card width, larger fonts), and gained a
shaded confidence band around the forecast line plus hover tooltips on
every point (period + value + Historical/Forecast label). **Why:** the
task's design goal explicitly calls for the chart to be the visual focus,
similar to Power BI/Tableau; the old chart was one of six equally-weighted
cards with no way to inspect an individual point's value without reading
the table below it.

### Section 3 — Forecast Results (restructured)
Replaced the old always-expanded forecast table with compact monthly
cards (e.g. "January — ₹12.4L") as the primary view, with a "View
Detailed Forecast Table" button that expands a full table only on
request. **Why:** a table of raw numbers is the ML-debugging-interface
look the task asks to move away from; the monthly cards give the same
information at a glance, with the precise table still one click away for
anyone who wants it (progressive disclosure, nothing removed).

### Section 4 — Selected Model (condensed)
Model name, reason for selection, horizon, validation metrics (as small
inline chips, not a large stat grid), and alternative models (as compact
pills), with no other implementation detail. **Why:** the old panel mixed
this with a full stats grid and Phase 1 classification data in the same
card; splitting it out and trimming it to exactly what Step 4/7 of the
brief ask for keeps it scannable in a few seconds.

### Section 5 — Dataset Analysis (moved into a collapsed accordion)
Every statistic Phase 1's analyzer computes (observations, frequency,
trend/seasonality strength, variance, coefficient of variation, missing
values/timestamps, history length, outlier %, zero-sales %) is unchanged
in content, but now lives behind a "Dataset Analysis" accordion, collapsed
by default. **Why:** this is precisely the "machine learning debugging
interface" content the task calls out — still fully available (nothing
removed, nothing recalculated differently), just not forced on every
visitor by default.

### Section 6 — Model Comparison (moved into a collapsed accordion)
The full MAE/RMSE/MAPE/SMAPE comparison table for every model the trainer
attempted, unchanged in content, moved behind its own collapsed
accordion. **Why:** same rationale as Section 5 — advanced/technical
detail for the users who want it, out of the way for those who don't.

### Section 7 — Warnings (rewritten to be business-friendly)
Backend warning strings (which can include phrases like "SARIMA needs at
least 24 observations" or a wrapped exception message) are now passed
through a new frontend-only `humanizeWarning()` function before display.
Recognised patterns are rewritten in plain language (e.g. "Prophet was not
evaluated because a required component is not installed."); anything
unrecognised has its raw technical suffix stripped rather than shown
verbatim. The section is hidden entirely when there are no warnings.
**Why:** the task gives an explicit example — showing
`'Prophet' object has no attribute 'stan_backend'` instead of "Prophet was
not evaluated because the dataset is too short" — as exactly the failure
mode to avoid. The backend's warning strings themselves were not changed
(that would be backend logic); the translation happens entirely in the
browser.

## Other Redesign Details

- **Colour discipline.** Every colour in the new UI now maps to one of
  four meanings: green = good/selected, orange = warning/skipped, red =
  reserved for total failure only (e.g. "no model could be trained"), blue
  = primary action. The old Model Comparison table used the same
  red/amber/green pill for "Unavailable" as for "Failed" as for
  percentages; the redesign gives "Unavailable" and untrained rows a
  neutral grey (`.conf-neutral`) instead, since a model not being
  installed is not a system failure and shouldn't read as one.
- **Whitespace, shadows, typography.** Hero numbers are set at 30px/700
  weight; cards use a softer shadow and larger radius; the hero and
  accordion sections use flat background panels (`var(--bg)`) instead of
  bordered boxes to reduce visual noise, per the task's "reduce borders,
  use subtle shadows" instruction.
- **Responsiveness.** The hero grid and monthly-card grid both use
  `repeat(auto-fit, minmax(...))` (the same responsive idiom already used
  elsewhere in this codebase) and step down to 2 columns then 1 column at
  900px/560px breakpoints, so the page stays usable on laptop-width
  screens without horizontal scrolling.
- **Nothing was removed.** The legacy baseline forecast (existing
  `analytics.run_forecast`, fetched via the unchanged `/api/forecast`
  endpoint) still loads automatically and populates the hero/chart/monthly
  cards immediately, exactly as before — it just no longer gets its own
  separate "3-Month Sales Forecast" card competing for attention with the
  richer Auto Forecasting Engine result. Once "Run Auto Forecast" is
  clicked, the same elements are repainted with the model-selected result
  instead of showing a second, duplicate forecast section. Changing the
  horizon after an auto-forecast has run now re-runs the auto engine
  (rather than silently reverting to the simpler baseline), which is a
  UI-orchestration improvement, not a backend change — `/api/forecast` and
  `/api/forecast/auto` are called exactly as they were before this phase.

## How This Improves Usability

A business user landing on the Forecast page now sees, without scrolling
or clicking anything: what the forecast is (Expected Revenue), which
model produced it, and how confident the system is — the three questions
the task states a business user should immediately understand. Everything
that used to compete for attention at the same visual weight (dataset
statistics, per-model MAE/RMSE tables, candidate-model reasoning) is still
fully present and unchanged in content, but now requires a deliberate
click to reveal, matching the "progressive disclosure" and "reduce
clutter" goals. The chart — the single most informative visual for a
non-technical stakeholder — is now the largest, most detailed element on
the page instead of one of six equally-sized boxes.

## Files Modified

| File | Change |
|---|---|
| `ui/templates/index.html` | `#page-forecast` section fully restructured into the 7-section layout. No other `<section>` touched. Two element IDs (`forecast-title-text`, `forecast-horizon`) were deliberately kept so the shared `loadData()` bootstrap function in `app.js` required zero changes. |
| `ui/static/js/app.js` | Rewrote `renderForecast`, `updateForecast`, `renderCombinedChart`, `drawLineChart`, `renderForecastAnalysis` (internals only, same signature), `renderAutoForecast`. Added `renderMonthCards`, `monthLabel`, `toggleForecastTable`, `toggleAccordion`, `showChartTooltip`, `hideChartTooltip`, `humanizeWarning`, `renderForecastWarnings`. Every KPI/Anomaly/Schema/Chat/Duplicate/Overview function, and `loadData()` itself, is untouched (verified byte-identical). |
| `ui/static/css/style.css` | ~230 lines appended for the new hero cards, accordions, monthly cards, chart tooltip/legend, and warning banners. No existing rule was edited. |
| `DESIGN_CHANGELOG.md` | This file. |

## Verification

- `sha256sum` of all 29 `.py` files, taken immediately before and after
  this change: **identical**.
- Live server test after the redesign: `/api/load`, `/api/kpis`,
  `/api/anomalies`, `/api/forecast`, `/api/forecast/analysis`,
  `/api/forecast/auto`, `/api/query`, `/api/duplicates` all return the
  same values as before the redesign (total revenue ₹2,99,11,606.44,
  legacy forecast ₹12,19,044.31 / ₹12,16,862.50 / ₹12,14,680.69 for
  Jan-Mar 2025, auto-forecast selects "Naive" at 49% confidence — all
  unchanged).
- `humanizeWarning()` tested directly (Node.js) against the exact raw
  strings the backend produces, plus the literal
  `'Prophet' object has no attribute 'stan_backend'` example from the
  brief (wrapped in its real `"Prophet failed to train: ..."` prefix) —
  all six cases render as clean, jargon-free sentences with no attribute
  names, quotes, or exception text visible.
- Every `id` referenced by the rewritten JavaScript was cross-checked
  against the rewritten HTML (and vice versa) — no dangling references.
- CSS brace-balance and parse-validity checked (224/224 balanced, 254
  rules parsed).
