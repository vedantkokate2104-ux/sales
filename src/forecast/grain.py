"""
grain.py
---------
Forecast grain selection and forecasting-series construction (Phase 1,
Steps 3-6).

TRANSACTION GRAIN IS NOT FORECAST GRAIN
----------------------------------------
The ERP table and the forecasting series are two different objects:

  TRANSACTION GRAIN — one row per ERP record. Its cadence is however
    often the business happens to raise invoices. It is a property of
    paperwork.

  FORECAST GRAIN — the period at which demand is aggregated and modelled
    (daily / weekly / monthly / quarterly). It is a property of the
    signal you want to forecast.

`utils.infer_frequency()` measures the first and the previous pipeline
used it directly as the second. On the bundled dataset that produced
"weekly", because invoices arrive every ~4 days on average — but the
resulting weekly series is 43% empty periods, while the same data
aggregated monthly is essentially complete. Inter-arrival time answers
"how often does this business invoice", which is not the same question as
"at what period is this demand forecastable".

This module therefore evaluates every candidate grain on its own merits
and picks one, rather than deriving it from arrival gaps.

MISSING IS NOT ZERO
--------------------
`trainer.build_series()` reindexed onto a gapless period range with
`fill_value=0.0`, which silently asserts that a period with no invoice had
zero demand. Those are three distinguishable things:

  1. EXPLICIT ZERO   — the period was observed and genuinely totalled zero
                       (e.g. sales exactly offset by refunds).
  2. MISSING PERIOD  — no transaction was recorded. Could be true zero
                       demand, could be a recording gap. Unknown.
  3. NO DATA AT ALL  — outside the observed date range entirely.

`build_forecast_series()` keeps (1) and (2) apart via an explicit
`observed_mask`, and applies a fill strategy chosen from the series'
own density (see `resolve_fill_strategy`) rather than assuming zero
everywhere. Downstream statistics can then report "12% of modelled
periods are zero" and "4% of periods were never observed" as the separate
facts they are.

LEAKAGE
--------
Grain selection must not be tuned on the data a model is later scored
against. Two guarantees, both tested:

  1. Selection never trains, fits, or scores any forecasting model. It
     reads only structural properties (period counts, density, dispersion,
     seasonal support). No model error of any kind enters the decision.
  2. The statistics that DRIVE the choice are computed on a leading
     training span only — `SELECTION_HOLDOUT_FRACTION` of the calendar
     range at the end is excluded, so the periods a later walk-forward
     validation will hold out did not influence which grain was chosen.
     The per-candidate diagnostics REPORTED back to the caller are still
     computed over the full series (they are for display and dataset
     understanding); only `selection_basis` statistics decide the winner.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import _clean_numeric
from .preprocessing import parse_dates_deterministic
from .utils import (
    PERIOD_FREQ_CODE,
    detect_seasonality,
    period_freq_code,
    seasonal_period_for_freq,
    trend_strength,
)

logger = logging.getLogger(__name__)

# Candidate forecast grains, finest first. Yearly is deliberately excluded:
# a yearly grain cannot support a meaningful validation window or a
# multi-period horizon on any dataset this project realistically sees.
CANDIDATE_GRAINS: List[str] = ["daily", "weekly", "monthly", "quarterly"]

# ── Rejection thresholds ───────────────────────────────────────────────────
# A grain must produce at least this many modelled periods to be usable at
# all — below it there is not enough history to fit anything beyond a
# flat baseline, let alone validate it.
MIN_PERIODS = 12

# Horizon compatibility: a grain must yield at least this many periods per
# unit of requested horizon, so the series is not almost entirely consumed
# by a single forecast window.
MIN_PERIODS_PER_HORIZON = 4

# Density = observed periods / total periods in range. Below this, the
# series is mostly fabricated fill rather than measured demand, and any
# "pattern" a model finds is dominated by the filling strategy.
MIN_DENSITY = 0.60

# Above this density, gaps are rare enough to treat as recording gaps and
# interpolate. Below it (but above MIN_DENSITY) gaps are common enough
# that interpolating them would invent a large share of the series, so
# they are zero-filled and flagged instead.
INTERPOLATE_DENSITY_THRESHOLD = 0.75

# Runs of missing periods longer than this are never interpolated even at
# high density — a long gap looks far more like genuine inactivity (or a
# systemic reporting break) than a blip worth bridging.
MAX_INTERPOLATED_RUN = 2

# Fraction of the calendar span held out of the SELECTION statistics, so
# the grain decision cannot be influenced by the periods a later
# walk-forward validation will score against. See the module docstring.
SELECTION_HOLDOUT_FRACTION = 0.2


@dataclass
class SeriesConstruction:
    """A forecasting series plus the full provenance of how it was built —
    specifically, which periods were actually observed and which were
    filled, so "zero demand" and "no data" never get conflated downstream."""
    series: pd.Series                 # modelled series, gapless PeriodIndex
    observed_mask: pd.Series          # bool per period: was any transaction recorded?
    frequency: str
    fill_strategy: str
    n_periods: int
    n_observed: int
    n_missing: int                    # periods with no transaction at all
    n_explicit_zero: int              # observed periods whose total was exactly 0
    n_interpolated: int               # missing periods bridged by interpolation
    n_zero_filled: int                # missing periods left/set at 0
    notes: List[str] = field(default_factory=list)

    @property
    def density(self) -> float:
        return round(self.n_observed / self.n_periods, 4) if self.n_periods else 0.0

    def to_dict(self) -> dict:
        return {
            "frequency":       self.frequency,
            "fill_strategy":   self.fill_strategy,
            "n_periods":       self.n_periods,
            "n_observed":      self.n_observed,
            "n_missing":       self.n_missing,
            "n_explicit_zero": self.n_explicit_zero,
            "n_interpolated":  self.n_interpolated,
            "n_zero_filled":   self.n_zero_filled,
            "density":         self.density,
            "notes":           self.notes,
        }


@dataclass
class GrainDiagnostics:
    """Everything measured about one candidate forecast grain. Purely
    descriptive — no model is fit to produce any of these numbers."""
    frequency: str
    n_periods: int
    observed_periods: int
    empty_periods: int
    zero_value_periods: int
    zero_pct: float
    density: float
    mean: float
    std: float
    coefficient_of_variation: float
    trend_strength: float
    seasonality_strength: float
    seasonal_period: Optional[int]
    history_length_days: int
    usable: bool
    rejection_reason: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "frequency":                self.frequency,
            "n_periods":                self.n_periods,
            "observed_periods":         self.observed_periods,
            "empty_periods":            self.empty_periods,
            "zero_value_periods":       self.zero_value_periods,
            "zero_pct":                 self.zero_pct,
            "density":                  self.density,
            "mean":                     self.mean,
            "std":                      self.std,
            "coefficient_of_variation": self.coefficient_of_variation,
            "trend_strength":           self.trend_strength,
            "seasonality_strength":     self.seasonality_strength,
            "seasonal_period":          self.seasonal_period,
            "history_length_days":      self.history_length_days,
            "usable":                   self.usable,
            "rejection_reason":         self.rejection_reason,
            "score":                    self.score,
        }


@dataclass
class GrainSelection:
    """Result of choosing a forecast grain."""
    selected_frequency: str
    candidate_frequencies: Dict[str, GrainDiagnostics]
    reason: str
    selection_basis: str
    fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "selected_frequency":    self.selected_frequency,
            "candidate_frequencies": {k: v.to_dict() for k, v in self.candidate_frequencies.items()},
            "reason":                self.reason,
            "selection_basis":       self.selection_basis,
            "fallback":              self.fallback,
        }


# ── series construction ────────────────────────────────────────────────────

def aggregate_to_period(
    dates: pd.Series, values: pd.Series, frequency: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Sum `values` into whole periods at `frequency`, returning

        (period_totals, observed_mask)

    both indexed by the SAME gapless PeriodIndex spanning first to last
    observation. `period_totals` carries NaN — not 0.0 — for periods with
    no transaction, and `observed_mask` is True exactly where a
    transaction existed. Keeping "no data" as NaN at this layer is what
    lets `build_forecast_series()` make an explicit, documented decision
    about it instead of the old reindex(fill_value=0.0) silently deciding
    that no invoice means no demand.
    """
    # Deterministic parsing (one explicit format applied to the whole
    # column) rather than pandas' per-row inference — see
    # preprocessing.parse_dates_deterministic() for why day-first,
    # 2-digit-year columns like this project's own sample data are a
    # silent day/month-swap risk under inference.
    dates = parse_dates_deterministic(pd.Series(dates))
    valid = dates.notna()
    dates = dates[valid]
    values = _clean_numeric(pd.Series(values)[valid.to_numpy()])

    if len(dates) == 0:
        empty_idx = pd.PeriodIndex([], freq=period_freq_code(frequency))
        return pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=bool, index=empty_idx)

    code = period_freq_code(frequency)
    periods = dates.dt.to_period(code)
    totals = values.groupby(periods).sum().sort_index()

    full_range = pd.period_range(totals.index.min(), totals.index.max(), freq=code)
    observed_mask = pd.Series(full_range.isin(totals.index), index=full_range)
    period_totals = totals.reindex(full_range)  # missing -> NaN, deliberately not 0.0
    return period_totals, observed_mask


def resolve_fill_strategy(density: float) -> str:
    """
    Choose how missing periods are handled, from the series' own density.

    Documented rationale (Phase 1, Step 5) — the right treatment of a
    missing period depends on how many of them there are:

      density >= 0.99  ->  "none_needed"
          Effectively no gaps. Nothing is invented.

      density >= 0.75  ->  "interpolate_short_gaps"
          Gaps are the exception. A run of <= MAX_INTERPOLATED_RUN missing
          periods bounded by real activity on both sides is far more
          likely a recording gap than a genuine collapse to zero demand,
          so it is bridged linearly. Longer runs are still zero-filled:
          fabricating a long synthetic trend is a much stronger assumption
          than admitting the period was quiet.

      density <  0.75  ->  "zero_filled_sparse"
          Gaps are common. Interpolating them would mean inventing a large
          fraction of the modelled series, so missing periods are set to
          zero AND the series is flagged as sparse. At this density the
          grain itself is usually the problem, which is exactly what grain
          selection is for.
    """
    if density >= 0.99:
        return "none_needed"
    if density >= INTERPOLATE_DENSITY_THRESHOLD:
        return "interpolate_short_gaps"
    return "zero_filled_sparse"


def _interpolate_short_runs(values: np.ndarray, observed: np.ndarray, max_run: int) -> Tuple[np.ndarray, int]:
    """Linearly bridge runs of <= max_run consecutive UNOBSERVED periods
    that have observed activity on both sides. Operates on the observed
    mask, not on zero values, so an explicit zero is never mistaken for a
    gap to be filled in."""
    result = values.copy()
    n = len(values)
    n_filled = 0
    i = 0
    while i < n:
        if observed[i]:
            i += 1
            continue
        j = i
        while j < n and not observed[j]:
            j += 1
        run_len = j - i
        has_left = i > 0 and observed[i - 1]
        has_right = j < n and observed[j]
        if run_len <= max_run and has_left and has_right:
            left_val, right_val = result[i - 1], result[j]
            for k in range(run_len):
                frac = (k + 1) / (run_len + 1)
                result[i + k] = left_val + frac * (right_val - left_val)
            n_filled += run_len
        i = j
    return result, n_filled


def build_forecast_series(
    dates: pd.Series,
    values: pd.Series,
    frequency: str,
    fill_strategy: Optional[str] = None,
) -> SeriesConstruction:
    """
    Build the modelled forecasting series at `frequency`, keeping missing
    periods and explicit zeros distinguishable throughout (Phase 1,
    Step 5). `fill_strategy` defaults to `resolve_fill_strategy(density)`;
    pass it explicitly only to override that decision.

    The returned series is gapless and finite (every model in this library
    needs a value at every period), but `observed_mask` and the n_* counts
    record exactly which values were measured and which were supplied, so
    nothing downstream has to guess.
    """
    period_totals, observed_mask = aggregate_to_period(dates, values, frequency)

    if len(period_totals) == 0:
        return SeriesConstruction(
            series=period_totals, observed_mask=observed_mask, frequency=frequency,
            fill_strategy="none_needed", n_periods=0, n_observed=0, n_missing=0,
            n_explicit_zero=0, n_interpolated=0, n_zero_filled=0,
            notes=["No valid dated observations — no forecasting series could be built."],
        )

    observed = observed_mask.to_numpy(dtype=bool)
    raw = period_totals.to_numpy(dtype=float)

    n_periods = len(raw)
    n_observed = int(observed.sum())
    n_missing = n_periods - n_observed
    n_explicit_zero = int(((raw == 0.0) & observed).sum())
    density = n_observed / n_periods if n_periods else 0.0

    if fill_strategy is None:
        fill_strategy = resolve_fill_strategy(density)

    filled = np.where(observed, raw, 0.0)
    n_interpolated = 0
    if fill_strategy == "interpolate_short_gaps" and n_missing:
        filled, n_interpolated = _interpolate_short_runs(filled, observed, MAX_INTERPOLATED_RUN)
    n_zero_filled = n_missing - n_interpolated

    notes: List[str] = []
    if n_missing:
        notes.append(
            f"{n_missing} of {n_periods} {frequency} period(s) ({round(n_missing / n_periods * 100, 1)}%) "
            f"had no transaction recorded. These are missing observations, not confirmed zero demand."
        )
    if n_interpolated:
        notes.append(
            f"{n_interpolated} missing period(s) sat in short runs of {MAX_INTERPOLATED_RUN} or fewer "
            f"surrounded by real activity and were interpolated as likely recording gaps."
        )
    if n_zero_filled:
        notes.append(
            f"{n_zero_filled} missing period(s) were set to zero because the gap was too long "
            f"(or the series too sparse) to justify inventing a value."
        )
    if n_explicit_zero:
        notes.append(
            f"{n_explicit_zero} period(s) were observed with a genuine total of zero — these are "
            f"real zero-demand periods, counted separately from missing periods above."
        )
    if fill_strategy == "zero_filled_sparse":
        notes.append(
            f"Series density at {frequency} grain is only {round(density * 100, 1)}% — a large share "
            f"of the modelled series is filled rather than measured."
        )

    return SeriesConstruction(
        series=pd.Series(filled, index=period_totals.index),
        observed_mask=observed_mask,
        frequency=frequency,
        fill_strategy=fill_strategy,
        n_periods=n_periods,
        n_observed=n_observed,
        n_missing=n_missing,
        n_explicit_zero=n_explicit_zero,
        n_interpolated=n_interpolated,
        n_zero_filled=n_zero_filled,
        notes=notes,
    )


# ── candidate evaluation ───────────────────────────────────────────────────

def evaluate_grain(
    dates: pd.Series, values: pd.Series, frequency: str, horizon: int = 3
) -> GrainDiagnostics:
    """
    Measure one candidate grain and decide whether it is usable. Purely
    descriptive: no model is fit anywhere in here, which is half of this
    module's leakage guarantee (see module docstring).

    Rejection is applied for too little history, horizon incompatibility,
    or excessive sparsity — each with a stated reason rather than a silent
    drop.
    """
    construction = build_forecast_series(dates, values, frequency)
    series = construction.series
    n = construction.n_periods

    if n == 0:
        return GrainDiagnostics(
            frequency=frequency, n_periods=0, observed_periods=0, empty_periods=0,
            zero_value_periods=0, zero_pct=0.0, density=0.0, mean=0.0, std=0.0,
            coefficient_of_variation=0.0, trend_strength=0.0, seasonality_strength=0.0,
            seasonal_period=None, history_length_days=0, usable=False,
            rejection_reason="No observations could be aggregated at this grain.",
        )

    arr = series.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    cv = round(abs(std / mean), 4) if mean else 0.0

    # Zero periods in the MODELLED series — what a model actually sees.
    # Distinct from empty_periods (never observed), reported alongside it.
    zero_value_periods = int((arr == 0.0).sum())
    zero_pct = round(zero_value_periods / n * 100, 2)
    density = construction.density

    trend = trend_strength(arr) if n >= 3 else 0.0
    seasonality, seasonal_period = detect_seasonality(arr, frequency)

    ts_index = series.index.to_timestamp()
    history_days = int((ts_index[-1] - ts_index[0]).days) if n > 1 else 0

    usable, rejection = True, None
    if n < MIN_PERIODS:
        usable, rejection = False, (
            f"Only {n} {frequency} period(s) — at least {MIN_PERIODS} are needed to model and "
            f"validate a series at this grain."
        )
    elif n < horizon * MIN_PERIODS_PER_HORIZON:
        usable, rejection = False, (
            f"{n} {frequency} period(s) is too short for a {horizon}-period horizon — at least "
            f"{horizon * MIN_PERIODS_PER_HORIZON} periods "
            f"({MIN_PERIODS_PER_HORIZON}x the horizon) are required."
        )
    elif density < MIN_DENSITY:
        usable, rejection = False, (
            f"Only {round(density * 100, 1)}% of {frequency} periods contain any transaction "
            f"(minimum {round(MIN_DENSITY * 100)}%) — at this grain most of the series would be "
            f"filled rather than measured."
        )

    return GrainDiagnostics(
        frequency=frequency,
        n_periods=n,
        observed_periods=construction.n_observed,
        empty_periods=construction.n_missing,
        zero_value_periods=zero_value_periods,
        zero_pct=zero_pct,
        density=density,
        mean=round(mean, 2),
        std=round(std, 2),
        coefficient_of_variation=cv,
        trend_strength=trend,
        seasonality_strength=seasonality,
        seasonal_period=seasonal_period,
        history_length_days=history_days,
        usable=usable,
        rejection_reason=rejection,
    )


# Granularity preference, finest to coarsest. This is a deliberate
# counterweight, not a tie-break flourish: aggregating to a coarser grain
# mechanically raises density (fewer, fuller buckets) and mechanically
# lowers the coefficient of variation (averaging cancels noise), so a
# score built only from density and dispersion would drift toward
# quarterly on almost any dataset. A finer grain that clears every
# quality gate is genuinely more useful — it yields more modelled periods
# for validation to work with and a forecast at a resolution the business
# can act on — and it can always be summed up afterwards, whereas a
# coarse grain cannot be split back down.
_RESOLUTION_PREFERENCE = {"daily": 1.0, "weekly": 0.75, "monthly": 0.5, "quarterly": 0.25}


def _score_grain(d: GrainDiagnostics) -> float:
    """
    Composite 0-1 desirability score for a usable grain. Five weighted
    structural components — none of them involves fitting a model or
    measuring forecast error, which is what keeps grain selection outside
    the evaluation loop entirely:

      density (0.35)      — the strongest single signal. A series that is
                            mostly measured beats one that is mostly filled.
      history (0.25)      — more modelled periods to learn from, saturating
                            at 24 (two annual cycles at monthly grain).
      stability (0.15)    — lower dispersion relative to the mean is more
                            forecastable; saturates at CV 2.0. Weighted
                            below density because it is partly an artefact
                            of aggregation (see _RESOLUTION_PREFERENCE).
      seasonal fit (0.15) — full credit when the series covers at least two
                            complete cycles of a detectable seasonal period.
      resolution (0.10)   — prefer the finest grain that still clears every
                            gate above.
    """
    density_component = min(1.0, max(0.0, d.density))
    history_component = min(1.0, d.n_periods / 24.0)
    stability_component = max(0.0, 1.0 - min(1.0, d.coefficient_of_variation / 2.0))
    seasonal_component = 1.0 if d.seasonal_period else 0.0
    resolution_component = _RESOLUTION_PREFERENCE.get(d.frequency, 0.5)

    return round(
        0.35 * density_component
        + 0.25 * history_component
        + 0.15 * stability_component
        + 0.15 * seasonal_component
        + 0.10 * resolution_component,
        4,
    )


def _training_span_mask(dates: pd.Series, holdout_fraction: float) -> pd.Series:
    """
    Boolean mask selecting the leading (1 - holdout_fraction) of the
    CALENDAR span — the portion of history a later walk-forward validation
    would already have trained on before it starts scoring. Selection
    statistics are restricted to this span so the grain decision cannot be
    shaped by the periods it will later be judged against.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    valid = parsed.dropna()
    if len(valid) < 2 or holdout_fraction <= 0:
        return parsed.notna()
    start, end = valid.min(), valid.max()
    cutoff = start + (end - start) * (1.0 - holdout_fraction)
    return parsed.notna() & (parsed <= cutoff)


def select_forecast_grain(
    dates: pd.Series,
    values: pd.Series,
    horizon: int = 3,
    holdout_fraction: float = SELECTION_HOLDOUT_FRACTION,
) -> GrainSelection:
    """
    Choose the forecast grain for this dataset (Phase 1, Step 6).

    Every candidate in CANDIDATE_GRAINS is evaluated, scored, and the
    highest-scoring USABLE one wins. Nothing is hardcoded to monthly or to
    weekly, and transaction inter-arrival time is never consulted.

    Leakage (see module docstring): the scores that decide the winner are
    computed on the leading `1 - holdout_fraction` of the calendar span
    only. The `candidate_frequencies` diagnostics returned to the caller
    are computed over the FULL series, because they exist to describe the
    dataset, not to make the choice.

    When no candidate is usable, the least-bad one (highest density, then
    most periods) is selected with `fallback=True` and a reason explaining
    what failed — forecasting still proceeds on the best available grain
    rather than the pipeline refusing to produce anything.
    """
    full_diagnostics: Dict[str, GrainDiagnostics] = {}
    for freq in CANDIDATE_GRAINS:
        try:
            full_diagnostics[freq] = evaluate_grain(dates, values, freq, horizon=horizon)
        except Exception:
            logger.exception("Grain evaluation failed for %s — treating as unusable", freq)
            full_diagnostics[freq] = GrainDiagnostics(
                frequency=freq, n_periods=0, observed_periods=0, empty_periods=0,
                zero_value_periods=0, zero_pct=0.0, density=0.0, mean=0.0, std=0.0,
                coefficient_of_variation=0.0, trend_strength=0.0, seasonality_strength=0.0,
                seasonal_period=None, history_length_days=0, usable=False,
                rejection_reason="This grain could not be evaluated on this dataset.",
            )

    # ── selection statistics: training span only ──────────────────────────
    train_mask = _training_span_mask(dates, holdout_fraction)
    train_dates, train_values = dates[train_mask], values[train_mask]
    selection_basis = (
        f"structural diagnostics computed on the leading {round((1 - holdout_fraction) * 100)}% "
        f"of the date range; no model was fit and no forecast error was measured"
    )

    selection_diagnostics: Dict[str, GrainDiagnostics] = {}
    for freq in CANDIDATE_GRAINS:
        try:
            selection_diagnostics[freq] = evaluate_grain(train_dates, train_values, freq, horizon=horizon)
        except Exception:
            logger.exception("Training-span grain evaluation failed for %s", freq)
            selection_diagnostics[freq] = full_diagnostics[freq]

    for freq, diag in selection_diagnostics.items():
        score = _score_grain(diag) if diag.usable else None
        full_diagnostics[freq].score = score

    usable = [(f, d) for f, d in selection_diagnostics.items() if d.usable]

    if not usable:
        # Nothing cleared the bar. Pick the least-bad grain rather than
        # failing outright, and be explicit that this is a fallback.
        ranked = sorted(
            full_diagnostics.items(),
            key=lambda kv: (kv[1].density, kv[1].n_periods),
            reverse=True,
        )
        chosen = ranked[0][0]
        rejections = "; ".join(
            f"{f}: {d.rejection_reason}" for f, d in selection_diagnostics.items() if d.rejection_reason
        )
        return GrainSelection(
            selected_frequency=chosen,
            candidate_frequencies=full_diagnostics,
            reason=(
                f"No candidate grain met the minimum requirements ({rejections}). Fell back to "
                f"'{chosen}' as the densest grain with the most history so a forecast can still be "
                f"produced — treat its output as provisional."
            ),
            selection_basis=selection_basis,
            fallback=True,
        )

    # Highest score wins; ties break toward the finer grain (more modelled
    # periods, so more data for validation to work with).
    scored = sorted(
        usable,
        key=lambda kv: (_score_grain(kv[1]), kv[1].n_periods),
        reverse=True,
    )
    winner_freq, winner = scored[0]
    full_winner = full_diagnostics[winner_freq]

    runners = [
        f"{f} (score {_score_grain(d)}, density {round(d.density * 100, 1)}%, {d.n_periods} periods)"
        for f, d in scored[1:]
    ]
    rejected = [
        f"{f} ({d.rejection_reason})"
        for f, d in selection_diagnostics.items() if not d.usable
    ]

    reason_parts = [
        f"Selected '{winner_freq}' grain: {full_winner.n_periods} modelled periods, "
        f"{round(full_winner.density * 100, 1)}% of them containing real transactions, "
        f"coefficient of variation {full_winner.coefficient_of_variation}"
    ]
    if full_winner.seasonal_period:
        reason_parts.append(
            f"and enough history to test a {full_winner.seasonal_period}-period seasonal cycle "
            f"(strength {full_winner.seasonality_strength})"
        )
    reason = ", ".join(reason_parts) + "."
    if runners:
        reason += " Ranked above " + "; ".join(runners) + "."
    if rejected:
        reason += " Rejected: " + "; ".join(rejected) + "."

    return GrainSelection(
        selected_frequency=winner_freq,
        candidate_frequencies=full_diagnostics,
        reason=reason,
        selection_basis=selection_basis,
        fallback=False,
    )
