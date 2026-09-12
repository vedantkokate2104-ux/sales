# IEEE Paper Package

**"Evidence-Grounded Automated Forecasting for Enterprise Sales Data: Grain
Selection, Leakage-Free Validation and Conformal Reliability"**

Everything here was produced from this repository's own code and the bundled
`data/erp_export.csv`. No number in the paper was estimated, rounded from
memory, or carried over from another build.

---

## Contents

```
paper/
├── main.tex                       IEEEtran conference paper (compile this)
├── references.bib                 30 verified references
├── check_tex.py                   static consistency checker for main.tex + bib
├── figures/                       8 vector figures (PDF), generated from results
├── experiments/
│   ├── exp1_data_and_grain.py     dataset, schema mapping, identity, grain
│   ├── exp2_forecast.py           end-to-end forecast + full-registry benchmark
│   ├── exp3_ablations.py          grain / protocol / leakage / feature ablations
│   ├── exp4_more.py               rank agreement, active-leakage, metrics, segments
│   ├── make_figures.py            regenerates every figure from results/*.json
│   └── results/                   raw JSON outputs (the paper's source of truth)
├── LITERATURE_MATRIX.md           30-paper review matrix → research gap
├── REFERENCE_VERIFICATION.md      per-reference verification record
└── PROJECT_TO_PAPER_MAPPING.md    every claim → source file → experiment output
```

---

## Building the PDF

`pdflatex` was **not available on the machine where this package was
generated**, so the PDF has not been produced here. The source has instead been
validated structurally (see below). To build:

```bash
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

You need `IEEEtran.cls`, which ships with TeX Live and MiKTeX and is
pre-installed on Overleaf. On Overleaf: upload `main.tex`, `references.bib` and
the `figures/` folder, set the compiler to pdfLaTeX, and it will build as is.

If MiKTeX is not installed on Windows:

```bash
winget install MiKTeX.MiKTeX
```

### What has been checked instead

```bash
python check_tex.py
```

Current output:

```
labels: 35   refs: 35   cites: 30   bib entries: 30
No structural problems found.
```

`check_tex.py` verifies: brace balance, `\begin`/`\end` matching, duplicate
labels, undefined `\ref`/`\eqref`, `\cite` keys with no bib entry, bib entries
never cited, duplicate bib keys, missing figure files, and tabular rows whose
cell count disagrees with the column specification. It is not a LaTeX compiler,
so it cannot rule out overfull boxes or float-placement warnings.

**Before submitting**, fill in the affiliation placeholders in the author block
of `main.tex` (marked with a `TODO` comment) and check the page count against
your venue's limit — at roughly 9,700 source words with 8 figures and 11
tables, this will run longer than a 6-page limit and sections would need to be
cut. §II (Related Work) and §VI-F/§VI-I are the natural candidates.

---

## Reproducing every number

```bash
cd paper/experiments
python exp1_data_and_grain.py      #  ~40 s  (loads the E5 embedding model)
python exp2_forecast.py            #  ~5 min (Prophet failure handling dominates)
python exp3_ablations.py           #  ~8 min (weekly-grain backtests dominate)
python exp4_more.py                #  ~5 min (12 segment pipelines)
python make_figures.py             #  ~5 s
```

Each script writes `results/expN.json`; `make_figures.py` reads those and
rewrites `../figures/*.pdf` (and `.png` previews). The scripts import the
project's own modules from `src/` and re-implement nothing, with two
deliberate exceptions, both labelled as such in the paper:

1. `exp4_more.py :: epsilon_mape()` — a reimplementation of the *rejected*
   ε-guarded MAPE formulation, so it can be compared against the pipeline's own
   metric (§VI-F).
2. `exp4_more.py` temporarily sets `backtest.OUTLIER_IQR_MULTIPLIER = 1.5` for
   the leakage sensitivity analysis, and restores it. The deployed value is
   3.0; the paper says so explicitly (§VI-E).

### Environment used

Python 3.11 on Windows 11 — pandas 2.2.0, NumPy 1.26.4, scikit-learn 1.4.2,
statsmodels 0.14.2, XGBoost 2.1.1, SciPy 1.16.3, sentence-transformers 2.7.0,
matplotlib 3.9.2. Prophet 1.1.5 was installed but its compiled Stan backend was
non-functional (`'Prophet' object has no attribute 'stan_backend'`); the paper
reports this rather than omitting it, because it exercised the pipeline's
fault-isolation path.

---

## Headline results

| Finding | Value | Source |
|---|---|---|
| Naive identifier dedup destroys | 54.08 % of rows, 56.65 % of revenue | `exp1.json` |
| Grain choice (weekly → monthly) improves best WAPE by | 21.55 pp (69.03 → 47.48) | `exp3.json` |
| Single hold-out understates its own chosen model's error by | 52.37 pp (11.52 vs 63.89) | `exp3.json` |
| Hold-out vs walk-forward rank correlation | ρ = −0.381 (p = 0.352) | `exp4.json` |
| Whole-series preprocessing understates error by | 2.91–6.28 pp (mean 4.30) | `exp4.json` |
| Best model under the correct protocol | SES, 47.48 % WAPE | `exp2.json` |
| ML configurations beating it by ≥ 1 pp | 0 of 15 | `exp3.json` |
| Published 80 % interval coverage | 66.7 % over 21 points | `exp2.json` |
| Reliability rating | 0.602, MODERATE | `exp2.json` |

---

## Academic-integrity notes

- Every experimental number in `main.tex` traces to a key in
  `experiments/results/*.json`. `PROJECT_TO_PAPER_MAPPING.md` gives the trace
  for each one.
- The paper claims no new algorithm. It is framed as an integrated empirical
  and architectural study, and the conclusion says so explicitly.
- Null and negative results are reported: the leakage guard is inert at the
  deployed setting; hyperparameter tuning hurt two of three tunable models;
  the ensemble was measured and rejected; the intervals under-cover; and
  quarterly aggregation achieves a lower WAPE than the grain the system
  actually selects.
- Prophet is absent from every table because it could not run here, and §V-C
  and §VIII say so rather than leaving a silent gap.
- Correctness of the 21 schema mappings rests on manual inspection, not an
  independent labelled benchmark; §VIII states this limitation.
