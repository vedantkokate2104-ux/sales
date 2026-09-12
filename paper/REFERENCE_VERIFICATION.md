# Reference Verification Report

Every one of the 30 references in `references.bib` was checked against a
publisher or archive record before inclusion. Nothing in the bibliography was
written from memory; any candidate that could not be verified was dropped
rather than included with an approximate citation.

**What was verified for each entry:** title, full author list, year, venue,
volume/issue, page range, and DOI (or arXiv identifier for the single
preprint).

**Numbering** below is the IEEE reference number, i.e. order of first citation
in `main.tex`.

---

### [1] Makridakis, Spiliotis, Assimakopoulos (2018)
- **Title:** Statistical and Machine Learning forecasting methods: Concerns and ways forward
- **Venue:** PLOS ONE 13(3), e0194889
- **DOI:** 10.1371/journal.pone.0194889
- **Verified via:** PLOS ONE article record (journals.plos.org), PubMed 29584784
- **Relevance:** External evidence that ML methods are dominated by statistical methods on short monthly series — the population-level counterpart of our Table V result.

### [2] Makridakis, Spiliotis, Assimakopoulos (2022)
- **Title:** M5 accuracy competition: Results, findings, and conclusions
- **Venue:** International Journal of Forecasting 38(4), 1346–1364
- **DOI:** 10.1016/j.ijforecast.2021.11.013
- **Verified via:** ScienceDirect article record; UNIC research portal record
- **Relevance:** Establishes when complex methods do win (data-rich, hierarchical retail) and that gains shrink at sparse levels — the boundary condition for our finding.

### [3] Makridakis, Spiliotis, Assimakopoulos, Chen, Gaba, Tsetlin, Winkler (2022)
- **Title:** The M5 uncertainty competition: Results, findings and conclusions
- **Venue:** International Journal of Forecasting 38(4), 1365–1385
- **DOI:** 10.1016/j.ijforecast.2021.10.009
- **Verified via:** ScienceDirect article record; INSEAD publication record
- **Relevance:** Motivates auditing interval quality empirically instead of asserting nominal levels (§III-G4, §VII-D).

### [4] Spiliotis, Makridakis, Semenoglou, Assimakopoulos (2022)
- **Title:** Comparison of statistical and machine learning methods for daily SKU demand forecasting
- **Venue:** Operational Research 22(3), 3037–3061
- **DOI:** 10.1007/s12351-020-00605-2
- **Verified via:** Springer article record (link.springer.com)
- **Relevance:** The closest published setting to ours (real business demand, statistical vs ML head-to-head).

### [5] Tashman (2000)
- **Title:** Out-of-sample tests of forecasting accuracy: an analysis and review
- **Venue:** International Journal of Forecasting 16(4), 437–450
- **DOI:** 10.1016/S0169-2070(00)00065-0
- **Verified via:** ScienceDirect article record; Semantic Scholar record
- **Relevance:** The standard reference for rolling-origin evaluation, which our shared fold plan implements and our E4 experiment quantifies.

### [6] Bergmeir, Benítez (2012)
- **Title:** On the use of cross-validation for time series predictor evaluation
- **Venue:** Information Sciences 191, 192–213
- **DOI:** 10.1016/j.ins.2011.12.028
- **Verified via:** ScienceDirect article record; ACM DL record; Monash research portal
- **Relevance:** Empirical basis for preferring multi-origin evaluation over a single hold-out; our rank-inversion result is a concrete instance.

### [7] Hyndman, Koehler (2006)
- **Title:** Another look at measures of forecast accuracy
- **Venue:** International Journal of Forecasting 22(4), 679–688
- **DOI:** 10.1016/j.ijforecast.2006.03.001
- **Verified via:** ScienceDirect article record; RePEc listing v22y2006i4p679-688
- **Relevance:** Documents percentage-error failure near zero actuals — the mechanism our E6 experiment reproduces at 6.4 × 10¹⁴ %.

### [8] Kolassa (2020)
- **Title:** Why the "best" point forecast depends on the error or accuracy measure
- **Venue:** International Journal of Forecasting 36(1), 208–211
- **DOI:** 10.1016/j.ijforecast.2019.02.017
- **Verified via:** ScienceDirect article record; Lancaster University research portal
- **Relevance:** Justifies fixing one primary metric as part of the specification rather than reporting several and choosing after the fact.

### [9] Meisenbacher, Turowski, Phipps, Rätz, Müller, Hagenmeyer, Mikut (2022)
- **Title:** Review of automated time series forecasting pipelines
- **Venue:** WIREs Data Mining and Knowledge Discovery 12(6), e1475
- **DOI:** 10.1002/widm.1475
- **Verified via:** Wiley Online Library article record; arXiv:2202.01712; RWTH Aachen repository copy
- **Relevance:** Primary evidence for the claim that automated pipelines begin downstream of the decisions this paper studies.

### [10] Shchur, Turkmen, Erickson, Shen, Shirkov, Hu, Wang (2023)
- **Title:** AutoGluon–TimeSeries: AutoML for Probabilistic Time Series Forecasting
- **Venue:** Proc. 2nd Int. Conf. on Automated Machine Learning (AutoML), PMLR 224, article 9
- **Verified via:** PMLR proceedings record (proceedings.mlr.press/v224/shchur23a.html); AutoML 2023 accepted-papers list; arXiv:2308.05566
- **Note:** PMLR volumes carry no DOI; the proceedings URL is the canonical record.
- **Relevance:** The closest system-level comparator; occupies the "assumes a prepared series" row of Table I.

### [11] Montero-Manso, Athanasopoulos, Hyndman, Talagala (2020)
- **Title:** FFORMA: Feature-based forecast model averaging
- **Venue:** International Journal of Forecasting 36(1), 86–92
- **DOI:** 10.1016/j.ijforecast.2019.02.011
- **Verified via:** ScienceDirect article record; Monash research portal; RePEc v36y2020i1p86-92
- **Relevance:** Represents corpus-trained meta-learning, which a single organisation cannot apply — the reason our selector uses per-series evidence.

### [12] Talagala, Hyndman, Athanasopoulos (2023)
- **Title:** Meta-learning how to forecast time series
- **Venue:** Journal of Forecasting 42(6), 1476–1501
- **DOI:** 10.1002/for.2963
- **Verified via:** Wiley Online Library article record; Monash research portal
- **Relevance:** As above (FFORMS variant, model selection rather than weighting).

### [13] Petropoulos et al. (2022)
- **Title:** Forecasting: theory and practice
- **Venue:** International Journal of Forecasting 38(3), 705–871
- **DOI:** 10.1016/j.ijforecast.2021.11.001
- **Verified via:** Oxford ORA record; Bristol and Monash research portals; IJF volume 38 page range confirmed
- **Note:** This is a 68-author collaborative review; the `.bib` entry lists the leading authors followed by the final author, as is conventional for very large author lists. Expand if your venue requires the complete list.
- **Relevance:** General framing reference for current forecasting theory and practice.

### [14] Hyndman, Koehler, Snyder, Grose (2002)
- **Title:** A state space framework for automatic forecasting using exponential smoothing methods
- **Venue:** International Journal of Forecasting 18(3), 439–454
- **DOI:** 10.1016/S0169-2070(01)00110-8
- **Verified via:** Monash research portal; RePEc v18y2002i3p439-454
- **Relevance:** Theoretical basis of the exponential-smoothing family (SES, Holt, Holt-Winters) that wins our comparison.

### [15] Hyndman, Khandakar (2008)
- **Title:** Automatic Time Series Forecasting: The forecast Package for R
- **Venue:** Journal of Statistical Software 27(3), 1–22
- **DOI:** 10.18637/jss.v027.i03
- **Verified via:** JSS article record (jstatsoft.org/v027/i03); CRAN vignette
- **Relevance:** The canonical automatic-forecasting baseline, whose `ts`-object input assumption our grain stage removes.

### [16] Kaufman, Rosset, Perlich, Stitelman (2012)
- **Title:** Leakage in data mining: Formulation, detection, and avoidance
- **Venue:** ACM Transactions on Knowledge Discovery from Data 6(4), article 15
- **DOI:** 10.1145/2382577.2382579
- **Verified via:** ACM Digital Library record; Tel Aviv University CRIS record
- **Relevance:** Defines the leakage class our per-fold preprocessing prevents and our E5 experiment measures.

### [17] Davydenko, Fildes (2013)
- **Title:** Measuring forecasting accuracy: The case of judgmental adjustments to SKU-level demand forecasts
- **Venue:** International Journal of Forecasting 29(3), 510–522
- **DOI:** 10.1016/j.ijforecast.2012.09.002
- **Verified via:** ScienceDirect article record; Lancaster EPrints 67185; RePEc v29y2013i3p510-522
- **Relevance:** The canonical argument for aggregate (ratio-of-sums) error measures such as WAPE over averaged per-point ratios.

### [18] Vovk, Gammerman, Shafer (2005)
- **Title:** Algorithmic Learning in a Random World
- **Venue:** Springer, New York (book)
- **DOI:** 10.1007/b106715
- **Verified via:** Royal Holloway and Rutgers research portals; publisher record; authors' companion site alrw.net
- **Relevance:** Foundational conformal-prediction reference; also the source of the exchangeability caveat we report against.

### [19] Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018)
- **Title:** Distribution-Free Predictive Inference for Regression
- **Venue:** Journal of the American Statistical Association 113(523), 1094–1111
- **DOI:** 10.1080/01621459.2017.1307116
- **Verified via:** Taylor & Francis article record; arXiv:1604.04173
- **Relevance:** The split-conformal construction and order-statistic attainability condition implemented in eq. (7).

### [20] Xu, Xie (2021)
- **Title:** Conformal prediction interval for dynamic time-series
- **Venue:** Proc. 38th Int. Conf. on Machine Learning (ICML), PMLR 139, 11559–11569
- **Verified via:** PMLR proceedings record (proceedings.mlr.press/v139/xu21h.html); ICML 2021 virtual programme (long presentation)
- **Note:** PMLR volumes carry no DOI.
- **Relevance:** The principled remedy for the exchangeability violation behind our measured under-coverage; named in Limitations and Future Work.

### [21] Gneiting, Raftery (2007)
- **Title:** Strictly Proper Scoring Rules, Prediction, and Estimation
- **Venue:** Journal of the American Statistical Association 102(477), 359–378
- **DOI:** 10.1198/016214506000001437
- **Verified via:** Taylor & Francis article record; University of Washington Statistics technical-report record
- **Relevance:** Justifies scoring intervals with a proper rule rather than reporting coverage alone.

### [22] Winkler (1972)
- **Title:** A Decision-Theoretic Approach to Interval Estimation
- **Venue:** Journal of the American Statistical Association 67(337), 187–191
- **DOI:** 10.1080/01621459.1972.10481224
- **Verified via:** Taylor & Francis article record (Vol. 67, No. 337)
- **Relevance:** The interval score implemented in eq. (8), which prevents achieving coverage by widening.

### [23] Athanasopoulos, Hyndman, Kourentzes, Petropoulos (2017)
- **Title:** Forecasting with temporal hierarchies
- **Venue:** European Journal of Operational Research 262(1), 60–74
- **DOI:** 10.1016/j.ejor.2017.02.046
- **Verified via:** Monash research portal; RePEc v262y2017i1p60-74; Bath research portal
- **Relevance:** Establishes that temporal aggregation level has measurable forecasting consequences; the future-work alternative to selecting a single grain.

### [24] Rahm, Bernstein (2001)
- **Title:** A survey of approaches to automatic schema matching
- **Venue:** The VLDB Journal 10(4), 334–350
- **DOI:** 10.1007/s007780100057
- **Verified via:** Springer article record; ACM DL record; DBLP journals/vldb/RahmB01
- **Relevance:** Places our rule + embedding mapper in the established hybrid-matcher taxonomy.

### [25] Hulsebos, Hu, Bakker, Zgraggen, Satyanarayan, Kraska, Demiralp, Hidalgo (2019)
- **Title:** Sherlock: A Deep Learning Approach to Semantic Data Type Detection
- **Venue:** Proc. 25th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), 1500–1508
- **DOI:** 10.1145/3292500.3330993
- **Verified via:** ACM Digital Library record; MIT Visualization Group publication page; arXiv:1905.10688
- **Relevance:** The learned, value-based alternative to name-and-glossary column mapping.

### [26] Wang, Yang, Huang, Jiao, Yang, Jiang, Majumder, Wei (2022)
- **Title:** Text Embeddings by Weakly-Supervised Contrastive Pre-training
- **Venue:** arXiv preprint arXiv:2212.03533
- **DOI:** 10.48550/arXiv.2212.03533
- **Verified via:** arXiv listing 2212.03533 (v1, v2); Hugging Face papers record; Semantic Scholar record
- **Note:** The only non-peer-reviewed entry. It is included because it is the primary description of the exact model (`intfloat/e5-large-v2`) the implementation loads; no peer-reviewed substitute describes that model.
- **Relevance:** Documents the embedding model used in the semantic schema-mapping stage.

### [27] Elmagarmid, Ipeirotis, Verykios (2007)
- **Title:** Duplicate Record Detection: A Survey
- **Venue:** IEEE Transactions on Knowledge and Data Engineering 19(1), 1–16
- **DOI:** 10.1109/TKDE.2007.250581
- **Verified via:** IEEE/ACM DL records; Purdue author copy (TKDE-0240-0605-1)
- **Relevance:** Frames record identity as a first-class problem and contextualises the composite-business-key rule.

### [28] Bates, Granger (1969)
- **Title:** The Combination of Forecasts
- **Venue:** Operational Research Quarterly 20(4), 451–468
- **DOI:** 10.1057/jors.1969.103
- **Verified via:** Springer / Taylor & Francis records for the journal (now Journal of the Operational Research Society); CRAN `ForecastComb`/`GeomComb` method documentation
- **Note:** The journal was titled *Operational Research Quarterly* in 1969 and is indexed today under its successor title, *Journal of the Operational Research Society*. Either journal name is acceptable; the volume, pages and DOI are unaffected.
- **Relevance:** The inverse-error combination weights of eq. (6).

### [29] Taylor, Letham (2018)
- **Title:** Forecasting at Scale
- **Venue:** The American Statistician 72(1), 37–45
- **DOI:** 10.1080/00031305.2017.1380080
- **Verified via:** Taylor & Francis article record (Vol. 72, No. 1); PeerJ Preprints 3190
- **Relevance:** Describes Prophet, one of the nine registry models.

### [30] Chen, Guestrin (2016)
- **Title:** XGBoost: A Scalable Tree Boosting System
- **Venue:** Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), 785–794
- **DOI:** 10.1145/2939672.2939785
- **Verified via:** ACM Digital Library record; arXiv:1603.02754; Semantic Scholar record
- **Relevance:** The gradient-boosting implementation used in the model registry and throughout the feature ablation.

---

## Integrity notes

- **No fabricated entries.** Every title, author list, venue, volume, page
  range and DOI above was taken from a publisher or archive record, not
  reconstructed.
- **Two entries have no DOI by design** ([10] and [20]). Both are PMLR
  proceedings volumes, which are not assigned DOIs; the canonical
  `proceedings.mlr.press` record is cited instead.
- **One preprint** ([26]) is included, with the reason stated in its entry.
- **Every reference is cited** in `main.tex`, and every citation resolves to
  an entry here — verified mechanically by `check_tex.py`
  (30 cite keys, 30 bib entries, zero unmatched in either direction).
- **Candidates that were dropped.** Several verified papers were removed
  during trimming to stay within the requested 25–30 range, on the grounds
  that they were not load-bearing for any specific claim: the M4 competition
  report, Januschowski et al. on classifying forecasting methods, He et al.'s
  AutoML survey, Croston (1972), Syntetos & Boylan (2005), Nikolopoulos et
  al. (ADIDA), Kourentzes et al. (MAPA), Salinas et al. (DeepAR), Hewamalage
  et al. (RNNs), Suhara et al. (Doduo), Abedjan et al. (data errors),
  Stankevičiūtė et al. (conformal RNN) and Fildes et al. (judgmental
  adjustments). They are listed here so the trimming decision is transparent
  and so they can be reinstated if a section is expanded.
