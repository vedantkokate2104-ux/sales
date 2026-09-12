"""
tabular.py
-----------
A supervised-learning forecaster parameterised by a feature configuration
and an estimator (Phase 3, Steps 8-11).

WHY
----
The Phase 3 ablations need to hold one thing constant while varying
another: the same estimator across different feature sets, and the same
feature set across different estimators. models/xgboost.py cannot do
that — its features and its estimator are welded together, and its
feature set is fixed at 48 columns regardless of how much data exists.

This class takes both as configuration, so "XGBoost with lags only" and
"Ridge with lags plus calendar" are the same code path with different
settings. That is what makes the ablation table a controlled experiment
rather than a collection of separately-written models.

LEAKAGE
--------
Features come from forecast.features.build_feature_frame(), which applies
`shift(1)` before every window, and the SAME function is used for
training rows and for recursive forecast rows. There is exactly one
implementation of "what does the feature row look like at time t", which
is the property that makes the guarantee checkable.

Multi-step forecasting is recursive: predict one period, append the
prediction to the working history, rebuild features from the extended
history, predict the next. Each appended value is the model's own output,
never a real future observation.

OVERFITTING (Step 11)
----------------------
`training_error_` records in-sample WAPE at fit time. The experiment
runner compares it against walk-forward validation error; a model whose
training error is far below its validation error is overfitting, and the
validation number is the one that decides anything.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..evaluator import wape
from ..features import FeatureConfig, build_feature_frame, select_features

logger = logging.getLogger(__name__)

# A supervised model needs at least this many complete feature rows
# before fitting is anything other than memorisation.
MIN_TRAINING_ROWS = 8


class TabularForecastModel(BaseForecastModel):
    """
    Base for feature-driven forecasters. Subclasses (usually produced by
    `make_tabular_model`) set `name`, `estimator_factory` and
    `feature_config` as class attributes so the class can be instantiated
    with no arguments — which is what forecast.backtest requires when it
    constructs a fresh model per fold.
    """

    name: str = "Tabular"
    estimator_factory: Optional[Callable[..., object]] = None
    feature_config: FeatureConfig = FeatureConfig()
    apply_feature_selection: bool = True

    def __init__(self):
        super().__init__()
        self.training_error_: Optional[float] = None
        self.feature_names_: List[str] = []
        self.dropped_features_: Dict[str, str] = {}
        self._estimator = None

    # ── suitability ────────────────────────────────────────────────────────
    def check_requirements(self, series: pd.Series, frequency: str = "monthly") -> Tuple[bool, Optional[str]]:
        X, _y, _specs = build_feature_frame(series, frequency, self.feature_config)
        if len(X) < MIN_TRAINING_ROWS:
            return False, (
                f"{self.name} needs at least {MIN_TRAINING_ROWS} complete feature rows after lags "
                f"are built — only {len(X)} available from {len(series)} periods — skipped."
            )
        return True, None

    # ── fit / predict ──────────────────────────────────────────────────────
    def fit(self, series: pd.Series, frequency: str = "monthly", **hyperparams) -> "TabularForecastModel":
        if self.estimator_factory is None:
            raise ValueError(f"{self.name} has no estimator configured.")

        X, y, specs = build_feature_frame(series, frequency, self.feature_config)
        if len(X) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"{self.name} needs at least {MIN_TRAINING_ROWS} usable training rows — "
                f"only {len(X)} available."
            )

        if self.apply_feature_selection:
            # Fitted on this training window only, exactly like every other
            # preprocessing step since Phase 2.
            X, specs, self.dropped_features_ = select_features(X, specs)

        if X.shape[1] == 0:
            raise ValueError(f"{self.name} has no usable features left after selection.")

        self.series_ = series
        self.frequency_ = frequency
        self.feature_names_ = list(X.columns)

        self._estimator = self.estimator_factory(**hyperparams)
        self._estimator.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))

        # Step 11 — in-sample error, kept for the overfitting comparison.
        try:
            in_sample = self._estimator.predict(X.to_numpy(dtype=float))
            self.training_error_ = wape(y.to_numpy(dtype=float), np.maximum(in_sample, 0.0))
        except Exception:
            self.training_error_ = None

        self.fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"{self.name} must be fit() before predict().")

        working = self.series_.copy()
        predictions: List[float] = []

        for _ in range(steps):
            next_period = working.index[-1] + 1
            # Extend with a placeholder so the feature frame produces a row
            # for `next_period`; the placeholder is never read, because
            # every target-derived feature is shifted by at least one
            # period before any window touches it.
            extended = pd.concat([working, pd.Series([np.nan], index=[next_period])])
            X, _y, _specs = build_feature_frame(
                extended.ffill(), self.frequency_, self.feature_config,
            )
            if next_period not in X.index:
                # Not enough history to build a complete row — fall back to
                # the last observed level rather than fabricating a number.
                value = float(working.iloc[-1])
            else:
                row = X.loc[[next_period]].reindex(columns=self.feature_names_, fill_value=0.0)
                value = float(self._estimator.predict(row.to_numpy(dtype=float))[0])

            value = max(0.0, value)
            predictions.append(value)
            working = pd.concat([working, pd.Series([value], index=[next_period])])

        return np.asarray(predictions, dtype=float)


def make_tabular_model(
    name: str,
    estimator_factory: Callable[..., object],
    feature_config: FeatureConfig,
    apply_feature_selection: bool = True,
) -> type:
    """
    Build a zero-argument-constructible TabularForecastModel subclass.

    forecast.backtest instantiates candidates with `model_cls()`, so an
    experiment cannot pass configuration through the constructor. Baking
    it into a generated subclass keeps that contract intact while letting
    the ablation vary features and estimator independently.
    """
    return type(
        f"Tabular_{name.replace(' ', '_').replace('+', '_')}",
        (TabularForecastModel,),
        {
            "name": name,
            "estimator_factory": staticmethod(estimator_factory),
            "feature_config": feature_config,
            "apply_feature_selection": apply_feature_selection,
        },
    )


# ── Estimator factories ────────────────────────────────────────────────────
# Each returns a fresh, unfitted estimator. Defaults are deliberately
# conservative: this project's series are short, so the priors that matter
# are shallow trees, strong regularisation and few splits.

def _try_sklearn(module: str, cls: str):
    try:
        mod = __import__(f"sklearn.{module}", fromlist=[cls])
        return getattr(mod, cls)
    except Exception:
        return None


def ridge_factory(**hyperparams):
    Ridge = _try_sklearn("linear_model", "Ridge")
    if Ridge is None:
        raise ImportError("scikit-learn is not installed — Ridge is unavailable.")
    params = {"alpha": 1.0, "random_state": 42}
    params.update(hyperparams)
    return Ridge(**params)


def elasticnet_factory(**hyperparams):
    ElasticNet = _try_sklearn("linear_model", "ElasticNet")
    if ElasticNet is None:
        raise ImportError("scikit-learn is not installed — ElasticNet is unavailable.")
    params = {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42, "max_iter": 5000}
    params.update(hyperparams)
    return ElasticNet(**params)


def random_forest_factory(**hyperparams):
    RandomForestRegressor = _try_sklearn("ensemble", "RandomForestRegressor")
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn is not installed — Random Forest is unavailable.")
    params = {
        "n_estimators": 200, "max_depth": 3, "min_samples_leaf": 2,
        "random_state": 42, "n_jobs": 1,
    }
    params.update(hyperparams)
    return RandomForestRegressor(**params)


def gradient_boosting_factory(**hyperparams):
    GradientBoostingRegressor = _try_sklearn("ensemble", "GradientBoostingRegressor")
    if GradientBoostingRegressor is None:
        raise ImportError("scikit-learn is not installed — Gradient Boosting is unavailable.")
    params = {
        "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05,
        "subsample": 0.8, "random_state": 42,
    }
    params.update(hyperparams)
    return GradientBoostingRegressor(**params)


def xgboost_factory(**hyperparams):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is not installed — XGBoost is unavailable.")
    params = {
        "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "verbosity": 0, "n_jobs": 1,
        "objective": "reg:squarederror",
    }
    params.update(hyperparams)
    return XGBRegressor(**params)


ESTIMATOR_FACTORIES: Dict[str, Callable[..., object]] = {
    "Ridge": ridge_factory,
    "ElasticNet": elasticnet_factory,
    "RandomForest": random_forest_factory,
    "GradientBoosting": gradient_boosting_factory,
    "XGBoost": xgboost_factory,
}
