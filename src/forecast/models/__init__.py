"""
forecast.models package
------------------------
The Forecast Model Library (Phase 2, Step 1). Every model in this package
implements the common BaseForecastModel interface (fit / predict / evaluate
/ forecast) so forecast.trainer can train and compare them uniformly.

Models with no external dependency (Moving Average, Naive, Simple
Exponential Smoothing, Holt Linear) are pure numpy/pandas and always
available. Models that depend on an optional third-party library
(Holt-Winters, SARIMA -> statsmodels; Prophet -> prophet; XGBoost ->
xgboost) degrade gracefully via is_available() if that library is not
installed, exactly like Phase 1's embedding_mapper.py falls back when
sentence-transformers is missing.
"""

from .base import BaseForecastModel
from .moving_average import MovingAverageModel
from .naive import NaiveModel
from .seasonal_naive import SeasonalNaiveModel
from .simple_exponential import SimpleExponentialSmoothingModel
from .holt import HoltLinearModel
from .holt_winters import HoltWintersModel
from .sarima import SARIMAModel
from .prophet import ProphetModel
from .xgboost import XGBoostModel

# Registry used by forecast.trainer to look up a model class by the display
# name Phase 1's model_selector.py already uses for recommendations, so the
# two phases speak the same vocabulary without any string translation layer.
MODEL_REGISTRY = {
    "Moving Average":                MovingAverageModel,
    "Naive":                         NaiveModel,
    "Seasonal Naive":                SeasonalNaiveModel,
    "Simple Exponential Smoothing":  SimpleExponentialSmoothingModel,
    "Holt Linear":                   HoltLinearModel,
    "Holt-Winters":                  HoltWintersModel,
    "SARIMA":                        SARIMAModel,
    "Prophet":                       ProphetModel,
    "XGBoost":                       XGBoostModel,
}

__all__ = ["BaseForecastModel", "MODEL_REGISTRY"] + list(MODEL_REGISTRY.keys())
