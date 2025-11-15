"""ARIMAX package providing modular AR, MA, I (differencing) and exogenous components.
"""
from .ar import make_lag_matrix, ar_predict
from .ma import compute_ma_component
from .i import difference, inverse_difference
from .exog import prepare_exog
from .arimax import ARIMAX
from .model_selection import select_order, select_best_model
from .model_selection import auto_arima_select
from .plotting import plot_series_and_forecast

__all__ = [
    "make_lag_matrix",
    "ar_predict",
    "compute_ma_component",
    "difference",
    "inverse_difference",
    "prepare_exog",
    "ARIMAX",
    "plot_series_and_forecast",
    "select_order",
    "select_best_model",
    "auto_arima_select",
]
