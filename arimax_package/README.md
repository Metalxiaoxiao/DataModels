
# ARIMAX Package (statsmodels-backed)

This package is intended for mathematical modeling with ARIMAX models. It now wraps the
well-tested `statsmodels` implementation (`SARIMAX`) to provide a reliable, feature-rich
estimation and forecasting tool while keeping a simple API for typical workflows.

Highlights
- `ARIMAX` class: friendly wrapper around `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Forecasting with confidence intervals, model summary, and diagnostic plots.
- Examples showing simulation, fit, diagnostics and plotting.

Files
- `arimax.py`: ARIMAX wrapper exposing `ARIMAX.fit`, `ARIMAX.predict`, `ARIMAX.summary`, `ARIMAX.plot_diagnostics`.
- `plotting.py`: plotting helpers to visualize forecasts with CIs using `seaborn`.
- `example.py`: runnable example that simulates data, fits the model, prints summary, shows diagnostics, and plots forecasts.
- Utility modules (`ar.py`, `ma.py`, `i.py`, `exog.py`) remain for reference but the main modeling uses `statsmodels`.

Installation

Install dependencies (recommended in a virtualenv):

```
pip install -r requirements.txt
```

Quick start

```python
from arimax_package.arimax import ARIMAX

# y: 1-D array, exog: 2-D array or None
model = ARIMAX(p=1, d=0, q=1, trend='c')
model.fit(y, exog=exog, disp=False)
print(model.summary())
mean_forecast, ci = model.predict(steps=20, exog=exog_future, return_conf_int=True)
```

Notes and recommendations
- Use `fit` kwargs to control optimization (e.g., `method`, `maxiter`, `disp`).
- For model selection consider using information criteria (`results.aic`, `results.bic`).
- Diagnostics: call `plot_diagnostics()` on a fitted model to inspect residuals, QQ-plot, and ACF of residuals.

Model selection helper
----------------------

This package includes a simple grid-search helper to choose ARIMA orders by information criteria.

Usage example:

```python
from arimax_package import select_order, select_best_model, auto_arima_select

# search orders (p,d,q) in ranges: p=0..3, d=0..2, q=0..3
# parallel example: use n_jobs>1 for parallel execution; set lang='zh' for Chinese messages
results_table = select_order(y, exog=exog, p_max=3, d_max=2, q_max=3, criterion='aic', n_jobs=4, lang='en')
print(results_table.head())

# get best fitted model (tries best candidates until one converges)
best_res, table = select_best_model(y, exog=exog, p_max=3, d_max=2, q_max=3, n_jobs=4, lang='en')
print(best_res.summary())

# optionally use pmdarima.auto_arima for automatic selection (may choose d/D automatically)
auto_model = auto_arima_select(y, exog=exog, seasonal=False, start_p=0, start_q=0, max_p=5, max_q=5, stepwise=True)
print(auto_model.summary())
```

Notes
- `select_order` returns a `pandas.DataFrame` sorted by the requested criterion (`aic`, `bic`, or `llf`).
- `select_order` supports parallel execution via `n_jobs` (requires `joblib` installed). Set `n_jobs=1` to run serially.
- `select_order` and `select_best_model` accept `lang` parameter (`'en'` or `'zh'`) to control progress/failure messages language.
- `select_best_model` will attempt to refit the top converged candidate and return the fitted `SARIMAXResults`.
- `auto_arima_select` wraps `pmdarima.auto_arima` for automated model selection; it may automatically select differencing orders `d`/`D`.

Example run

```
python -m arimax_package.example
```

This will print a model summary and open diagnostic and forecast plots.

