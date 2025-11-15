"""Example usage: simulate ARIMAX-like data, fit model with statsmodels, and plot results.

This example shows how to:
- simulate a simple ARIMAX process with one exogenous variable
- fit the improved `ARIMAX` wrapper (uses `statsmodels` under the hood)
- print summary and diagnostics
- produce forecasts with confidence intervals and plot results
"""

import numpy as np
from .arimax import ARIMAX
from .plotting import plot_series_and_forecast


def simulate_arimax(n=200, ar_coef=None, ma_coef=None, exog_coef=None, d=0, seed=1):
    rng = np.random.RandomState(seed)
    eps = rng.normal(scale=1.0, size=n + 100)
    ar_coef = np.asarray(ar_coef) if ar_coef is not None else np.array([])
    ma_coef = np.asarray(ma_coef) if ma_coef is not None else np.array([])
    p = len(ar_coef)
    q = len(ma_coef)
    x = rng.normal(size=(n + 100, 1)) if exog_coef is not None else None
    exog_coef = np.asarray(exog_coef) if exog_coef is not None else None

    y = np.zeros(n + 100)
    errors = np.zeros(n + 100)
    for t in range(n + 100):
        ar_part = 0.0
        for j in range(1, p + 1):
            if t - j >= 0:
                ar_part += ar_coef[j - 1] * y[t - j]
        ma_part = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                ma_part += ma_coef[j - 1] * errors[t - j]
        exog_part = 0.0
        if x is not None:
            exog_part = x[t, :].dot(exog_coef)
        y[t] = ar_part + ma_part + exog_part + eps[t]
        errors[t] = eps[t]

    y = y[100:]
    if x is not None:
        x = x[100:]
    if d > 0:
        for _ in range(d):
            y = np.diff(y)
    return y, x


def run_example():
    # simulate
    ar = [0.6]
    ma = [0.2]
    ex_coef = [0.5]
    y, x = simulate_arimax(n=300, ar_coef=ar, ma_coef=ma, exog_coef=ex_coef, d=0)

    # fit with statsmodels-based wrapper
    model = ARIMAX(p=1, d=0, q=1, trend='c')
    model.fit(y, exog=x, disp=False)

    # summary
    print(model.summary())

    # diagnostics (opens a matplotlib figure)
    model.plot_diagnostics(figsize=(10, 8))

    # forecast 30 steps ahead, providing future exog (for demo re-use first 30 rows)
    exog_future = x[:30] if x is not None else None
    mean_forecast, ci = model.predict(steps=30, exog=exog_future, return_conf_int=True)

    # plot series and forecast with confidence interval
    fig, ax = plot_series_and_forecast(y, mean_forecast, conf_int=ci, title="ARIMAX fit example")
    fig.show()


if __name__ == "__main__":
    run_example()
