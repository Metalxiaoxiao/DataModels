"""Model selection utilities for ARIMAX using statsmodels.

Provides a grid search over (p, d, q) and returns a results table sorted by AIC/BIC,
and a helper to return the best fitted `SARIMAXResults` object.
"""
from typing import Optional, Tuple
import time
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None


def select_order(y, exog: Optional[np.ndarray] = None, p_max: int = 3, d_max: int = 2, q_max: int = 3,
                 criterion: str = "aic", method: str = "lbfgs", maxiter: int = 50,
                 disp: bool = False, verbose: bool = False, n_jobs: int = 1, lang: str = "en") -> pd.DataFrame:
    """Grid-search (p,d,q) and return a DataFrame of results sorted by `criterion`.

    Parameters
    - y: 1-D endogenous series
    - exog: optional 2-D array of exogenous regressors (aligned with y)
    - p_max, d_max, q_max: inclusive maxima for orders to search (searches 0..p_max etc.)
    - criterion: one of 'aic', 'bic', or 'llf' to sort by (lower is better for aic/bic; higher for llf)
    - method, maxiter, disp: passed to `SARIMAX.fit`
    - verbose: print progress

    Returns
    - DataFrame with columns ['p','d','q','aic','bic','llf','converged','fit_time_seconds'] sorted by criterion
    """
    y = np.asarray(y)
    if exog is not None:
        exog = np.asarray(exog)

    # i18n messages
    msgs = {
        "en": {"fitting": "Fitting (p,d,q)=({p},{d},{q}) [{i}/{total}]", "failed": "-> failed: {e}",
               "grid_done": "Grid search completed."},
        "zh": {"fitting": "拟合 (p,d,q)=({p},{d},{q}) [{i}/{total}]", "failed": "-> 拟合失败: {e}",
               "grid_done": "网格搜索完成。"},
    }
    if lang not in msgs:
        lang = "en"

    candidates = [(p, d, q) for p in range(p_max + 1) for d in range(d_max + 1) for q in range(q_max + 1)]
    total = len(candidates)

    def _fit_candidate(args):
        p, d, q, idx = args
        if verbose:
            print(msgs[lang]["fitting"].format(p=p, d=d, q=q, i=idx + 1, total=total))
        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(endog=y, exog=exog, order=(p, d, q), enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit(disp=disp, method=method, maxiter=maxiter)
            fit_time = time.time() - start
            aic = float(getattr(res, "aic", np.nan))
            bic = float(getattr(res, "bic", np.nan))
            llf = float(getattr(res, "llf", np.nan))
            converged = bool(getattr(res, "converged", True))
        except Exception as e:
            fit_time = time.time() - start
            aic = np.nan
            bic = np.nan
            llf = np.nan
            converged = False
            if verbose:
                print(msgs[lang]["failed"].format(e=e))

        return {
            "p": p,
            "d": d,
            "q": q,
            "aic": aic,
            "bic": bic,
            "llf": llf,
            "converged": converged,
            "fit_time_seconds": fit_time,
        }

    records = []
    if n_jobs is None:
        n_jobs = 1

    # Parallel execution if requested and joblib is available
    if n_jobs != 1 and Parallel is not None:
        # prepare arguments with index for progress messages
        args_list = [(p, d, q, idx) for idx, (p, d, q) in enumerate(candidates)]
        results = Parallel(n_jobs=n_jobs)(delayed(_fit_candidate)(args) for args in args_list)
        records = results
    else:
        # fall back to serial
        for idx, (p, d, q) in enumerate(candidates):
            rec = _fit_candidate((p, d, q, idx))
            records.append(rec)

    if verbose:
        print(msgs[lang]["grid_done"])

    df = pd.DataFrame.from_records(records)
    if criterion not in {"aic", "bic", "llf"}:
        raise ValueError("criterion must be one of 'aic','bic','llf'")

    # for llf higher is better -> sort descending; for aic/bic lower is better
    ascending = False if criterion == "llf" else True
    df_sorted = df.sort_values(by=criterion, ascending=ascending).reset_index(drop=True)
    return df_sorted


def select_best_model(y, exog: Optional[np.ndarray] = None, p_max: int = 3, d_max: int = 2, q_max: int = 3,
                      criterion: str = "aic", method: str = "lbfgs", maxiter: int = 50,
                      disp: bool = False, verbose: bool = False, n_jobs: int = 1, lang: str = "en"):
    """Run `select_order` and return the best fitted SARIMAXResults along with the results table.

    Returns (best_result, results_table). If the top candidate failed to fit, it will try the next one.
    """
    df = select_order(y, exog=exog, p_max=p_max, d_max=d_max, q_max=q_max,
                      criterion=criterion, method=method, maxiter=maxiter, disp=disp, verbose=verbose, n_jobs=n_jobs, lang=lang)

    # iterate candidates until one yields a fitted results object
    for idx, row in df.iterrows():
        if not row.get("converged", False):
            continue
        p, d, q = int(row["p"]), int(row["d"]), int(row["q"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(endog=y, exog=exog, order=(p, d, q), enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit(disp=disp, method=method, maxiter=maxiter)
            return res, df
        except Exception as e:
            if verbose:
                if lang == "zh":
                    print(f"最佳候选 (p,d,q)=({p},{d},{q}) 重新拟合失败，尝试下一个。错误：{e}")
                else:
                    print(f"Best candidate (p,d,q)=({p},{d},{q}) failed to refit; trying next. Error: {e}")
            continue

    # if none successful
    raise RuntimeError("No converged model found in grid search")


def auto_arima_select(y, exog: Optional[np.ndarray] = None,
                       seasonal: bool = False, m: int = 1,
                       start_p: int = 0, start_q: int = 0, max_p: int = 3, max_q: int = 3,
                       start_P: int = 0, start_Q: int = 0, max_P: int = 1, max_Q: int = 1,
                       d: Optional[int] = None, D: Optional[int] = None,
                       information_criterion: str = 'aic',
                       stepwise: bool = True, suppress_warnings: bool = True,
                       **auto_kwargs):
    """Use pmdarima.auto_arima to automatically select an ARIMA/SARIMAX model.

    This is a thin wrapper around `pmdarima.auto_arima`. It returns the fitted
    pmdarima `ARIMA` object (or raises if `pmdarima` is not installed).

    Parameters
    - y: 1-D array-like endogenous series
    - exog: optional 2-D array of exogenous regressors
    - seasonal: whether to fit a seasonal ARIMA (SARIMAX)
    - m: the number of periods in seasonal cycle (e.g., 12 for monthly data with yearly seasonality)
    - start_p, start_q, max_p, max_q: search range for non-seasonal AR/MA orders
    - start_P, start_Q, max_P, max_Q: search range for seasonal AR/MA orders
    - d, D: differencing orders (None lets auto_arima determine)
    - information_criterion: criterion to minimize ('aic', 'bic', etc.)
    - stepwise: whether to use stepwise selection (faster) or full grid search
    - suppress_warnings: forward to pmdarima to hide warnings
    - auto_kwargs: additional keyword args forwarded to `pmdarima.auto_arima`

    Returns
    - fitted pmdarima ARIMA object (has methods `summary()`, `predict(n_periods, exogenous=...)`, etc.)

    Raises
    - RuntimeError if `pmdarima` is not available in the environment.
    """
    if pm is None:
        raise RuntimeError("pmdarima is not installed. Install it (pip install pmdarima) to use auto_arima.")

    y = np.asarray(y)
    if exog is not None:
        exog = np.asarray(exog)

    model = pm.auto_arima(y=y, exogenous=exog,
                          start_p=start_p, start_q=start_q,
                          max_p=max_p, max_q=max_q,
                          start_P=start_P, start_Q=start_Q,
                          max_P=max_P, max_Q=max_Q,
                          d=d, D=D,
                          seasonal=seasonal, m=m,
                          information_criterion=information_criterion,
                          stepwise=stepwise,
                          suppress_warnings=suppress_warnings,
                          **auto_kwargs)
    return model
