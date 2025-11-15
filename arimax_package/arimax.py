"""ARIMAX wrapper using statsmodels' SARIMAX for reliable modeling.

This module provides a thin, user-friendly wrapper `ARIMAX` around
`statsmodels.tsa.statespace.sarimax.SARIMAX` that makes fitting and forecasting
with exogenous variables straightforward for math modeling workflows.
"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAX:
    """封装 `statsmodels` 的 `SARIMAX`，提供简洁的建模接口与文档化参数。

    属性与方法
    - `ARIMAX(p, d, q, trend='c', enforce_stationarity=True, enforce_invertibility=True)`:
        构造函数参数：
        - `p` (int): 自回归阶数 (AR)。
        - `d` (int): 差分阶数 (I)。
        - `q` (int): 移动平均阶数 (MA)。
        - `trend` (str or None): 趋势项；'c' 表示包含常数项（截距），None 表示无趋势项。
        - `enforce_stationarity` (bool): 是否在拟合中强制平稳性约束。
        - `enforce_invertibility` (bool): 是否在拟合中强制可逆性约束。

    方法说明
    - `fit(y, exog=None, **fit_kwargs)`:
        拟合模型。
        - `y`: 一维内生序列，array-like。
        - `exog`: 可选的外生解释变量，形状应为 `(n_obs, k_exog)`。
        - `fit_kwargs`: 转发给 `SARIMAX.fit` 的关键字（例如 `disp=False`, `method='lbfgs'`, `maxiter=50`）。
        返回 `self`，并保存已拟合结果在 `self.results_`。

    - `summary()`:
        返回 `statsmodels` 的 summary 对象（含参数估计、标准误、信息准则等）。

    - `predict(start=None, end=None, steps=None, exog=None, return_conf_int=False, alpha=0.05)`:
        预测或预测区间。
        - `start`, `end`: 用于指定内样本或部分样本预测的索引（与 `statsmodels` 的接口一致）。
        - `steps` (int): 若提供则进行前向 `steps` 步的外样本预测。
        - `exog`: 若模型使用外生变量，预测外样本时需提供未来的 `exog` 矩阵，形状 `(steps, k_exog)`。
        - `return_conf_int` (bool): 是否返回置信区间；若为 True，返回 `(predicted, conf_int)`。
        - `alpha` (float): 置信区间显著性水平，默认 0.05 对应 95% CI。

    - `plot_diagnostics(figsize=(10, 8))`:
        绘制标准诊断图（残差自相关、正态Q-Q图、残差时序等），基于 `statsmodels` 的 `plot_diagnostics`。
    """

    def __init__(self, p=1, d=0, q=0, trend='c', enforce_stationarity=True, enforce_invertibility=True):
        self.order = (int(p), int(d), int(q))
        self.trend = trend
        self.enforce_stationarity = bool(enforce_stationarity)
        self.enforce_invertibility = bool(enforce_invertibility)
        self.model_ = None
        self.results_ = None

    def fit(self, y, exog=None, **fit_kwargs):
        """拟合 ARIMAX 模型并保存结果到 `self.results_`。

        参数:
        - `y` (array-like): 内生变量序列，长度应大于模型阶数和差分阶数之和。
        - `exog` (array-like 或 None): 外生变量矩阵，若提供其行数应与 `y` 对齐。
        - `fit_kwargs`: 传递给 `SARIMAX.fit`（如 `disp`, `maxiter`, `method` 等）。

        返回:
        - `self`（方便链式调用）。
        """
        y = np.asarray(y)
        if exog is not None:
            exog = np.asarray(exog)
        self.model_ = SARIMAX(endog=y, exog=exog, order=self.order, trend=self.trend,
                              enforce_stationarity=self.enforce_stationarity,
                              enforce_invertibility=self.enforce_invertibility)
        self.results_ = self.model_.fit(**fit_kwargs)
        return self

    def summary(self):
        """返回已拟合模型的 `statsmodels` summary（包含参数估计与信息准则）。"""
        if self.results_ is None:
            raise RuntimeError("Model not fitted")
        return self.results_.summary()

    def predict(self, start=None, end=None, steps=None, exog=None, return_conf_int=False, alpha=0.05):
        """预测：支持内样本预测与外样本预测（forecast）。

        详见 `ARIMAX` 类注释中的参数说明。
        """
        if self.results_ is None:
            raise RuntimeError("Model not fitted")

        if steps is not None:
            # out-of-sample forecast
            forecast_obj = self.results_.get_forecast(steps=steps, exog=exog)
            mean = forecast_obj.predicted_mean
            if return_conf_int:
                ci = forecast_obj.conf_int(alpha=alpha)
                return mean, ci
            return mean
        else:
            pred = self.results_.predict(start=start, end=end, exog=exog)
            if return_conf_int:
                # use get_prediction for in-sample conf intervals
                pr = self.results_.get_prediction(start=start, end=end, exog=exog)
                return pred, pr.conf_int(alpha=alpha)
            return pred

    def plot_diagnostics(self, figsize=(10, 8)):
        """绘制模型诊断图。

        返回 `matplotlib` 的 figure 对象（由 `statsmodels` 生成）。
        """
        if self.results_ is None:
            raise RuntimeError("Model not fitted")
        return self.results_.plot_diagnostics(figsize=figsize)

