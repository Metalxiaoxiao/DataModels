# ARIMAX 包

本包基于 `statsmodels` 的 `SARIMAX` 封装，提供用于数学建模的 ARIMAX 功能，包括拟合、预测（含置信区间）、诊断以及模型选择工具。

目录（主要文件）
- `arimax.py`：`ARIMAX` 类（封装 `statsmodels`），提供 `fit`, `predict`, `summary`, `plot_diagnostics`。
- `model_selection.py`：模型选择工具（`select_order`, `select_best_model`），用于网格搜索 (p,d,q)。
- `plotting.py`：绘图工具，支持置信区间显示。
- `example.py`：演示拟合与预测的示例。
- `model_selection_example.py`：模型选择的示例（本文件）。

安装

建议在虚拟环境中安装依赖：

```powershell
pip install -r c:\dev\TimeManager\DataModels\arimax_package\requirements.txt
```

快速开始

1) 直接拟合 ARIMAX：

```python
from arimax_package import ARIMAX

model = ARIMAX(p=1, d=0, q=1, trend='c')
model.fit(y, exog=exog, disp=False)
print(model.summary())
mean_fc, ci = model.predict(steps=20, exog=exog_future, return_conf_int=True)
```

2) 使用模型选择（网格搜索）选择(p,d,q)：

```python
from arimax_package import select_order, select_best_model

table = select_order(y, exog=exog, p_max=3, d_max=1, q_max=3, criterion='aic')
print(table.head())

best_res, table = select_best_model(y, exog=exog, p_max=3, d_max=1, q_max=3)
print(best_res.summary())
```

示例

运行模型选择示例（会输出候选表、最佳模型摘要并打开诊断与预测图）：

```powershell
python -m arimax_package.model_selection_example
```

注意事项与建议
- `select_order` 会在所给的 p/d/q 范围内暴力搜索，若范围较大拟合耗时会显著增加；可考虑并行化或使用 `pmdarima.auto_arima`（需额外安装 `pmdarima`）。
- `select_best_model` 会尝试按排序好的候选重试拟合直到找到可收敛的模型。
- 使用 `fit` 的关键字参数（如 `method`, `maxiter`, `disp`）来控制拟合算法。
- 对于正式建模，建议结合 AIC/BIC、残差诊断和领域知识来选择模型，而非仅靠单一指标。

如果你希望我：
- 把中文文档合并回 `README.md`（双语）；
- 为 `select_order` 添加并行化实现；
- 或在例子中加入自动化模型选择（与 `pmdarima` 对比），请告诉我需要哪一项，我会继续实现。

新增：`pmdarima.auto_arima` 支持
--------------------------------

本仓库已添加 `pmdarima` 的支持，你可以使用 `auto_arima_select` 来做自动化模型选择。下面是对主要函数/类及其参数的详细说明，便于数学建模时直接使用。

API 详细说明
----------------

1. `ARIMAX(p=1, d=0, q=0, trend='c', enforce_stationarity=True, enforce_invertibility=True)`
- 说明：封装 `statsmodels` 的 `SARIMAX`，用于拟合 ARIMAX 模型。
- 参数：
	- `p` (int): 自回归阶数 (AR)。
	- `d` (int): 差分阶数 (I)。
	- `q` (int): 移动平均阶数 (MA)。
	- `trend` (str or None): 趋势项，'c' 表示包含常数截距，None 表示无趋势。
	- `enforce_stationarity` (bool): 拟合时是否强制平稳约束。
	- `enforce_invertibility` (bool): 拟合时是否强制可逆约束。
- 重要方法：
	- `fit(y, exog=None, **fit_kwargs)`：拟合模型。`y` 为一维数组；若有外生变量 `exog`，其形状为 `(n_obs, k_exog)`。`fit_kwargs` 转发给 `SARIMAX.fit`（例如 `disp=False`, `method='lbfgs'`, `maxiter=50`）。
	- `summary()`：返回参数估计、标准误、信息准则的文本/对象表示。
	- `predict(start=None, end=None, steps=None, exog=None, return_conf_int=False, alpha=0.05)`：预测或预测并返回置信区间。`steps` 指定外样本预测步数；若模型有外生变量则需提供 `exog`（未来期的 exog 矩阵）。
	- `plot_diagnostics(figsize=(10,8))`：显示残差诊断图（残差自相关、QQ 图、残差与滞后等）。

2. `select_order(y, exog=None, p_max=3, d_max=2, q_max=3, criterion='aic', method='lbfgs', maxiter=50, disp=False, verbose=False)`
- 说明：在给定范围内对 (p,d,q) 做暴力网格搜索，并返回按信息准则排序的结果表。
- 参数：
	- `y` (array-like): 待拟合的内生序列。
	- `exog` (array-like 或 None): 外生变量矩阵，行数需与 `y` 对齐。
	- `p_max`, `d_max`, `q_max` (int): 要搜索的最大阶数（包含 0）。
	- `criterion` (str): 用于排序的准则，支持 `'aic'`, `'bic'`, `'llf'`。
	- `method`, `maxiter`, `disp`：传给 `SARIMAX.fit` 的优化参数（例如 `method='lbfgs'`）。
	- `verbose` (bool)：若 True，会打印进度与失败信息。
- 返回：`pandas.DataFrame` 包含列 `['p','d','q','aic','bic','llf','converged','fit_time_seconds']`，按所选准则排序。

3. `select_best_model(y, exog=None, p_max=3, d_max=2, q_max=3, criterion='aic', method='lbfgs', maxiter=50, disp=False, verbose=False)`
- 说明：先调用 `select_order`，然后尝试按排序顺序重拟合候选模型，直到找到能收敛的模型并返回其 `SARIMAXResults` 对象。
- 返回：`(best_result, results_table)`，其中 `best_result` 为已拟合的 `statsmodels` 结果对象，`results_table` 为排序表。

4. `auto_arima_select(y, exog=None, seasonal=False, m=1, start_p=0, start_q=0, max_p=3, max_q=3, start_P=0, start_Q=0, max_P=1, max_Q=1, d=None, D=None, information_criterion='aic', stepwise=True, suppress_warnings=True, **auto_kwargs)`
- 说明：使用 `pmdarima.auto_arima` 自动化搜索 ARIMA/SARIMA 模型及差分阶数。
- 参数（主要项解释）：
	- `y`: 一维内生序列。
	- `exog`: 外生变量矩阵（若有）。
	- `seasonal` (bool): 是否搜索季节性模型（SARIMA）。
	- `m` (int): 季节周期长度（例如月度数据 `m=12`）。
	- `start_p/start_q/max_p/max_q`: 非季节性 AR/MA 的搜索范围。
	- `start_P/start_Q/max_P/max_Q`: 季节性 AR/MA 的搜索范围（仅当 `seasonal=True` 有效）。
	- `d/D`: 可选的差分阶数；None 表示让 `auto_arima` 自动判别。
	- `information_criterion`: 用于选择模型的准则，如 `'aic'` 或 `'bic'`。
	- `stepwise` (bool): 是否使用逐步搜索（速度更快）或完整网格搜索。
	- `suppress_warnings` (bool): 是否屏蔽 pmdarima 中的警告输出。
	- `auto_kwargs`: 额外参数转发给 `pmdarima.auto_arima`，例如 `seasonal_test='ocsb'`, `approx=False` 等。
- 返回：已拟合的 `pmdarima` 模型对象（支持 `summary()` 与 `predict(n_periods, exogenous=...)`）。

5. `plot_series_and_forecast(y, forecast, conf_int=None, title='Series and Forecast', xlabel='t')`
- 说明：使用 seaborn 绘制观测序列与预测序列，并可绘制置信区间。
- 参数：
	- `y`：观测序列（1 维）。
	- `forecast`：预测序列（1 维）。
	- `conf_int`：置信区间矩阵 `(m,2)` 或 pandas DataFrame。
	- `title`, `xlabel`：图表标题与 x 轴标签。
	- 返回 `(fig, ax)` 以便进一步自定义或保存。

示例：使用 `pmdarima` 与 `select_order` 的组合
--------------------------------------

你可以先用 `select_order` 进行一个受控的网格搜索，再使用 `auto_arima_select` 做更自动化的探索；下列是一个工作流程示例：

```python
from arimax_package import select_order, select_best_model, auto_arima_select

# 1) 简单网格搜索（快速筛选候选）
table = select_order(y, exog=exog, p_max=3, d_max=1, q_max=3)
print(table.head())

# 2) 使用 pmdarima 的 auto_arima 进行更全面的搜索（自动选择 d/D）
auto_est = auto_arima_select(y, exog=exog, seasonal=False, start_p=0, start_q=0, max_p=5, max_q=5, stepwise=True)
print(auto_est.summary())

# 3) 对比并最终选择模型
