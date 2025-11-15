import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_series_and_forecast(y, forecast, conf_int=None, title="Series and Forecast", xlabel="t"):
    """使用 seaborn 绘制观测序列与预测结果并可选择显示置信区间。

    参数:
    - `y` (array-like): 观测值序列，1 维。
    - `forecast` (array-like): 预测值序列，长度为预测步数 `m`。
    - `conf_int` (可选): 置信区间，形状应为 `(m, 2)` 或 pandas DataFrame，列为 [lower, upper]。
    - `title` (str): 图表标题。
    - `xlabel` (str): x 轴标签，默认 't'。

    返回:
    - `(fig, ax)` matplotlib Figure 与 Axes，便于进一步自定义或保存。
    """
    y = np.asarray(y)
    forecast = np.asarray(forecast)
    n = len(y)
    m = len(forecast)
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.lineplot(x=np.arange(n), y=y, label="observed", ax=ax)
    ax = sns.lineplot(x=np.arange(n, n + m), y=forecast, label="forecast", ax=ax)
    if conf_int is not None:
        ci = np.asarray(conf_int)
        ax.fill_between(np.arange(n, n + m), ci[:, 0], ci[:, 1], color="b", alpha=0.2)
    ax.axvline(n - 1, linestyle="--", color="gray", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax

