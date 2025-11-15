"""模型选择示例（中文注释）。

演示如何使用 `select_order` 和 `select_best_model` 在小网格上搜索 ARIMA(p,d,q)
并查看最佳拟合模型的摘要、诊断图与预测（含置信区间）。
"""
import numpy as np

from .example import simulate_arimax
from .model_selection import select_order, select_best_model
from .plotting import plot_series_and_forecast


def run_selection_example():
    # 模拟数据（与 example.py 中相同的生成器）
    ar = [0.6]
    ma = [0.2]
    ex_coef = [0.5]
    y, x = simulate_arimax(n=300, ar_coef=ar, ma_coef=ma, exog_coef=ex_coef, d=0)

    # 网格搜索 (p,d,q) = 0..3, 0..1, 0..3（示例范围，可按需扩大）
    print("Running select_order grid search (this may take a moment)...")
    # 并行搜索示例：使用 n_jobs>1 来并行拟合候选模型，lang 可设为 'zh' 或 'en' 控制输出语言
    table = select_order(y, exog=x, p_max=3, d_max=1, q_max=3, criterion='aic', verbose=True,
                         method='lbfgs', maxiter=50, n_jobs=4, lang='zh')
    print('\nTop candidates by AIC:')
    print(table.head(10))

    # 选出最佳模型并拟合返回 SARIMAXResults
    print('\nFitting best candidate until convergence...')
    best_res, results_table = select_best_model(y, exog=x, p_max=3, d_max=1, q_max=3, criterion='aic', verbose=True,
                                                 method='lbfgs', maxiter=50, n_jobs=4, lang='zh')

    print('\nBest model summary:')
    print(best_res.summary())

    # 诊断图
    print('\nOpening diagnostic plots...')
    best_res.plot_diagnostics(figsize=(10, 8))

    # 预测 30 步，给出未来 exog（示例中复用前 30 行作为 future exog）
    exog_future = x[:30] if x is not None else None
    forecast_obj = best_res.get_forecast(steps=30, exog=exog_future)
    mean_fc = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int()

    print('\nPlotting forecast with confidence interval...')
    fig, ax = plot_series_and_forecast(y, mean_fc, conf_int=ci.values if hasattr(ci, 'values') else ci, title='Model selection forecast')
    fig.show()


if __name__ == '__main__':
    run_selection_example()
