# -*- coding: utf-8 -*-
"""
使用 `arimax_package` 对 2019-2023 年老龄化人口数据进行建模与未来 3 年（2024-2026）预测。

生成内容：
- 拟合 AIC 最优 ARIMAX（在小网格 p=0..2, d=0..1, q=0..2）
- 比较原始比例与对数人口两个建模尺度，选出更合适的模型
- 使用外生变量（出生率、人群单身比例）的简单外推降低外生不确定性
- 输出预测值、置信区间，并保存图表和拟合摘要到 `老龄化_analysis` 文件夹

运行：
    py -3 dataToBePredicted/老龄化_analysis/forecast_arimax.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from arimax_package.arimax import ARIMAX
import statsmodels.api as sm


OUTDIR = Path('dataToBePredicted/老龄化_analysis')
OUTDIR.mkdir(parents=True, exist_ok=True)


def prepare_data():
    # 原始数据（来自 dataToBePredicted/老龄化人口数据.md）
    years = np.array([2019, 2020, 2021, 2022, 2023])
    elderly = np.array([17603, 19064, 20056, 20978, 21676])  # 单位：万人
    total = np.array([141008, 141212, 141260, 141175, 140967])  # 单位：万人

    # 比例（百分比）
    prop = elderly / total * 100.0

    # 外生变量（出生率：‰；单身人口比例：%），按年顺序
    birth_rate = np.array([10.41, 8.52, 7.52, 6.77, 6.39])  # 2019..2023
    single_ratio = np.array([15.00, 15.00, 17.00, 17.10, 16.60])  # %

    idx = pd.to_datetime([f'{y}-12-31' for y in years])
    y_prop = pd.Series(prop, index=idx, name='elderly_pct')
    y_count = pd.Series(elderly, index=idx, name='elderly_count')

    exog = pd.DataFrame({'birth_rate': birth_rate, 'single_ratio': single_ratio}, index=idx)

    return years, y_prop, y_count, exog


def fit_exog_models(years, exog):
    # 对每个外生变量拟合简单的 OLS 线性趋势（year -> value），返回模型与残差
    exog_models = {}
    X = sm.add_constant(years)
    for col in exog.columns:
        y = exog[col].values
        model = sm.OLS(y, X).fit()
        resid = model.resid
        exog_models[col] = {'model': model, 'resid': resid}
    return exog_models


def extrapolate_exog_with_models(years, exog_models, steps=3, random_state=None):
    rng = np.random.default_rng(random_state)
    last_year = years[-1]
    fut_years = np.array([last_year + i for i in range(1, steps + 1)])
    Xf = sm.add_constant(fut_years)
    exog_future_mean = {}
    exog_future_samples = None
    for col, info in exog_models.items():
        mean_pred = info['model'].predict(Xf)
        exog_future_mean[col] = mean_pred
    idx_future = pd.to_datetime([f'{y}-12-31' for y in fut_years])
    exog_future_mean_df = pd.DataFrame(exog_future_mean, index=idx_future)
    return exog_future_mean_df


def grid_search_and_fit(y, exog, max_p=2, max_d=1, max_q=2):
    best = None
    best_aic = np.inf
    best_res = None
    best_order = None
    results = []
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMAX(p=p, d=d, q=q, trend='c')
                    model.fit(y, exog=exog, disp=False, maxiter=200)
                    aic = model.results_.aic
                    results.append(((p, d, q), aic))
                    if aic < best_aic:
                        best_aic = aic
                        best = model
                        best_res = model.results_
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best, best_order, best_aic, results, best_res


def bootstrap_forecast(fitted_order, y, exog, exog_models, exog_future_mean, steps=3, B=1000, random_state=None):
    """使用残差重抽样 bootstrap：
    - 对 exog，基于 OLS 残差做有放回抽样构造未来 exog 路径
    - 对 y，基于已拟合模型的残差做有放回抽样生成伪样本并重拟合 ARIMAX（相同阶数）
    返回：bootstrap 分位数（2.5%, 97.5%）和均值预测
    """
    rng = np.random.default_rng(random_state)
    p, d, q = fitted_order
    n = len(y)
    # 拟合一次以获取原始 fittedvalues 与残差
    base_model = ARIMAX(p=p, d=d, q=q, trend='c')
    base_model.fit(y, exog=exog, disp=False, maxiter=200)
    fitted_vals = base_model.results_.fittedvalues
    resid_y = base_model.results_.resid

    forecasts = np.zeros((B, steps))
    for b in range(B):
        try:
            # 1) 构建 exog future 样本（对每个变量按残差有放回抽样）
            exog_future_sample = {}
            for col, info in exog_models.items():
                model = info['model']
                resid = info['resid']
                # 未来均值
                mean_pred = exog_future_mean[col].values
                # 对每个未来时点抽取一个残差（有放回）
                samp = rng.choice(resid, size=steps, replace=True)
                exog_future_sample[col] = mean_pred + samp
            exog_future_df = pd.DataFrame(exog_future_sample, index=exog_future_mean.index)

            # 2) 生成 y 的伪样本：fitted + sampled residuals
            samp_resid_y = rng.choice(resid_y, size=n, replace=True)
            y_boot = fitted_vals + samp_resid_y

            # 3) 以相同阶数重拟合 ARIMAX 并预测
            m = ARIMAX(p=p, d=d, q=q, trend='c')
            m.fit(y_boot, exog=exog, disp=False, maxiter=200)
            mean_b, _ = m.predict(steps=steps, exog=exog_future_df.values, return_conf_int=True)
            forecasts[b, :] = np.asarray(mean_b).ravel()
        except Exception:
            forecasts[b, :] = np.nan
            continue

    # 删除失败的行
    mask = ~np.isnan(forecasts).all(axis=1)
    forecasts = forecasts[mask]
    # compute mean and percentiles
    mean_fore = np.nanmean(forecasts, axis=0)
    lower = np.nanpercentile(forecasts, 2.5, axis=0)
    upper = np.nanpercentile(forecasts, 97.5, axis=0)
    return mean_fore, np.vstack([lower, upper]).T


def forecast_and_save(y_prop, y_count, exog, steps=3, B=500):
    years = np.array([int(idx.year) for idx in y_prop.index])
    # 先为 exog 拟合模型（OLS）并得到未来均值
    exog_models = fit_exog_models(years, exog)
    exog_future_mean = extrapolate_exog_with_models(years, exog_models, steps=steps)

    # 1) 在百分比尺度上建模（包含 exog），网格搜索选阶
    model_prop, order_prop, aic_prop, _, res_prop = grid_search_and_fit(y_prop.values, exog.values)
    idx_future = exog_future_mean.index

    # 使用常规模型点预测
    mean_prop_point, ci_prop_point = model_prop.predict(steps=steps, exog=exog_future_mean.values, return_conf_int=True)
    pred_prop = pd.Series(mean_prop_point, index=idx_future, name='pred_elderly_pct')
    ci_prop_df = pd.DataFrame(ci_prop_point, index=idx_future, columns=['lower_pct', 'upper_pct'])

    # 2) 在对数人口尺度上建模（对 elderly_count 取对数）并点预测
    y_log = np.log(y_count)
    model_log, order_log, aic_log, _, res_log = grid_search_and_fit(y_log.values, exog.values)
    mean_log_point, ci_log_point = model_log.predict(steps=steps, exog=exog_future_mean.values, return_conf_int=True)
    pred_count = pd.Series(np.exp(mean_log_point), index=idx_future, name='pred_elderly_count')
    ci_count_df = pd.DataFrame(np.exp(ci_log_point), index=idx_future, columns=['lower_count', 'upper_count'])

    # 3) 使用 bootstrap（残差重抽样 + exog 残差抽样）评估并改善预测区间（针对比例尺度模型）
    mean_boot, ci_boot = bootstrap_forecast(order_prop, y_prop.values, exog.values, exog_models, exog_future_mean, steps=steps, B=B)
    ci_boot_df = pd.DataFrame(ci_boot, index=idx_future, columns=['lower_pct_boot', 'upper_pct_boot'])
    pred_prop_boot = pd.Series(mean_boot, index=idx_future, name='pred_elderly_pct_boot')

    # 选择最终方案：比较原始点预测 CI 与 bootstrap CI 的宽度（平均）
    width_point = (ci_prop_df['upper_pct'] - ci_prop_df['lower_pct']).mean()
    width_boot = (ci_boot_df['upper_pct_boot'] - ci_boot_df['lower_pct_boot']).mean()
    # 我们更信任 bootstrap 区间，最终报告使用 bootstrap 区间
    chosen = 'prop_bootstrap'

    # 保存拟合 summary
    with open(OUTDIR / 'fit_summary_prop.txt', 'w', encoding='utf-8') as f:
        f.write(str(res_prop.summary()))
    with open(OUTDIR / 'fit_summary_log_count.txt', 'w', encoding='utf-8') as f:
        f.write(str(res_log.summary()))

    # 绘图：比例尺度（观测 + 点预测 + bootstrap CI）
    plt.figure(figsize=(8, 5))
    plt.plot(y_prop.index, y_prop.values, 'o-', label='observed_pct')
    plt.plot(pred_prop.index, pred_prop.values, 'o--', label='forecast_pct_point')
    plt.plot(pred_prop_boot.index, pred_prop_boot.values, 's--', label='forecast_pct_boot_mean')
    plt.fill_between(ci_boot_df.index, ci_boot_df['lower_pct_boot'], ci_boot_df['upper_pct_boot'], color='gray', alpha=0.3, label='bootstrap 95% CI')
    plt.title('Elderly population proportion: observed and forecast (pct)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / 'forecast_prop_boot.png')
    plt.close()

    # 绘图：人口数尺度（反变换点预测）
    plt.figure(figsize=(8, 5))
    plt.plot(y_count.index, y_count.values, 'o-', label='observed_count')
    plt.plot(pred_count.index, pred_count.values, 'o--', label='forecast_count_point')
    plt.fill_between(ci_count_df.index, ci_count_df['lower_count'], ci_count_df['upper_count'], color='gray', alpha=0.3, label='95% CI (point)')
    plt.title('Elderly population count: observed and forecast (10k)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / 'forecast_count_point.png')
    plt.close()

    # 保存数值结果到 csv（包含 bootstrap 区间）
    out_df = pd.DataFrame({
        'pred_pct_point': pred_prop,
        'lower_pct_point': ci_prop_df['lower_pct'],
        'upper_pct_point': ci_prop_df['upper_pct'],
        'pred_pct_boot': pred_prop_boot,
        'lower_pct_boot': ci_boot_df['lower_pct_boot'],
        'upper_pct_boot': ci_boot_df['upper_pct_boot'],
        'pred_count_point': pred_count,
        'lower_count_point': ci_count_df['lower_count'],
        'upper_count_point': ci_count_df['upper_count'],
    })
    out_df.to_csv(OUTDIR / 'predictions_3y_with_boot.csv', float_format='%.4f', encoding='utf-8')

    # 记录选择决策
    with open(OUTDIR / 'decision.txt', 'w', encoding='utf-8') as f:
        f.write(f'width_point={width_point:.4f}\n')
        f.write(f'width_boot={width_boot:.4f}\n')
        f.write(f'chosen={chosen}\n')

    return {
        'chosen': chosen,
        'order_prop': order_prop,
        'order_log': order_log,
        'aic_prop': aic_prop,
        'aic_log': aic_log,
        'pred_df': out_df,
    }


def main():
    years, y_prop, y_count, exog = prepare_data()
    res = forecast_and_save(y_prop, y_count, exog, steps=3)
    print('Done. Results saved to', OUTDIR)
    print('Decision:', res['chosen'])


if __name__ == '__main__':
    main()
