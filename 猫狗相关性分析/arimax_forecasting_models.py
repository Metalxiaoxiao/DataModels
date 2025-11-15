import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建历史数据
data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Cat_Population': [6300, 6800, 7500, 8200, 8900, 9500, 10200],
    'Dog_Population': [13500, 14200, 14800, 15200, 15500, 15800, 16000],
    'Pet_Penetration_Rate': [15.8, 17.2, 19.6, 22.1, 23.7, 24.3, 25.0],
    'Per_Capita_GDP': [59660, 64644, 70078, 71828, 75676, 79538, 83402],
    'Disposable_Income': [25974, 28228, 30733, 32189, 35128, 36883, 39218],
    'Urbanization_Rate': [58.52, 59.58, 60.60, 61.43, 62.51, 63.89, 65.22],
    'Aging_Rate': [11.4, 11.9, 12.6, 13.5, 14.2, 14.9, 15.4]
}

df = pd.DataFrame(data)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# 数据预处理：扩展数据到2026年用于预测
forecast_years = pd.date_range(start='2024', end='2026', freq='Y')
extended_df = df.copy()

# 为预测期创建外生变量（基于趋势外推）
for year in forecast_years:
    new_row = {}
    # 基于历史趋势外推
    new_row['Pet_Penetration_Rate'] = extended_df['Pet_Penetration_Rate'].iloc[-1] + 0.7
    new_row['Per_Capita_GDP'] = extended_df['Per_Capita_GDP'].iloc[-1] * 1.04
    new_row['Disposable_Income'] = extended_df['Disposable_Income'].iloc[-1] * 1.05
    new_row['Urbanization_Rate'] = extended_df['Urbanization_Rate'].iloc[-1] + 0.8
    new_row['Aging_Rate'] = extended_df['Aging_Rate'].iloc[-1] + 0.5
    
    # 创建临时DataFrame行
    temp_df = pd.DataFrame([new_row], index=[year])
    extended_df = pd.concat([extended_df, temp_df])

print("=== ARIMAX预测模型构建 ===")
print("=" * 50)

# 1. 猫数量ARIMAX模型
print("\n1. 猫数量ARIMAX(2,1,1)模型构建")
print("-" * 40)

# 准备数据
cat_data = df['Cat_Population']
cat_exog = df[['Pet_Penetration_Rate', 'Per_Capita_GDP', 'Disposable_Income']]

# 构建ARIMAX模型
cat_model = ARIMA(cat_data, exog=cat_exog, order=(2, 1, 1))
cat_fit = cat_model.fit()

print("猫数量ARIMAX模型参数:")
print(cat_fit.summary())

# 准备预测期的外生变量
cat_exog_forecast = extended_df.loc[forecast_years, ['Pet_Penetration_Rate', 'Per_Capita_GDP', 'Disposable_Income']]

# 进行预测
cat_forecast = cat_fit.forecast(steps=3, exog=cat_exog_forecast)
# 获取置信区间
cat_pred_results = cat_fit.get_forecast(steps=3, exog=cat_exog_forecast)
cat_conf_int = cat_pred_results.conf_int()

print(f"\n猫数量预测结果 (2024-2026):")
for i, year in enumerate(['2024', '2025', '2026']):
    print(f"{year}年: {cat_forecast.iloc[i]:.0f}万只 (95%置信区间: {cat_conf_int.iloc[i, 0]:.0f} - {cat_conf_int.iloc[i, 1]:.0f})")

# 2. 狗数量ARIMAX模型
print("\n\n2. 狗数量ARIMAX(0,1,1)模型构建")
print("-" * 40)

# 准备数据
dog_data = df['Dog_Population']
dog_exog = df[['Pet_Penetration_Rate', 'Urbanization_Rate']]

# 构建ARIMAX模型
dog_model = ARIMA(dog_data, exog=dog_exog, order=(0, 1, 1))
dog_fit = dog_model.fit()

print("狗数量ARIMAX模型参数:")
print(dog_fit.summary())

# 准备预测期的外生变量
dog_exog_forecast = extended_df.loc[forecast_years, ['Pet_Penetration_Rate', 'Urbanization_Rate']]

# 进行预测
dog_forecast = dog_fit.forecast(steps=3, exog=dog_exog_forecast)
# 获取置信区间
dog_pred_results = dog_fit.get_forecast(steps=3, exog=dog_exog_forecast)
dog_conf_int = dog_pred_results.conf_int()

print(f"\n狗数量预测结果 (2024-2026):")
for i, year in enumerate(['2024', '2025', '2026']):
    print(f"{year}年: {dog_forecast.iloc[i]:.0f}万只 (95%置信区间: {dog_conf_int.iloc[i, 0]:.0f} - {dog_conf_int.iloc[i, 1]:.0f})")

# 3. 模型评估指标
print("\n\n3. 模型评估指标")
print("-" * 30)

print("猫数量ARIMAX模型:")
print(f"AIC: {cat_fit.aic:.2f}")
print(f"BIC: {cat_fit.bic:.2f}")
print(f"对数似然值: {cat_fit.llf:.2f}")

print("\n狗数量ARIMAX模型:")
print(f"AIC: {dog_fit.aic:.2f}")
print(f"BIC: {dog_fit.bic:.2f}")
print(f"对数似然值: {dog_fit.llf:.2f}")

# 4. 残差分析
print("\n\n4. 残差诊断分析")
print("-" * 25)

cat_residuals = cat_fit.resid
dog_residuals = dog_fit.resid

# 残差统计
print("猫模型残差统计:")
print(f"均值: {cat_residuals.mean():.4f}")
print(f"标准差: {cat_residuals.std():.4f}")
print(f"偏度: {cat_residuals.skew():.4f}")
print(f"峰度: {cat_residuals.kurtosis():.4f}")

print("\n狗模型残差统计:")
print(f"均值: {dog_residuals.mean():.4f}")
print(f"标准差: {dog_residuals.std():.4f}")
print(f"偏度: {dog_residuals.skew():.4f}")
print(f"峰度: {dog_residuals.kurtosis():.4f}")

# 保存预测结果
forecast_results = pd.DataFrame({
    'Year': [2024, 2025, 2026],
    'Cat_Population_Forecast': cat_forecast.values,
    'Cat_Population_Lower': cat_conf_int.iloc[:, 0].values,
    'Cat_Population_Upper': cat_conf_int.iloc[:, 1].values,
    'Dog_Population_Forecast': dog_forecast.values,
    'Dog_Population_Lower': dog_conf_int.iloc[:, 0].values,
    'Dog_Population_Upper': dog_conf_int.iloc[:, 1].values
})

forecast_results.to_csv('arimax_forecast_results.csv', index=False, encoding='utf-8-sig')
print(f"\n预测结果已保存至: arimax_forecast_results.csv")

print("\n=== ARIMAX预测模型构建完成 ===")
print("=" * 50)