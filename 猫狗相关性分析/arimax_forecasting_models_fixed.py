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
print(f"AIC: {cat_fit.aic:.2f}")
print(f"BIC: {cat_fit.bic:.2f}")
print(f"对数似然值: {cat_fit.llf:.2f}")
print(f"模型系数数量: {len(cat_fit.params)}")

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
print(f"AIC: {dog_fit.aic:.2f}")
print(f"BIC: {dog_fit.bic:.2f}")
print(f"对数似然值: {dog_fit.llf:.2f}")
print(f"模型系数数量: {len(dog_fit.params)}")

# 3. 为预测期创建外生变量（基于趋势外推）
forecast_years = [2024, 2025, 2026]

# 基于历史趋势外推外生变量
pet_penetration_forecast = [25.7, 26.4, 27.1]  # 年均增长0.7
gdp_forecast = [86778, 90249, 93859]  # 年均增长4%
income_forecast = [41179, 43238, 45400]  # 年均增长5%
urbanization_forecast = [66.02, 66.82, 67.62]  # 年均增长0.8

# 创建预测期外生变量矩阵
cat_exog_forecast = np.array([
    [pet_penetration_forecast[i], gdp_forecast[i], income_forecast[i]] 
    for i in range(3)
])

dog_exog_forecast = np.array([
    [pet_penetration_forecast[i], urbanization_forecast[i]] 
    for i in range(3)
])

# 4. 进行预测
print("\n\n3. 未来三年预测结果")
print("-" * 30)

try:
    # 猫数量预测
    cat_forecast = cat_fit.forecast(steps=3, exog=cat_exog_forecast)
    print(f"\n猫数量预测结果 (2024-2026):")
    for i, year in enumerate(forecast_years):
        print(f"{year}年: {cat_forecast.iloc[i]:.0f}万只")
        
except Exception as e:
    print(f"猫数量预测出错: {e}")
    # 使用简单趋势外推作为备选
    cat_trend = np.polyfit(range(7), cat_data.values, 1)
    cat_forecast = np.polyval(cat_trend, range(7, 10))
    print(f"\n猫数量趋势外推预测 (2024-2026):")
    for i, year in enumerate(forecast_years):
        print(f"{year}年: {cat_forecast[i]:.0f}万只")

try:
    # 狗数量预测  
    dog_forecast = dog_fit.forecast(steps=3, exog=dog_exog_forecast)
    print(f"\n狗数量预测结果 (2024-2026):")
    for i, year in enumerate(forecast_years):
        print(f"{year}年: {dog_forecast.iloc[i]:.0f}万只")
        
except Exception as e:
    print(f"狗数量预测出错: {e}")
    # 使用简单趋势外推作为备选
    dog_trend = np.polyfit(range(7), dog_data.values, 1)
    dog_forecast = np.polyval(dog_trend, range(7, 10))
    print(f"\n狗数量趋势外推预测 (2024-2026):")
    for i, year in enumerate(forecast_years):
        print(f"{year}年: {dog_forecast[i]:.0f}万只")

# 5. 模型诊断
print("\n\n4. 模型诊断分析")
print("-" * 25)

cat_residuals = cat_fit.resid
dog_residuals = dog_fit.resid

print("猫模型残差统计:")
print(f"均值: {cat_residuals.mean():.4f}")
print(f"标准差: {cat_residuals.std():.4f}")
print(f"Ljung-Box检验p值: {cat_fit.test_serial_correlation(method='ljungbox')[1][0]:.4f}")

print("\n狗模型残差统计:")
print(f"均值: {dog_residuals.mean():.4f}")
print(f"标准差: {dog_residuals.std():.4f}")
print(f"Ljung-Box检验p值: {dog_fit.test_serial_correlation(method='ljungbox')[1][0]:.4f}")

# 保存预测结果
forecast_results = pd.DataFrame({
    'Year': forecast_years,
    'Cat_Population_Forecast': cat_forecast.values if hasattr(cat_forecast, 'values') else cat_forecast,
    'Dog_Population_Forecast': dog_forecast.values if hasattr(dog_forecast, 'values') else dog_forecast,
    'Cat_AIC': [cat_fit.aic] * 3,
    'Dog_AIC': [dog_fit.aic] * 3,
    'Cat_BIC': [cat_fit.bic] * 3,
    'Dog_BIC': [dog_fit.bic] * 3
})

forecast_results.to_csv('arimax_forecast_results.csv', index=False, encoding='utf-8-sig')
print(f"\n预测结果已保存至: arimax_forecast_results.csv")

print("\n=== ARIMAX预测模型构建完成 ===")
print("=" * 50)