import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 历史数据
historical_data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Cat_Population': [6300, 6800, 7500, 8200, 8900, 9500, 10200],
    'Dog_Population': [13500, 14200, 14800, 15200, 15500, 15800, 16000]
}

# 预测数据（从之前的ARIMAX模型获得）
forecast_data = {
    'Year': [2024, 2025, 2026],
    'Cat_Population_Forecast': [10165, 11665, 10880],
    'Dog_Population_Forecast': [16094, 16262, 16430]
}

# 创建完整的时间序列数据
all_years = historical_data['Year'] + forecast_data['Year']
cat_population = historical_data['Cat_Population'] + forecast_data['Cat_Population_Forecast']
dog_population = historical_data['Dog_Population'] + forecast_data['Dog_Population_Forecast']

# 创建数据框
df = pd.DataFrame({
    'Year': all_years,
    'Cat_Population': cat_population,
    'Dog_Population': dog_population
})

# 标记历史数据和预测数据
is_historical = [True] * len(historical_data['Year']) + [False] * len(forecast_data['Year'])
df['Is_Historical'] = is_historical

print("=== 创建ARIMAX预测可视化图表 ===")
print("=" * 50)

# 1. 创建综合预测图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('宠物数量ARIMAX模型预测分析 (2017-2026)', fontsize=16, fontweight='bold')

# 图表1：猫数量预测
ax1.plot(df[df['Is_Historical']]['Year'], df[df['Is_Historical']]['Cat_Population'], 
         'o-', color='blue', linewidth=2, markersize=6, label='历史数据')
ax1.plot(df[~df['Is_Historical']]['Year'], df[~df['Is_Historical']]['Cat_Population'], 
         's--', color='red', linewidth=2, markersize=6, label='ARIMAX预测')
ax1.axvline(x=2023.5, color='gray', linestyle=':', alpha=0.7, label='预测起点')
ax1.set_title('猫数量预测', fontsize=14, fontweight='bold')
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('数量（万只）', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图表2：狗数量预测
ax2.plot(df[df['Is_Historical']]['Year'], df[df['Is_Historical']]['Dog_Population'], 
         'o-', color='green', linewidth=2, markersize=6, label='历史数据')
ax2.plot(df[~df['Is_Historical']]['Year'], df[~df['Is_Historical']]['Dog_Population'], 
         's--', color='red', linewidth=2, markersize=6, label='ARIMAX预测')
ax2.axvline(x=2023.5, color='gray', linestyle=':', alpha=0.7, label='预测起点')
ax2.set_title('狗数量预测', fontsize=14, fontweight='bold')
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('数量（万只）', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图表3：猫狗数量对比
ax3.plot(df['Year'], df['Cat_Population'], 'o-', color='blue', linewidth=2, 
         markersize=5, label='猫数量')
ax3.plot(df['Year'], df['Dog_Population'], 'o-', color='green', linewidth=2, 
         markersize=5, label='狗数量')
ax3.axvline(x=2023.5, color='gray', linestyle=':', alpha=0.7)
ax3.set_title('猫狗数量对比预测', fontsize=14, fontweight='bold')
ax3.set_xlabel('年份', fontsize=12)
ax3.set_ylabel('数量（万只）', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图表4：增长率分析
cat_growth_rates = [0] + [(df['Cat_Population'].iloc[i] - df['Cat_Population'].iloc[i-1]) / 
                          df['Cat_Population'].iloc[i-1] * 100 for i in range(1, len(df))]
dog_growth_rates = [0] + [(df['Dog_Population'].iloc[i] - df['Dog_Population'].iloc[i-1]) / 
                          df['Dog_Population'].iloc[i-1] * 100 for i in range(1, len(df))]

# 只显示预测期的增长率
forecast_period = df[~df['Is_Historical']]
cat_forecast_growth = cat_growth_rates[-3:]
dog_forecast_growth = dog_growth_rates[-3:]

ax4.bar(forecast_period['Year'], cat_forecast_growth, alpha=0.7, color='blue', 
        label='猫增长率', width=0.35)
ax4.bar(forecast_period['Year'] + 0.35, dog_forecast_growth, alpha=0.7, color='green', 
        label='狗增长率', width=0.35)
ax4.set_title('预测期增长率对比 (2024-2026)', fontsize=14, fontweight='bold')
ax4.set_xlabel('年份', fontsize=12)
ax4.set_ylabel('增长率 (%)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.savefig('arimax_forecast_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 综合预测图表已保存: arimax_forecast_comprehensive.png")

# 2. 创建模型评估图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ARIMAX模型评估与诊断', fontsize=16, fontweight='bold')

# 模型性能指标
models = ['猫ARIMAX(2,1,1)', '狗ARIMAX(0,1,1)']
aic_values = [106.04, 82.74]
bic_values = [104.58, 81.90]
log_likelihood = [-46.02, -37.37]

# 图表1：AIC对比
ax1.bar(models, aic_values, color=['blue', 'green'], alpha=0.7)
ax1.set_title('AIC信息准则对比', fontsize=14, fontweight='bold')
ax1.set_ylabel('AIC值', fontsize=12)
ax1.grid(True, alpha=0.3)
for i, v in enumerate(aic_values):
    ax1.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# 图表2：BIC对比
ax2.bar(models, bic_values, color=['blue', 'green'], alpha=0.7)
ax2.set_title('BIC信息准则对比', fontsize=14, fontweight='bold')
ax2.set_ylabel('BIC值', fontsize=12)
ax2.grid(True, alpha=0.3)
for i, v in enumerate(bic_values):
    ax2.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# 图表3：对数似然值对比
ax3.bar(models, log_likelihood, color=['blue', 'green'], alpha=0.7)
ax3.set_title('对数似然值对比', fontsize=14, fontweight='bold')
ax3.set_ylabel('对数似然值', fontsize=12)
ax3.grid(True, alpha=0.3)
for i, v in enumerate(log_likelihood):
    ax3.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# 图表4：预测准确性评估（模拟残差分析）
# 基于之前的残差分析结果
cat_residual_mean, cat_residual_std = -112.75, 160.35
dog_residual_mean, dog_residual_std = 1162.28, 2995.00

residual_stats = ['均值', '标准差']
cat_residuals = [cat_residual_mean, cat_residual_std]
dog_residuals = [dog_residual_mean, dog_residual_std]

x = np.arange(len(residual_stats))
width = 0.35

ax4.bar(x - width/2, cat_residuals, width, label='猫模型残差', color='blue', alpha=0.7)
ax4.bar(x + width/2, dog_residuals, width, label='狗模型残差', color='green', alpha=0.7)
ax4.set_title('残差统计对比', fontsize=14, fontweight='bold')
ax4.set_ylabel('残差值', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(residual_stats)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arimax_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 模型评估图表已保存: arimax_model_evaluation.png")

# 3. 创建预测趋势分析图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('ARIMAX预测趋势深度分析', fontsize=16, fontweight='bold')

# 预测数据
cat_forecast = [10165, 11665, 10880]
dog_forecast = [16094, 16262, 16430]
years = [2024, 2025, 2026]

# 图表1：预测值与历史趋势对比
historical_cat = [6300, 6800, 7500, 8200, 8900, 9500, 10200]
historical_dog = [13500, 14200, 14800, 15200, 15500, 15800, 16000]
hist_years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

# 计算历史平均增长率
hist_cat_growth = np.mean([(historical_cat[i+1] - historical_cat[i]) / historical_cat[i] * 100 
                          for i in range(len(historical_cat)-1)])
hist_dog_growth = np.mean([(historical_dog[i+1] - historical_dog[i]) / historical_dog[i] * 100 
                          for i in range(len(historical_dog)-1)])

# 预测增长率
cat_forecast_growth = [(cat_forecast[i+1] - cat_forecast[i]) / cat_forecast[i] * 100 
                      for i in range(len(cat_forecast)-1)]
dog_forecast_growth = [(dog_forecast[i+1] - dog_forecast[i]) / dog_forecast[i] * 100 
                      for i in range(len(dog_forecast)-1)]

ax1.plot(hist_years[-3:], historical_cat[-3:], 'o-', color='blue', linewidth=2, 
         markersize=6, label='猫历史趋势')
ax1.plot(years, cat_forecast, 's--', color='red', linewidth=2, 
         markersize=6, label='猫ARIMAX预测')
ax1.plot(hist_years[-3:], historical_dog[-3:], 'o-', color='green', linewidth=2, 
         markersize=6, label='狗历史趋势')
ax1.plot(years, dog_forecast, 's--', color='orange', linewidth=2, 
         markersize=6, label='狗ARIMAX预测')
ax1.axvline(x=2023.5, color='gray', linestyle=':', alpha=0.7)
ax1.set_title('历史趋势与预测对比', fontsize=14, fontweight='bold')
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('数量（万只）', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图表2：预测不确定性分析
# 基于模型残差计算预测区间
cat_std = 160.35  # 猫模型残差标准差
dog_std = 2995.00  # 狗模型残差标准差
confidence_factor = 1.96  # 95%置信区间

cat_upper = [x + confidence_factor * cat_std for x in cat_forecast]
cat_lower = [x - confidence_factor * cat_std for x in cat_forecast]
dog_upper = [x + confidence_factor * dog_std for x in dog_forecast]
dog_lower = [x - confidence_factor * dog_std for x in dog_forecast]

ax2.fill_between(years, cat_lower, cat_upper, alpha=0.3, color='blue', label='猫95%置信区间')
ax2.fill_between(years, dog_lower, dog_upper, alpha=0.3, color='green', label='狗95%置信区间')
ax2.plot(years, cat_forecast, 'o-', color='blue', linewidth=2, markersize=6, label='猫预测值')
ax2.plot(years, dog_forecast, 'o-', color='green', linewidth=2, markersize=6, label='狗预测值')
ax2.set_title('预测不确定性分析（95%置信区间）', fontsize=14, fontweight='bold')
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('数量（万只）', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arimax_forecast_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 趋势分析图表已保存: arimax_forecast_trend_analysis.png")

print("\n=== 可视化图表创建完成 ===")
print("=" * 50)
print("已生成以下图表文件:")
print("1. arimax_forecast_comprehensive.png - 综合预测图表")
print("2. arimax_model_evaluation.png - 模型评估图表") 
print("3. arimax_forecast_trend_analysis.png - 趋势分析图表")