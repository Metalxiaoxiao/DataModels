import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_heatmap():
    """
    创建Pearson和Spearman对比热力图
    """
    # 宠物数量数据 (2019-2023)
    pet_data = {
        '年份': [2019, 2020, 2021, 2022, 2023],
        '猫数量': [4412, 4862, 5806, 6536, 6980],
        '狗数量': [5503, 5222, 5429, 5119, 5175]
    }
    
    # 自变量数据
    independent_vars = {
        '年份': [2019, 2020, 2021, 2022, 2023],
        '人均国内生产总值(元)': [71453, 73338, 83111, 87385, 91746],
        '人均可支配收入(元)': [30733, 32189, 35128, 36883, 39218],
        '65岁以上老龄化率(%)': [12.60, 13.50, 14.20, 14.90, 15.40],
        '单身人口比例(%)': [15.00, 15.00, 17.00, 17.10, 16.60],
        '城镇人口比例(%)': [60.60, 63.89, 64.72, 65.22, 66.16]
    }
    
    # 创建DataFrame
    df_pets = pd.DataFrame(pet_data)
    df_vars = pd.DataFrame(independent_vars)
    df = pd.merge(df_vars, df_pets, on='年份', how='inner')
    
    # 定义变量
    independent_variables = ['人均国内生产总值(元)', '人均可支配收入(元)', '65岁以上老龄化率(%)', 
                           '单身人口比例(%)', '城镇人口比例(%)']
    dependent_variables = ['猫数量', '狗数量']
    
    # 计算Pearson相关系数
    pearson_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    for var in independent_variables:
        for target in dependent_variables:
            corr, _ = stats.pearsonr(df[var], df[target])
            pearson_corr.loc[var, target] = corr
    
    pearson_corr = pearson_corr.astype(float)
    
    # 计算Spearman相关系数
    spearman_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    for var in independent_variables:
        for target in dependent_variables:
            corr, _ = stats.spearmanr(df[var], df[target])
            spearman_corr.loc[var, target] = corr
    
    spearman_corr = spearman_corr.astype(float)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pearson热力图
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Pearson相关系数'}, 
                square=True, linewidths=0.5, ax=ax1)
    ax1.set_title('Pearson相关系数热力图', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('宠物数量指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('自变量', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(['猫的数量', '狗的数量'], rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Spearman热力图
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Spearman相关系数'}, 
                square=True, linewidths=0.5, ax=ax2)
    ax2.set_title('Spearman秩相关系数热力图', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('宠物数量指标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('自变量', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['猫的数量', '狗的数量'], rotation=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # 添加总标题
    fig.suptitle('自变量与宠物数量相关性对比分析\n(2019-2023年时间序列数据)', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 添加颜色编码说明
    fig.text(0.02, 0.02, 
             '颜色编码规则：\n红色：强正相关 (0.7-1.0)\n橙色：中等正相关 (0.3-0.7)\n白色：弱相关 (-0.3-0.3)\n浅蓝：中等负相关 (-0.7--0.3)\n深蓝：强负相关 (-1.0--0.7)',
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig('pet_correlation_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建详细统计表
    print("=== 相关系数详细对比表 ===")
    print("\nPearson相关系数:")
    print("=" * 60)
    for var in pearson_corr.index:
        for target in pearson_corr.columns:
            corr = pearson_corr.loc[var, target]
            if abs(corr) >= 0.7:
                strength = "强"
            elif abs(corr) >= 0.3:
                strength = "中等"
            else:
                strength = "弱"
            direction = "正相关" if corr > 0 else "负相关"
            print(f"{var} ↔ {target}: {corr:.3f} ({strength}{direction})")
    
    print("\nSpearman秩相关系数:")
    print("=" * 60)
    for var in spearman_corr.index:
        for target in spearman_corr.columns:
            corr = spearman_corr.loc[var, target]
            if abs(corr) >= 0.7:
                strength = "强"
            elif abs(corr) >= 0.3:
                strength = "中等"
            else:
                strength = "弱"
            direction = "正相关" if corr > 0 else "负相关"
            print(f"{var} ↔ {target}: {corr:.3f} ({strength}{direction})")
    
    return pearson_corr, spearman_corr

if __name__ == "__main__":
    pearson_corr, spearman_corr = create_comparison_heatmap()