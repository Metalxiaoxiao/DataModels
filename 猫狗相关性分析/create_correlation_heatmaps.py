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

def create_heatmaps():
    """
    创建相关性热力图
    """
    # 宠物数量数据 (2019-2023，与自变量数据对齐)
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
    
    # 创建Pearson热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Pearson相关系数'}, 
                square=True, linewidths=0.5)
    plt.title('自变量与宠物数量的Pearson相关系数热力图\n(2019-2023年时间序列数据)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('宠物数量指标', fontsize=12, fontweight='bold')
    plt.ylabel('自变量', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # 添加颜色编码说明
    plt.figtext(0.02, 0.02, 
                '颜色编码规则：\n红色：强正相关 (0.7-1.0)\n橙色：中等正相关 (0.3-0.7)\n白色：弱相关 (-0.3-0.3)\n浅蓝：中等负相关 (-0.7--0.3)\n深蓝：强负相关 (-1.0--0.7)',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pet_correlation_pearson_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建Spearman热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Spearman相关系数'}, 
                square=True, linewidths=0.5)
    plt.title('自变量与宠物数量的Spearman秩相关系数热力图\n(2019-2023年时间序列数据)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('宠物数量指标', fontsize=12, fontweight='bold')
    plt.ylabel('自变量', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # 添加颜色编码说明
    plt.figtext(0.02, 0.02, 
                '颜色编码规则：\n红色：强正相关 (0.7-1.0)\n橙色：中等正相关 (0.3-0.7)\n白色：弱相关 (-0.3-0.3)\n浅蓝：中等负相关 (-0.7--0.3)\n深蓝：强负相关 (-1.0--0.7)',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pet_correlation_spearman_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("=== 相关性分析结果 ===")
    print("\nPearson相关系数:")
    print(pearson_corr.round(3))
    
    print("\nSpearman秩相关系数:")
    print(spearman_corr.round(3))
    
    print("\n=== 关键发现 ===")
    print("1. 猫的数量与所有自变量都呈强正相关关系")
    print("2. 狗的数量与大多数自变量呈负相关关系")
    print("3. Pearson和Spearman结果趋势一致，说明相关性稳定")
    
    return pearson_corr, spearman_corr

if __name__ == "__main__":
    pearson_corr, spearman_corr = create_heatmaps()