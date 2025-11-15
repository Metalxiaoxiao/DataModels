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

def load_and_preprocess_data():
    """
    加载和预处理宠物数据
    """
    # 宠物数量数据 (2017-2023)
    pet_data = {
        '年份': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        '猫数量': [3500, 4064, 4412, 4862, 5806, 6536, 6980],
        '狗数量': [5246, 5085, 5503, 5222, 5429, 5119, 5175]
    }
    
    # 自变量数据 (2019-2023，与宠物数据的时间范围对齐)
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
    
    # 合并数据，只保留有自变量数据的年份 (2019-2023)
    df_merged = pd.merge(df_vars, df_pets, on='年份', how='inner')
    
    return df_merged

def calculate_correlations(df):
    """
    计算各种相关系数
    """
    # 定义自变量和因变量
    independent_variables = ['人均国内生产总值(元)', '人均可支配收入(元)', '65岁以上老龄化率(%)', 
                           '单身人口比例(%)', '城镇人口比例(%)']
    dependent_variables = ['猫数量', '狗数量']
    
    # 计算Pearson相关系数
    pearson_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    pearson_pvalues = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    
    # 计算Spearman秩相关系数
    spearman_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    spearman_pvalues = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    
    for var in independent_variables:
        for target in dependent_variables:
            # Pearson相关系数
            pearson_r, pearson_p = stats.pearsonr(df[var], df[target])
            pearson_corr.loc[var, target] = pearson_r
            pearson_pvalues.loc[var, target] = pearson_p
            
            # Spearman秩相关系数
            spearman_r, spearman_p = stats.spearmanr(df[var], df[target])
            spearman_corr.loc[var, target] = spearman_r
            spearman_pvalues.loc[var, target] = spearman_p
    
    # 转换为数值类型
    pearson_corr = pearson_corr.astype(float)
    pearson_pvalues = pearson_pvalues.astype(float)
    spearman_corr = spearman_corr.astype(float)
    spearman_pvalues = spearman_pvalues.astype(float)
    
    return {
        'pearson': {'correlation': pearson_corr, 'p_values': pearson_pvalues},
        'spearman': {'correlation': spearman_corr, 'p_values': spearman_pvalues}
    }

def create_correlation_heatmap(correlation_data, method_name, save_path=None):
    """
    创建相关性热力图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建热力图
    sns.heatmap(correlation_data, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                fmt='.3f',
                cbar_kws={'label': '相关系数'},
                square=True,
                linewidths=0.5,
                ax=ax)
    
    ax.set_title(f'自变量与宠物数量的{method_name}相关系数热力图\n(2019-2023年时间序列数据)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('宠物数量指标', fontsize=12, fontweight='bold')
    ax.set_ylabel('自变量', fontsize=12, fontweight='bold')
    
    # 调整标签
    ax.set_xticklabels(['猫的数量', '狗的数量'], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 添加颜色编码说明
    plt.figtext(0.02, 0.02, 
                '颜色编码规则：\n红色：强正相关 (0.7-1.0)\n橙色：中等正相关 (0.3-0.7)\n白色：弱相关 (-0.3-0.3)\n浅蓝：中等负相关 (-0.7--0.3)\n深蓝：强负相关 (-1.0--0.7)',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    
    plt.show()
    plt.close(fig)  # 关闭图形以释放内存

def perform_statistical_analysis():
    """
    执行完整的统计分析
    """
    print("=== 宠物数据时间序列相关性分析 ===\n")
    
    # 1. 数据预处理
    print("1. 数据预处理")
    df = load_and_preprocess_data()
    print(f"   - 数据时间范围: {df['年份'].min()}-{df['年份'].max()}")
    print(f"   - 样本数量: {len(df)} 年")
    print(f"   - 自变量数量: {len(['人均国内生产总值(元)', '人均可支配收入(元)', '65岁以上老龄化率(%)', '单身人口比例(%)', '城镇人口比例(%)'])}")
    print(f"   - 因变量数量: {len(['猫数量', '狗数量'])}")
    print("\n原始数据:")
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # 2. 计算相关性
    print("2. 相关系数计算")
    correlations = calculate_correlations(df)
    
    # 3. 分析Pearson相关性结果
    print("3. Pearson相关系数分析:")
    pearson_results = correlations['pearson']
    print("\n相关系数矩阵:")
    print(pearson_results['correlation'].round(3))
    print("\nP值矩阵:")
    print(pearson_results['p_values'].round(4))
    
    # 4. 分析Spearman相关性结果
    print("\n4. Spearman秩相关系数分析:")
    spearman_results = correlations['spearman']
    print("\n相关系数矩阵:")
    print(spearman_results['correlation'].round(3))
    print("\nP值矩阵:")
    print(spearman_results['p_values'].round(4))
    
    # 5. 创建可视化
    print("\n5. 生成相关性热力图...")
    
    # Pearson热力图
    create_correlation_heatmap(
        pearson_results['correlation'], 
        'Pearson',
        'pet_correlation_pearson_heatmap.png'
    )
    
    # Spearman热力图
    create_correlation_heatmap(
        spearman_results['correlation'], 
        'Spearman',
        'pet_correlation_spearman_heatmap.png'
    )
    
    # 6. 详细分析结果
    print("\n6. 详细分析结果:")
    print("\nPearson相关性分析:")
    for var in pearson_results['correlation'].index:
        for target in pearson_results['correlation'].columns:
            corr = pearson_results['correlation'].loc[var, target]
            p_val = pearson_results['p_values'].loc[var, target]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"   {var} ↔ {target}: r = {corr:.3f}, p = {p_val:.4f} {significance}")
    
    print("\nSpearman秩相关性分析:")
    for var in spearman_results['correlation'].index:
        for target in spearman_results['correlation'].columns:
            corr = spearman_results['correlation'].loc[var, target]
            p_val = spearman_results['p_values'].loc[var, target]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"   {var} ↔ {target}: ρ = {corr:.3f}, p = {p_val:.4f} {significance}")
    
    print("\n" + "="*80)
    print("分析完成！热力图已保存为PNG文件。")
    
    return df, correlations

if __name__ == "__main__":
    # 执行完整分析
    df, correlations = perform_statistical_analysis()