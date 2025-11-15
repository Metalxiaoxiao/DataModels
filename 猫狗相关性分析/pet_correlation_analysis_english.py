import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set English font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def load_and_preprocess_data():
    """
    Load and preprocess pet data with new variables
    """
    # Pet population data (2019-2023, aligned with independent variables)
    pet_data = {
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Cat_Population': [4412, 4862, 5806, 6536, 6980],
        'Dog_Population': [5503, 5222, 5429, 5119, 5175]
    }
    
    # Independent variables data (including new variables)
    independent_vars = {
        'Year': [2019, 2020, 2021, 2022, 2023],
        'GDP_Per_Capita_Yuan': [71453, 73338, 83111, 87385, 91746],
        'Disposable_Income_Yuan': [30733, 32189, 35128, 36883, 39218],
        'Aging_Rate_Percent': [12.60, 13.50, 14.20, 14.90, 15.40],
        'Single_Population_Rate_Percent': [15.00, 15.00, 17.00, 17.10, 16.60],
        'Urbanization_Rate_Percent': [60.60, 63.89, 64.72, 65.22, 66.16],
        'Pet_Owning_Household_Penetration_Percent': [19.58, 20.0, 22.0, 24.0, 25.0],  # New variable
        'Annual_Pet_Consumption_Dog_Yuan': [5573, 5331, 2634, 2882, 2875],  # New variable
        'Annual_Pet_Consumption_Cat_Yuan': [4394, 5028, 1826, 1883, 1870]   # New variable
    }
    
    # Create DataFrames
    df_pets = pd.DataFrame(pet_data)
    df_vars = pd.DataFrame(independent_vars)
    
    # Merge data
    df_merged = pd.merge(df_vars, df_pets, on='Year', how='inner')
    
    return df_merged

def calculate_correlations(df):
    """
    Calculate various correlation coefficients
    """
    # Define independent and dependent variables
    independent_variables = [
        'GDP_Per_Capita_Yuan', 
        'Disposable_Income_Yuan', 
        'Aging_Rate_Percent',
        'Single_Population_Rate_Percent', 
        'Urbanization_Rate_Percent',
        'Pet_Owning_Household_Penetration_Percent',
        'Annual_Pet_Consumption_Dog_Yuan',
        'Annual_Pet_Consumption_Cat_Yuan'
    ]
    
    dependent_variables = ['Cat_Population', 'Dog_Population']
    
    # Calculate Pearson correlation coefficients
    pearson_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    pearson_pvalues = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    
    # Calculate Spearman rank correlation coefficients
    spearman_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    spearman_pvalues = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    
    for var in independent_variables:
        for target in dependent_variables:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(df[var], df[target])
            pearson_corr.loc[var, target] = pearson_r
            pearson_pvalues.loc[var, target] = pearson_p
            
            # Spearman rank correlation
            spearman_r, spearman_p = stats.spearmanr(df[var], df[target])
            spearman_corr.loc[var, target] = spearman_r
            spearman_pvalues.loc[var, target] = spearman_p
    
    # Convert to numeric
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
    Create correlation heatmap in English
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create readable labels for variables
    readable_labels = {
        'GDP_Per_Capita_Yuan': 'GDP per Capita (Yuan)',
        'Disposable_Income_Yuan': 'Disposable Income (Yuan)',
        'Aging_Rate_Percent': 'Aging Rate (%)',
        'Single_Population_Rate_Percent': 'Single Population Rate (%)',
        'Urbanization_Rate_Percent': 'Urbanization Rate (%)',
        'Pet_Owning_Household_Penetration_Percent': 'Pet-owning Household Penetration (%)',
        'Annual_Pet_Consumption_Dog_Yuan': 'Annual Pet Consumption - Dog (Yuan)',
        'Annual_Pet_Consumption_Cat_Yuan': 'Annual Pet Consumption - Cat (Yuan)',
        'Cat_Population': 'Cat Population',
        'Dog_Population': 'Dog Population'
    }
    
    # Apply readable labels
    correlation_data_labeled = correlation_data.copy()
    correlation_data_labeled.index = [readable_labels.get(idx, idx) for idx in correlation_data.index]
    correlation_data_labeled.columns = [readable_labels.get(col, col) for col in correlation_data.columns]
    
    # Create heatmap
    sns.heatmap(correlation_data_labeled, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'},
                square=True,
                linewidths=0.5,
                ax=ax)
    
    ax.set_title(f'Correlation Heatmap: Independent Variables vs Pet Population\n({method_name} Correlation Coefficients, 2019-2023)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Pet Population Indicators', fontsize=14, fontweight='bold')
    ax.set_ylabel('Independent Variables', fontsize=14, fontweight='bold')
    
    # Adjust labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Add color coding legend
    plt.figtext(0.02, 0.02, 
                'Color Coding Rules:\nRed: Strong Positive (0.7-1.0)\nOrange: Moderate Positive (0.3-0.7)\nWhite: Weak (-0.3-0.3)\nLight Blue: Moderate Negative (-0.7--0.3)\nDark Blue: Strong Negative (-1.0--0.7)',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def perform_statistical_analysis():
    """
    Perform complete statistical analysis in English
    """
    print("=== Pet Data Time Series Correlation Analysis (Enhanced) ===\n")
    
    # 1. Data preprocessing
    print("1. Data Preprocessing")
    df = load_and_preprocess_data()
    print(f"   - Time Range: {df['Year'].min()}-{df['Year'].max()}")
    print(f"   - Sample Size: {len(df)} years")
    print(f"   - Independent Variables: {len(['GDP_Per_Capita_Yuan', 'Disposable_Income_Yuan', 'Aging_Rate_Percent', 'Single_Population_Rate_Percent', 'Urbanization_Rate_Percent', 'Pet_Owning_Household_Penetration_Percent', 'Annual_Pet_Consumption_Dog_Yuan', 'Annual_Pet_Consumption_Cat_Yuan'])}")
    print(f"   - Dependent Variables: {len(['Cat_Population', 'Dog_Population'])}")
    print("\nOriginal Data:")
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # 2. Calculate correlations
    print("2. Correlation Calculation")
    correlations = calculate_correlations(df)
    
    # 3. Analyze Pearson correlation results
    print("3. Pearson Correlation Analysis:")
    pearson_results = correlations['pearson']
    print("\nCorrelation Matrix:")
    print(pearson_results['correlation'].round(3))
    print("\nP-values Matrix:")
    print(pearson_results['p_values'].round(4))
    
    # 4. Analyze Spearman correlation results
    print("\n4. Spearman Rank Correlation Analysis:")
    spearman_results = correlations['spearman']
    print("\nCorrelation Matrix:")
    print(spearman_results['correlation'].round(3))
    print("\nP-values Matrix:")
    print(spearman_results['p_values'].round(4))
    
    # 5. Create visualizations
    print("\n5. Generating Correlation Heatmaps...")
    
    # Pearson heatmap
    create_correlation_heatmap(
        pearson_results['correlation'], 
        'Pearson',
        'pet_correlation_enhanced_pearson_heatmap.png'
    )
    
    # Spearman heatmap
    create_correlation_heatmap(
        spearman_results['correlation'], 
        'Spearman',
        'pet_correlation_enhanced_spearman_heatmap.png'
    )
    
    # 6. Detailed analysis results
    print("\n6. Detailed Analysis Results:")
    print("\nPearson Correlation Analysis:")
    for var in pearson_results['correlation'].index:
        for target in pearson_results['correlation'].columns:
            corr = pearson_results['correlation'].loc[var, target]
            p_val = pearson_results['p_values'].loc[var, target]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"   {var} ↔ {target}: r = {corr:.3f}, p = {p_val:.4f} {significance}")
    
    print("\nSpearman Rank Correlation Analysis:")
    for var in spearman_results['correlation'].index:
        for target in spearman_results['correlation'].columns:
            corr = spearman_results['correlation'].loc[var, target]
            p_val = spearman_results['p_values'].loc[var, target]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"   {var} ↔ {target}: ρ = {corr:.3f}, p = {p_val:.4f} {significance}")
    
    print("\n" + "="*80)
    print("Analysis Complete! Enhanced heatmaps with new variables have been generated.")
    
    return df, correlations

if __name__ == "__main__":
    # Execute complete analysis
    df, correlations = perform_statistical_analysis()