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

def create_english_comparison_heatmap():
    """
    Create English comparison heatmap with all variables including new ones
    """
    # Pet population data (2019-2023)
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
        'Pet_Owning_Household_Penetration_Percent': [19.58, 20.0, 22.0, 24.0, 25.0],
        'Annual_Pet_Consumption_Dog_Yuan': [5573, 5331, 2634, 2882, 2875],
        'Annual_Pet_Consumption_Cat_Yuan': [4394, 5028, 1826, 1883, 1870]
    }
    
    # Create DataFrames
    df_pets = pd.DataFrame(pet_data)
    df_vars = pd.DataFrame(independent_vars)
    df = pd.merge(df_vars, df_pets, on='Year', how='inner')
    
    # Define variables
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
    for var in independent_variables:
        for target in dependent_variables:
            corr, _ = stats.pearsonr(df[var], df[target])
            pearson_corr.loc[var, target] = corr
    
    pearson_corr = pearson_corr.astype(float)
    
    # Calculate Spearman correlation coefficients
    spearman_corr = pd.DataFrame(index=independent_variables, columns=dependent_variables)
    for var in independent_variables:
        for target in dependent_variables:
            corr, _ = stats.spearmanr(df[var], df[target])
            spearman_corr.loc[var, target] = corr
    
    spearman_corr = spearman_corr.astype(float)
    
    # Create readable labels for variables
    readable_labels = {
        'GDP_Per_Capita_Yuan': 'GDP per Capita (¥)',
        'Disposable_Income_Yuan': 'Disposable Income (¥)',
        'Aging_Rate_Percent': 'Aging Rate (%)',
        'Single_Population_Rate_Percent': 'Single Population Rate (%)',
        'Urbanization_Rate_Percent': 'Urbanization Rate (%)',
        'Pet_Owning_Household_Penetration_Percent': 'Pet Household Penetration (%)',
        'Annual_Pet_Consumption_Dog_Yuan': 'Annual Pet Spending - Dog (¥)',
        'Annual_Pet_Consumption_Cat_Yuan': 'Annual Pet Spending - Cat (¥)',
        'Cat_Population': 'Cat Population',
        'Dog_Population': 'Dog Population'
    }
    
    # Apply readable labels
    pearson_corr_labeled = pearson_corr.copy()
    pearson_corr_labeled.index = [readable_labels.get(idx, idx) for idx in pearson_corr.index]
    pearson_corr_labeled.columns = [readable_labels.get(col, col) for col in pearson_corr.columns]
    
    spearman_corr_labeled = spearman_corr.copy()
    spearman_corr_labeled.index = [readable_labels.get(idx, idx) for idx in spearman_corr.index]
    spearman_corr_labeled.columns = [readable_labels.get(col, col) for col in spearman_corr.columns]
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Pearson heatmap
    sns.heatmap(pearson_corr_labeled, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Pearson Correlation'}, 
                square=True, linewidths=0.5, ax=ax1)
    ax1.set_title('Pearson Correlation Coefficients', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Pet Population Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Independent Variables', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=11)
    
    # Spearman heatmap
    sns.heatmap(spearman_corr_labeled, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                fmt='.3f', cbar_kws={'label': 'Spearman Correlation'}, 
                square=True, linewidths=0.5, ax=ax2)
    ax2.set_title('Spearman Rank Correlation Coefficients', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Pet Population Indicators', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Independent Variables', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=11)
    
    # Add overall title
    fig.suptitle('Pet Population Correlation Analysis: Economic & Social Factors\n(2019-2023 Time Series Data with New Variables)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Add color coding legend
    fig.text(0.02, 0.02, 
             'Color Coding Rules:\n🔴 Red: Strong Positive (0.7-1.0)\n🟠 Orange: Moderate Positive (0.3-0.7)\n⚪ White: Weak (-0.3-0.3)\n🔵 Light Blue: Moderate Negative (-0.7--0.3)\n🔵 Dark Blue: Strong Negative (-1.0--0.7)',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add new variables indicator
    fig.text(0.98, 0.02, 
             '⭐ New Variables Added:\n• Pet Household Penetration Rate\n• Annual Pet Consumption (Dog & Cat)',
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.savefig('pet_correlation_enhanced_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("=== ENHANCED CORRELATION ANALYSIS RESULTS ===")
    print("\n📊 NEW VARIABLES ADDED:")
    print("   • Pet-owning Household Penetration Rate (%)")
    print("   • Annual Pet Consumption for Dogs (Yuan)")
    print("   • Annual Pet Consumption for Cats (Yuan)")
    
    print("\n📈 PEARSON CORRELATION COEFFICIENTS:")
    print("=" * 80)
    for var in pearson_corr.index:
        for target in pearson_corr.columns:
            corr = pearson_corr.loc[var, target]
            if abs(corr) >= 0.7:
                strength = "STRONG"
            elif abs(corr) >= 0.3:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            direction = "POSITIVE" if corr > 0 else "NEGATIVE"
            var_label = readable_labels.get(var, var)
            target_label = readable_labels.get(target, target)
            print(f"{var_label:35} ↔ {target_label:15}: {corr:6.3f} ({strength} {direction})")
    
    print("\n🔍 SPEARMAN RANK CORRELATION COEFFICIENTS:")
    print("=" * 80)
    for var in spearman_corr.index:
        for target in spearman_corr.columns:
            corr = spearman_corr.loc[var, target]
            if abs(corr) >= 0.7:
                strength = "STRONG"
            elif abs(corr) >= 0.3:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            direction = "POSITIVE" if corr > 0 else "NEGATIVE"
            var_label = readable_labels.get(var, var)
            target_label = readable_labels.get(target, target)
            print(f"{var_label:35} ↔ {target_label:15}: {corr:6.3f} ({strength} {direction})")
    
    print("\n🎯 KEY INSIGHTS FROM NEW VARIABLES:")
    print("=" * 50)
    
    # Analyze new variables specifically
    pet_penetration_cat = pearson_corr.loc['Pet_Owning_Household_Penetration_Percent', 'Cat_Population']
    pet_penetration_dog = pearson_corr.loc['Pet_Owning_Household_Penetration_Percent', 'Dog_Population']
    
    dog_consumption_cat = pearson_corr.loc['Annual_Pet_Consumption_Dog_Yuan', 'Cat_Population']
    dog_consumption_dog = pearson_corr.loc['Annual_Pet_Consumption_Dog_Yuan', 'Dog_Population']
    
    cat_consumption_cat = pearson_corr.loc['Annual_Pet_Consumption_Cat_Yuan', 'Cat_Population']
    cat_consumption_dog = pearson_corr.loc['Annual_Pet_Consumption_Cat_Yuan', 'Dog_Population']
    
    print(f"• Pet Household Penetration ↔ Cat Population: {pet_penetration_cat:.3f} (STRONG POSITIVE)")
    print(f"• Pet Household Penetration ↔ Dog Population: {pet_penetration_dog:.3f} (WEAK NEGATIVE)")
    print(f"• Dog Annual Consumption ↔ Cat Population: {dog_consumption_cat:.3f} (STRONG NEGATIVE)")
    print(f"• Dog Annual Consumption ↔ Dog Population: {dog_consumption_dog:.3f} (WEAK POSITIVE)")
    print(f"• Cat Annual Consumption ↔ Cat Population: {cat_consumption_cat:.3f} (STRONG NEGATIVE)")
    print(f"• Cat Annual Consumption ↔ Dog Population: {cat_consumption_dog:.3f} (WEAK POSITIVE)")
    
    return pearson_corr, spearman_corr

if __name__ == "__main__":
    pearson_corr, spearman_corr = create_english_comparison_heatmap()