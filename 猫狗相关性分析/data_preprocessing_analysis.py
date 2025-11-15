import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set font for plotting
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== DATA PREPROCESSING ANALYSIS FOR ARIMAX MODELING ===")
print("=" * 70)

# Historical pet data (2017-2023)
pet_data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Cat_Population': [5806, 6700, 7656, 8206, 8864, 9446, 10078],  # in 10,000s
    'Dog_Population': [8746, 9149, 8954, 8682, 8826, 9114, 9325],  # in 10,000s
    'Pet_Penetration_Rate': [np.nan, np.nan, 19.58, 21.88, 22.88, 23.88, 25.0],  # %
    'Per_Capita_GDP': [59660, 64644, 70078, 71828, 80976, 85698, 89358],  # yuan
    'Disposable_Income': [25974, 28228, 30733, 32189, 35128, 36883, 39218],  # yuan
    'Urbanization_Rate': [58.52, 59.58, 60.6, 63.89, 64.72, 65.22, 66.16],  # %
    'Aging_Rate': [11.4, 11.9, 12.6, 13.5, 14.2, 14.9, 15.35],  # %
    'Single_Pet_Consumption_Dog': [np.nan, np.nan, 5573, 5029, 3349, 2875, 2634],  # yuan/dog
    'Single_Pet_Consumption_Cat': [np.nan, np.nan, 4394, 4010, 2588, 1870, 1826]  # yuan/cat
}

df_original = pd.DataFrame(pet_data)
df_original.set_index('Year', inplace=True)

print(f"1. ORIGINAL DATA OVERVIEW")
print(f"   Time period: {df_original.index.min()}-{df_original.index.max()}")
print(f"   Total observations: {len(df_original)}")
print(f"   Variables: {list(df_original.columns)}")

# 1. MISSING VALUE ANALYSIS
print(f"\n2. MISSING VALUE ANALYSIS")
print("-" * 50)
missing_summary = df_original.isnull().sum()
missing_pct = (missing_summary / len(df_original)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_summary,
    'Missing_Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Handle missing values
print(f"\n3. MISSING VALUE TREATMENT STRATEGY")
print("-" * 50)

df_processed = df_original.copy()

# For Pet_Penetration_Rate: Linear interpolation for 2017-2018
pet_pen_values = df_processed['Pet_Penetration_Rate'].dropna().values
years_with_data = df_processed[df_processed['Pet_Penetration_Rate'].notna()].index

# Linear extrapolation backwards
slope = (pet_pen_values[1] - pet_pen_values[0]) / (years_with_data[1] - years_with_data[0])
for year in [2017, 2018]:
    if pd.isna(df_processed.loc[year, 'Pet_Penetration_Rate']):
        df_processed.loc[year, 'Pet_Penetration_Rate'] = pet_pen_values[0] - slope * (years_with_data[0] - year)

# For consumption data: Use trend-based interpolation
for col in ['Single_Pet_Consumption_Dog', 'Single_Pet_Consumption_Cat']:
    # Calculate year-over-year change rate
    available_data = df_processed[col].dropna()
    if len(available_data) >= 3:
        avg_decline = (available_data.iloc[-1] / available_data.iloc[0]) ** (1/(len(available_data)-1))
        
        # Backward extrapolation
        for year in [2017, 2018]:
            if pd.isna(df_processed.loc[year, col]):
                next_year_data = df_processed.loc[year+1, col]
                if not pd.isna(next_year_data):
                    df_processed.loc[year, col] = next_year_data / avg_decline

print("Missing values treatment completed.")
print(f"Final dataset shape: {df_processed.shape}")
print(f"Missing values remaining: {df_processed.isnull().sum().sum()}")

# 2. OUTLIER DETECTION
print(f"\n4. OUTLIER DETECTION ANALYSIS")
print("-" * 50)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

outlier_summary = {}
for col in df_processed.columns:
    if col != 'Year':
        outliers, lower, upper = detect_outliers_iqr(df_processed, col)
        outlier_summary[col] = {
            'count': len(outliers),
            'lower_bound': lower,
            'upper_bound': upper,
            'outlier_years': outliers.index.tolist() if len(outliers) > 0 else []
        }

print("Outlier Detection Results (IQR Method):")
for var, summary in outlier_summary.items():
    if summary['count'] > 0:
        print(f"  {var}: {summary['count']} outliers in years {summary['outlier_years']}")
    else:
        print(f"  {var}: No outliers detected")

# 3. STATIONARITY TESTS
print(f"\n5. STATIONARITY TESTS (ADF TEST)")
print("-" * 50)

def adf_test(series, title):
    result = adfuller(series.dropna())
    print(f"\n{title}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] <= 0.05:
        print(f"  Result: STATIONARY (reject null hypothesis)")
        return True
    else:
        print(f"  Result: NON-STATIONARY (fail to reject null hypothesis)")
        return False

stationarity_results = {}
for col in ['Cat_Population', 'Dog_Population', 'Pet_Penetration_Rate', 
           'Per_Capita_GDP', 'Disposable_Income', 'Urbanization_Rate', 'Aging_Rate']:
    is_stationary = adf_test(df_processed[col], col)
    stationarity_results[col] = is_stationary

# Test first differences for non-stationary series
print(f"\n6. FIRST DIFFERENCE STATIONARITY TESTS")
print("-" * 50)

for var, is_stat in stationarity_results.items():
    if not is_stat:
        diff_series = df_processed[var].diff().dropna()
        print(f"\n{var} (First Difference):")
        result = adfuller(diff_series)
        print(f"  ADF Statistic: {result[0]:.4f}")
        print(f"  p-value: {result[1]:.4f}")
        if result[1] <= 0.05:
            print(f"  Result: STATIONARY after first differencing")
        else:
            print(f"  Result: Still non-stationary")

# 4. DESCRIPTIVE STATISTICS
print(f"\n7. DESCRIPTIVE STATISTICS")
print("-" * 50)
print(df_processed.describe().round(2))

# 5. DATA VISUALIZATION
print(f"\n8. CREATING DATA VISUALIZATION")
print("-" * 50)

# Time series plots
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()

variables = df_processed.columns[:9]  # Plot first 9 variables
for i, var in enumerate(variables):
    axes[i].plot(df_processed.index, df_processed[var], marker='o', linewidth=2)
    axes[i].set_title(var.replace('_', ' '))
    axes[i].set_xlabel('Year')
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('time_series_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation matrix heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df_processed.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix - All Variables')
plt.tight_layout()
plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
plt.close()

# First differences plots for non-stationary series
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
non_stationary_vars = [var for var, is_stat in stationarity_results.items() if not is_stat]

for i, var in enumerate(non_stationary_vars[:4]):  # Plot first 4 non-stationary
    row, col = i // 2, i % 2
    diff_data = df_processed[var].diff().dropna()
    
    axes[row, col].plot(diff_data.index[1:], diff_data.values, marker='o', color='red', alpha=0.7)
    axes[row, col].set_title(f'{var} - First Difference')
    axes[row, col].set_xlabel('Year')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('first_differences_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Data visualization completed.")
print("Generated files:")
print("  - time_series_overview.png")
print("  - correlation_matrix_full.png") 
print("  - first_differences_plot.png")

# 6. PREPROCESSED DATA SUMMARY
print(f"\n9. PREPROCESSED DATA SUMMARY")
print("-" * 50)
print(f"Final dataset shape: {df_processed.shape}")
print(f"Missing values: {df_processed.isnull().sum().sum()}")
print(f"Outliers detected: {sum([summary['count'] for summary in outlier_summary.values()])}")
print(f"Non-stationary variables requiring differencing: {len([var for var, is_stat in stationarity_results.items() if not is_stat])}")

# Save processed data
df_processed.to_csv('preprocessed_pet_data.csv')
print(f"\nPreprocessed data saved as: preprocessed_pet_data.csv")

print(f"\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETED - READY FOR ARIMAX MODELING")
print("=" * 70)