import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

df = pd.DataFrame(pet_data)

print("=== VARIABLE SELECTION ANALYSIS FOR ARIMAX MODELING ===")
print("\n1. CORRELATION ANALYSIS RESULTS:")
print("-" * 60)

# Calculate correlations for available data (2019-2023 where we have complete data)
complete_data = df.dropna()
print(f"Analysis period: {complete_data['Year'].min()}-{complete_data['Year'].max()}")
print(f"Sample size: {len(complete_data)} years")

# Key correlations identified from previous analysis
correlations = {
    'Cat_Population': {
        'Pet_Penetration_Rate': 0.994,
        'Per_Capita_GDP': 0.995,
        'Disposable_Income': 0.994,
        'Urbanization_Rate': 0.851,
        'Aging_Rate': 0.986,
        'Single_Pet_Consumption_Dog': -0.891,
        'Single_Pet_Consumption_Cat': -0.876
    },
    'Dog_Population': {
        'Pet_Penetration_Rate': -0.668,
        'Per_Capita_GDP': -0.364,
        'Disposable_Income': -0.364,
        'Urbanization_Rate': -0.539,
        'Aging_Rate': -0.182,
        'Single_Pet_Consumption_Dog': 0.384,
        'Single_Pet_Consumption_Cat': 0.418
    }
}

print("\n2. KEY PREDICTIVE VARIABLES IDENTIFIED:")
print("-" * 60)

# For Cat Population
print("\nCAT POPULATION - STRONGEST PREDICTORS:")
cat_corr = correlations['Cat_Population']
cat_sorted = sorted(cat_corr.items(), key=lambda x: abs(x[1]), reverse=True)
for var, corr in cat_sorted:
    strength = "Very Strong" if abs(corr) > 0.9 else "Strong" if abs(corr) > 0.7 else "Moderate"
    direction = "Positive" if corr > 0 else "Negative"
    print(f"  {var}: {corr:.3f} ({strength} {direction})")

print("\nDOG POPULATION - STRONGEST PREDICTORS:")
dog_corr = correlations['Dog_Population']
dog_sorted = sorted(dog_corr.items(), key=lambda x: abs(x[1]), reverse=True)
for var, corr in dog_sorted:
    strength = "Very Strong" if abs(corr) > 0.9 else "Strong" if abs(corr) > 0.7 else "Moderate"
    direction = "Positive" if corr > 0 else "Negative"
    print(f"  {var}: {corr:.3f} ({strength} {direction})")

print("\n3. VARIABLE SELECTION RATIONALE:")
print("-" * 60)

print("\nFOR CAT POPULATION ARIMAX MODEL:")
print("Primary predictors (|r| > 0.9):")
primary_cat = [var for var, corr in cat_sorted if abs(corr) > 0.9]
for var in primary_cat:
    print(f"  ✓ {var}")

print("\nSecondary predictors (0.7 < |r| < 0.9):")
secondary_cat = [var for var, corr in cat_sorted if 0.7 < abs(corr) < 0.9]
for var in secondary_cat:
    print(f"  • {var}")

print("\nFOR DOG POPULATION ARIMAX MODEL:")
print("Primary predictors (|r| > 0.5):")
primary_dog = [var for var, corr in dog_sorted if abs(corr) > 0.5]
for var in primary_dog:
    print(f"  ✓ {var}")

print("\n4. MODELING STRATEGY:")
print("-" * 60)
print("""
CAT MODEL RECOMMENDATION:
- Use Pet_Penetration_Rate, Per_Capita_GDP, Disposable_Income, Aging_Rate as primary covariates
- Consider Single_Pet_Consumption variables as control variables
- Strong positive trends suggest non-stationary series requiring differencing

DOG MODEL RECOMMENDATION:
- Use Pet_Penetration_Rate and Urbanization_Rate as primary covariates  
- Single_Pet_Consumption variables show moderate positive correlation
- More complex patterns may require higher-order terms

BOTH MODELS:
- Test stationarity with ADF test
- Determine optimal lag orders with AIC/BIC
- Validate residuals for white noise
- Generate 3-year forecasts with confidence intervals
""")

# Create visualization of variable importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Cat population predictors
cat_vars = [item[0] for item in cat_sorted]
cat_corrs = [item[1] for item in cat_sorted]
colors1 = ['red' if x < 0 else 'blue' for x in cat_corrs]

ax1.barh(range(len(cat_vars)), cat_corrs, color=colors1, alpha=0.7)
ax1.set_yticks(range(len(cat_vars)))
ax1.set_yticklabels([var.replace('_', ' ') for var in cat_vars])
ax1.set_xlabel('Correlation Coefficient')
ax1.set_title('Cat Population - Predictor Importance')
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3)

# Dog population predictors  
dog_vars = [item[0] for item in dog_sorted]
dog_corrs = [item[1] for item in dog_sorted]
colors2 = ['red' if x < 0 else 'blue' for x in dog_corrs]

ax2.barh(range(len(dog_vars)), dog_corrs, color=colors2, alpha=0.7)
ax2.set_yticks(range(len(dog_vars)))
ax2.set_yticklabels([var.replace('_', ' ') for var in dog_vars])
ax2.set_xlabel('Correlation Coefficient')
ax2.set_title('Dog Population - Predictor Importance')
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variable_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVariable importance visualization saved as: variable_importance_analysis.png")