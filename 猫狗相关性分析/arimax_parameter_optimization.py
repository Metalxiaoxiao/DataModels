import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== ARIMAX PARAMETER OPTIMIZATION ===")
print("=" * 60)

# Create preprocessed data directly
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

# Handle missing values
df_processed = df_original.copy()

# Linear extrapolation for Pet_Penetration_Rate
pet_pen_values = df_processed['Pet_Penetration_Rate'].dropna().values
years_with_data = df_processed[df_processed['Pet_Penetration_Rate'].notna()].index
slope = (pet_pen_values[1] - pet_pen_values[0]) / (years_with_data[1] - years_with_data[0])
df_processed.loc[2017, 'Pet_Penetration_Rate'] = pet_pen_values[0] - slope * (years_with_data[0] - 2017)
df_processed.loc[2018, 'Pet_Penetration_Rate'] = pet_pen_values[0] - slope * (years_with_data[0] - 2018)

# Handle consumption data
for col in ['Single_Pet_Consumption_Dog', 'Single_Pet_Consumption_Cat']:
    available_data = df_processed[col].dropna()
    if len(available_data) >= 3:
        avg_decline = (available_data.iloc[-1] / available_data.iloc[0]) ** (1/(len(available_data)-1))
        for year in [2017, 2018]:
            next_year_data = df_processed.loc[year+1, col]
            if not pd.isna(next_year_data):
                df_processed.loc[year, col] = next_year_data / avg_decline

# Create time series index
df_processed.index = pd.to_datetime(df_processed.index.astype(str))

print(f"Data shape: {df_processed.shape}")
print(f"Time period: {df_processed.index.min()}-{df_processed.index.max()}")

# Define target and predictor variables
cat_target = 'Cat_Population'
dog_target = 'Dog_Population'

# Primary predictors based on correlation analysis
cat_predictors = ['Pet_Penetration_Rate', 'Per_Capita_GDP', 'Disposable_Income', 'Aging_Rate']
dog_predictors = ['Pet_Penetration_Rate', 'Urbanization_Rate']

# Function to check stationarity
def check_stationarity(series, title):
    try:
        result = adfuller(series.dropna())
        return result[1] <= 0.05
    except:
        return False

# Function to make series stationary
def make_stationary(series):
    if check_stationarity(series, ""):
        return series, 0  # Already stationary
    else:
        # Try first difference
        diff_series = series.diff().dropna()
        if check_stationarity(diff_series, ""):
            return diff_series, 1  # Stationary after first differencing
        else:
            # Try second difference
            diff2_series = diff_series.diff().dropna()
            return diff2_series, 2  # Stationary after second differencing

print(f"\n1. STATIONARITY ANALYSIS")
print("-" * 40)

# Analyze stationarity for all series
targets = [cat_target, dog_target]
predictors = list(set(cat_predictors + dog_predictors))

stationarity_info = {}
for var in targets + predictors:
    series = df_processed[var]
    stationary_series, d_order = make_stationary(series)
    is_stationary = check_stationarity(stationary_series, var)
    stationarity_info[var] = {
        'original': series,
        'stationary': stationary_series,
        'd_order': d_order,
        'is_stationary': is_stationary
    }
    print(f"{var}: d={d_order}, Stationary: {is_stationary}")

# Simplified grid search for optimal ARIMAX parameters
def simple_grid_search(endog, exog, max_p=2, max_d=1, max_q=2, criterion='aic'):
    """
    Simplified grid search for optimal ARIMAX parameters
    """
    best_aic = np.inf
    best_bic = np.inf
    best_params = None
    best_model = None
    results = []
    
    print(f"\nGrid search for optimal parameters...")
    
    # Try different parameter combinations
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1): 
            for q in range(0, max_q + 1):
                try:
                    # Simple model without exogenous lags first
                    if exog is not None and len(exog) > 0:
                        # Align data
                        combined_data = pd.concat([endog, exog], axis=1).dropna()
                        if len(combined_data) < 5:
                            continue
                            
                        endog_aligned = combined_data.iloc[:, 0]
                        exog_aligned = combined_data.iloc[:, 1:]
                    else:
                        endog_aligned = endog.dropna()
                        exog_aligned = None
                    
                    if len(endog_aligned) < 5:
                        continue
                    
                    # Fit ARIMAX model
                    model = ARIMA(endog_aligned, exog=exog_aligned, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Get criteria
                    current_aic = fitted_model.aic
                    current_bic = fitted_model.bic
                    
                    # Store results
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'aic': current_aic, 'bic': current_bic,
                        'params': len(fitted_model.params)
                    })
                    
                    # Update best model
                    if criterion == 'aic' and current_aic < best_aic:
                        best_aic = current_aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                    elif criterion == 'bic' and current_bic < best_bic:
                        best_bic = current_bic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except Exception as e:
                    continue  # Skip problematic combinations
    
    return best_model, best_params, results

# Optimize Cat Population Model
print(f"\n2. CAT POPULATION ARIMAX OPTIMIZATION")
print("-" * 50)

cat_endog = stationarity_info[cat_target]['stationary']
cat_exog = df_processed[cat_predictors]

cat_best_model, cat_best_params, cat_results = simple_grid_search(
    cat_endog, cat_exog, max_p=2, max_d=1, max_q=2
)

if cat_best_model is not None:
    print(f"Best Cat Model Parameters (p,d,q): {cat_best_params}")
    print(f"AIC: {cat_best_model.aic:.2f}")
    print(f"BIC: {cat_best_model.bic:.2f}")
    print(f"Log-likelihood: {cat_best_model.llf:.2f}")
else:
    print("Cat model optimization failed - using default parameters")
    cat_best_params = (1, 1, 1)

# Optimize Dog Population Model  
print(f"\n3. DOG POPULATION ARIMAX OPTIMIZATION")
print("-" * 50)

dog_endog = stationarity_info[dog_target]['stationary']
dog_exog = df_processed[dog_predictors]

dog_best_model, dog_best_params, dog_results = simple_grid_search(
    dog_endog, dog_exog, max_p=2, max_d=1, max_q=2
)

if dog_best_model is not None:
    print(f"Best Dog Model Parameters (p,d,q): {dog_best_params}")
    print(f"AIC: {dog_best_model.aic:.2f}")
    print(f"BIC: {dog_best_model.bic:.2f}")
    print(f"Log-likelihood: {dog_best_model.llf:.2f}")
else:
    print("Dog model optimization failed - using default parameters")
    dog_best_params = (1, 1, 1)

# Model comparison and validation
print(f"\n4. MODEL VALIDATION")
print("-" * 50)

def validate_model(model, model_name):
    if model is None:
        print(f"{model_name}: Model not available for validation")
        return False, False
        
    print(f"\n{model_name} Validation:")
    
    try:
        # Residual diagnostics
        residuals = model.resid
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=3, return_df=True)
        print(f"Ljung-Box Test p-values: {lb_test['lb_pvalue'].values}")
        
        # Check if residuals are white noise (p > 0.05)
        white_noise = all(lb_test['lb_pvalue'] > 0.05)
        print(f"Residuals are white noise: {white_noise}")
        
        # Normality test
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        print(f"Jarque-Bera test p-value: {jb_pvalue:.4f}")
        normal_residuals = jb_pvalue > 0.05
        print(f"Residuals are normally distributed: {normal_residuals}")
        
        return white_noise, normal_residuals
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False, False

# Validate both models
cat_validation = validate_model(cat_best_model, "Cat Population")
dog_validation = validate_model(dog_best_model, "Dog Population")

# Create parameter comparison visualization if we have results
if cat_results and dog_results:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Top models by AIC
    cat_results_df = pd.DataFrame(cat_results)
    dog_results_df = pd.DataFrame(dog_results)

    cat_top = cat_results_df.nsmallest(10, 'aic')
    dog_top = dog_results_df.nsmallest(10, 'aic')

    # Plot AIC comparison
    ax1.bar(range(len(cat_top)), cat_top['aic'], alpha=0.7, color='blue')
    ax1.set_title('Cat Model - Top 10 AIC Scores')
    ax1.set_xlabel('Model Rank')
    ax1.set_ylabel('AIC')

    ax2.bar(range(len(dog_top)), dog_top['aic'], alpha=0.7, color='red')
    ax2.set_title('Dog Model - Top 10 AIC Scores')
    ax2.set_xlabel('Model Rank')
    ax2.set_ylabel('AIC')

    # Parameter distribution
    cat_params = cat_results_df.groupby(['p', 'd', 'q']).size().reset_index(name='count')
    dog_params = dog_results_df.groupby(['p', 'd', 'q']).size().reset_index(name='count')

    if len(cat_params) > 0:
        ax3.scatter(cat_params['p'], cat_params['d'], s=cat_params['count']*20, alpha=0.6, color='blue')
        ax3.set_title('Cat Model - Parameter Distribution')
        ax3.set_xlabel('p (AR order)')
        ax3.set_ylabel('d (Differencing)')

    if len(dog_params) > 0:
        ax4.scatter(dog_params['p'], dog_params['d'], s=dog_params['count']*20, alpha=0.6, color='red')
        ax4.set_title('Dog Model - Parameter Distribution')
        ax4.set_xlabel('p (AR order)')
        ax4.set_ylabel('d (Differencing)')

    plt.tight_layout()
    plt.savefig('arimax_parameter_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save optimal parameters
optimal_params = {
    'cat_model': {
        'order': cat_best_params,
        'predictors': cat_predictors,
        'aic': cat_best_model.aic if cat_best_model else None,
        'bic': cat_best_model.bic if cat_best_model else None,
        'validation': cat_validation
    },
    'dog_model': {
        'order': dog_best_params,
        'predictors': dog_predictors,
        'aic': dog_best_model.aic if dog_best_model else None,
        'bic': dog_best_model.bic if dog_best_model else None,
        'validation': dog_validation
    }
}

import json
with open('optimal_arimax_parameters.json', 'w') as f:
    json.dump(optimal_params, f, indent=2, default=str)

print(f"\n5. OPTIMIZATION RESULTS SUMMARY")
print("-" * 50)
print(f"Cat Model: ARIMAX{cat_best_params}")
if cat_best_model:
    print(f"  - AIC: {cat_best_model.aic:.2f}, BIC: {cat_best_model.bic:.2f}")
print(f"  - Predictors: {cat_predictors}")
print(f"  - Validation: White noise residuals: {cat_validation[0]}")

print(f"\nDog Model: ARIMAX{dog_best_params}")
if dog_best_model:
    print(f"  - AIC: {dog_best_model.aic:.2f}, BIC: {dog_best_model.bic:.2f}")
print(f"  - Predictors: {dog_predictors}")
print(f"  - Validation: White noise residuals: {dog_validation[0]}")

print(f"\nParameter optimization visualization saved: arimax_parameter_optimization.png")
print(f"Optimal parameters saved: optimal_arimax_parameters.json")