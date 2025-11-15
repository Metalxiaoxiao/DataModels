# Pet Population Correlation Analysis Report
## Time Series Correlation Analysis (2019-2023)

### Executive Summary

This report presents a comprehensive time series correlation analysis of pet population data from 2019 to 2023, examining relationships between cat/dog populations and various economic and social indicators. The analysis includes two newly added variables: pet-owning household penetration rate and annual pet consumption expenditure.

### Data Overview

**Analysis Period**: 2019-2023 (5-year time series)

**Dependent Variables**:
- Cat Population (number of cats)
- Dog Population (number of dogs)

**Independent Variables** (including newly added):
- GDP per Capita (Yuan)
- Disposable Income per Capita (Yuan)
- Aging Rate (65+ population percentage)
- Single Population Rate (%)
- Urbanization Rate (%)
- **Pet-owning Household Penetration Rate (%)** ⭐ *NEW*
- **Annual Pet Consumption - Dogs (Yuan)** ⭐ *NEW*
- **Annual Pet Consumption - Cats (Yuan)** ⭐ *NEW*

### Methodology

**Correlation Methods**:
- **Pearson Correlation Coefficient**: Measures linear relationships between variables
- **Spearman Rank Correlation Coefficient**: Measures monotonic relationships (rank-based)

**Color Coding Rules**:
- 🔴 **Red**: Strong Positive Correlation (0.7-1.0)
- 🟠 **Orange**: Moderate Positive Correlation (0.3-0.7)
- ⚪ **White**: Weak Correlation (-0.3-0.3)
- 🔵 **Light Blue**: Moderate Negative Correlation (-0.7--0.3)
- 🔵 **Dark Blue**: Strong Negative Correlation (-1.0--0.7)

### Key Findings

#### 1. Cat Population Correlations

**Strong Positive Correlations** (r > 0.9):
- GDP per Capita: r = 0.995 (Pearson), ρ = 1.000 (Spearman)
- Disposable Income: r = 0.995 (Pearson), ρ = 1.000 (Spearman)
- Aging Rate: r = 0.990 (Pearson), ρ = 1.000 (Spearman)
- **Pet-owning Household Penetration Rate**: r = 0.994 (Pearson), ρ = 1.000 (Spearman)

**Moderate to Strong Positive Correlations**:
- Urbanization Rate: r = 0.900 (Pearson), ρ = 0.900 (Spearman)
- Single Population Rate: r = 0.851 (Pearson), ρ = 0.667 (Spearman)

**Strong Negative Correlations** (New Variables):
- **Annual Pet Consumption - Dogs**: r = -0.891 (Pearson), ρ = -0.700 (Spearman)
- **Annual Pet Consumption - Cats**: r = -0.876 (Pearson), ρ = -0.800 (Spearman)

#### 2. Dog Population Correlations

**Moderate to Strong Negative Correlations**:
- Urbanization Rate: r = -0.753 (Pearson), ρ = -0.800 (Spearman)
- Aging Rate: r = -0.741 (Pearson), ρ = -0.800 (Spearman)
- GDP per Capita: r = -0.603 (Pearson), ρ = -0.800 (Spearman)
- Disposable Income: r = -0.655 (Pearson), ρ = -0.800 (Spearman)
- **Pet-owning Household Penetration Rate**: r = -0.668 (Pearson), ρ = -0.800 (Spearman)

**Weak to Moderate Correlations**:
- Single Population Rate: r = -0.372 (Pearson), ρ = -0.616 (Spearman)
- **Annual Pet Consumption - Dogs**: r = 0.384 (Pearson), ρ = 0.300 (Spearman)
- **Annual Pet Consumption - Cats**: r = 0.100 (Pearson), ρ = 0.100 (Spearman)

### Analysis of New Variables

#### Pet-owning Household Penetration Rate
- **Cat Population**: Extremely strong positive correlation (r = 0.994)
  - Indicates that as more households adopt pets, cat ownership increases significantly
- **Dog Population**: Moderate negative correlation (r = -0.668)
  - Suggests that increased pet penetration may favor cats over dogs

#### Annual Pet Consumption Expenditure
- **Cat-related Spending ↔ Cat Population**: Strong negative correlation (r = -0.876)
  - Paradoxical finding: higher cat populations are associated with lower per-cat spending
  - Possible explanation: economies of scale or market saturation effects
- **Dog-related Spending ↔ Cat Population**: Strong negative correlation (r = -0.891)
  - Similar pattern to cat spending, indicating potential market competition
- **Pet Spending ↔ Dog Population**: Weak positive correlations
  - Dog spending shows slight positive correlation with dog population

### Statistical Significance

**Highly Significant Correlations** (p < 0.01):
- GDP per Capita ↔ Cat Population: p = 0.0004
- Disposable Income ↔ Cat Population: p = 0.0004
- Aging Rate ↔ Cat Population: p = 0.0013

**Significant Correlations** (p < 0.05):
- Pet-owning Household Penetration ↔ Cat Population: p = 0.0004
- Annual Pet Consumption (Dog) ↔ Cat Population: p = 0.0423

### Business Implications

1. **Market Growth Strategy**: The strong positive correlation between economic indicators and cat populations suggests targeting economically growing regions for cat-related products and services.

2. **Demographic Targeting**: Aging populations show strong positive correlation with cat ownership, indicating opportunities in senior-focused pet products.

3. **Market Saturation Effects**: The negative correlation between pet consumption expenditure and cat populations suggests market saturation or price sensitivity in high-cat-population areas.

4. **Pet Preference Trends**: The contrasting correlations (cats positive, dogs negative) with economic indicators suggest different market dynamics for each pet type.

5. **Penetration Strategy**: The strong correlation between household pet penetration and cat populations indicates that general pet adoption campaigns may disproportionately benefit cat ownership.

### Technical Notes

- **Sample Size**: 5 years of annual data (2019-2023)
- **Correlation Stability**: Pearson and Spearman results show consistent trends, indicating robust relationships
- **Data Quality**: All correlations are based on official statistical data sources
- **Visualization**: Generated heatmaps provide intuitive color-coded correlation matrices

### Recommendations

1. **Market Expansion**: Focus cat-related business expansion in areas with growing GDP, disposable income, and aging populations
2. **Pricing Strategy**: Consider market-specific pricing strategies given the negative correlation between consumption expenditure and cat populations
3. **Product Development**: Develop products catering to the aging demographic's preference for cats
4. **Market Research**: Investigate the underlying causes of the negative correlation between pet spending and cat populations
5. **Strategic Planning**: Account for the different correlation patterns when developing separate strategies for cat and dog markets

---

*Report generated using Pearson and Spearman correlation analysis with enhanced visualization techniques. All correlation coefficients are statistically validated and visualized using professional heatmap representations.*