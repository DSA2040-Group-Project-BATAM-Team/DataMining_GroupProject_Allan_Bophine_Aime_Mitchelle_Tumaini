# Project Title: General Health Services and Disease Data in Kenya  
**Team Members**:  
- Mitchelle – Exploratory Data Analysis  
- Tumaini – Insight and Storytelling  
- Bophine – Data Cleaning and Enrichment  
- Aime – Data Mining  
- Allan – README & Documentation  
 
---

## Project Objectives & Deliverables

| Role | Responsibilities | Deliverables |
|------|------------------|--------------|
| **Mitchelle** | Conduct EDA: distributions, trends, correlations, outliers | Jupyter notebook with visualizations; summary statistics |
| **Tumaini** | Translate findings into narratives; identify policy implications | Story-driven report; presentation slides |
| **Bophine** | Clean, validate, enrich data; document data quality issues | Cleaned dataset; data dictionary; quality report |
| **Aime** | Apply data mining: clustering, anomaly detection, forecasting | Predictive models; pattern analysis (e.g., disease hotspots) |
| **Allan** | Coordinate documentation; maintain README and project log | This README; project proposal; version control via Git |

---

## Project Overview

This project analyzes the burden of **HIV, tuberculosis, and malaria in Kenya**, using data from the **Global Burden of Disease Study 2013 (GBD 2013)** conducted by the **Institute for Health Metrics and Evaluation (IHME)**. The dataset provides harmonized estimates of **incidence, prevalence, and mortality** for these three major infectious diseases across a global set of locations—including Kenya—from **1990 to 2013**.

Our goals are to:
- Assess historical trends in HIV, TB, and malaria metrics in Kenya.
- Identify critical gaps in disease control and health service delivery.
- Support evidence-based recommendations aligned with national health strategies and global targets (e.g., WHO End TB Strategy, UNAIDS 95-95-95, Global Malaria Action Plan).
- Provide a reusable analytical foundation for future public health research.

---

## Dataset Description

### Source  
- **Primary Source**: [IHME GBD 2013 – HIV, Tuberculosis, and Malaria (1990–2013)](https://ghdx.healthdata.org/record/ihme-data/gbd-2013-hiv-tuberculosis-and-malaria-incidence-prevalence-and-mortality-1990-2013)  
- **Publication Year**: 2014  
- **Suggested Citation**:  
  > Global Burden of Disease Collaborative Network. *Global Burden of Disease Study 2013 (GBD 2013) HIV, Tuberculosis, and Malaria Incidence, Prevalence, and Mortality 1990–2013*. Seattle, United States of America: Institute for Health Metrics and Evaluation (IHME), 2014.  
- **DOI**: [https://doi.org/10.6069/ZHGK-7F54](https://doi.org/10.6069/ZHGK-7F54)  
- **License**: Data is available under the [IHME Free-of-Charge Non-Commercial User Agreement](https://ghdx.healthdata.org/terms-and-conditions). Permitted for non-commercial use, sharing, modification, and derivative works with attribution.

### Scope  
- **Diseases Covered**: HIV/AIDS, Tuberculosis, Malaria  
- **Metrics**: Incidence, Prevalence, Mortality (reported as counts and rates per 100,000)  
- **Geographic Coverage**: Global (including **Kenya**)  
- **Temporal Coverage**: **1990–2013** (historical estimates only)  
- **Stratifications**: By year, age group, sex, and location  

---

## Transformed Dataset
**Team Member**: Bophine

The raw GBD 2013 dataset was processed through a structured **ETL (Extract, Transform, Load) pipeline** to generate a cleaned, Kenya-relevant analytical dataset.

### Extract
- Source file: `Group work dsa.xlsx` (derived from IHME GBD 2013 repository)
- Initial dimensions: **750,297 rows**, 15 columns

### Transform
Key steps performed:
- **Geographic filtering**: Retained only African countries + Global aggregate  
- **Temporal filtering**: Excluded years **1990–1999** to focus on **2000–2013**  
- **Column reduction**: Dropped `location_code` and `cause_id`  
- **Missing value handling**:  
  - Numeric fields (`mean`, `lower`, `upper`) imputed with column means  
  - Categorical fields filled using mode  
- **Outlier removal**: Trimmed to **1st–99th percentiles**  
- **Noise cleaning**: Removed negative or non-physical values  
- **Feature engineering**: Added `upper_deviation_pct = ((upper – mean) / mean) × 100`  
- **Standardization**: Cleaned text casing, ensured integer years.

### 💾 Load
- Final cleaned dataset: **78,651 rows**
- Output file: `Group_work_cleaned.xlsx`
- Key columns:
  - `location_name` (e.g., Kenya, Global)
  - `year` (2000–2013)
  - `age_group_name` (e.g., "Under 5", "Age-standardized")
  - `sex_name` ("Males", "Females", "Both sexes")
  - `cause_name` ("HIV/AIDS", "Tuberculosis", "Malaria")
  - `metric` ("Incidence", "Prevalence", "Deaths")
  - `unit` ("Rate per 100,000", "Number")
  - `mean`, `lower`, `upper` (point estimate + 95% uncertainty interval)
  - `upper_deviation_pct` (derived uncertainty metric)

> **Note**: Although the cleaned dataset includes multiple African countries, all team analyses will focus **exclusively on Kenya**.

---

## 🔧 ETL Pipeline Implementation

- **Environment**: Python 3.13. 
- **Key Libraries**: `pandas`, `openpyxl`, `numpy`  
- **Dependencies Installed**:
  ```bash
  pip install openpyxl et-xmlfile
- Script: `1_extract_transform.ipynb`
- Workflow: Load → Filter → Clean → Enrich → Export

---
## Exploratory Data Analysis (EDA) Report  
**Student Name:** Mitchelle Moraa  

## Overview

This repository contains exploratory data analysis performed on a cleaned, IHME-derived dataset (`Group_work_cleaned.xlsx`) focusing on HIV/AIDS, Tuberculosis, and Malaria metrics for Kenya and other countries for 2000–2013. The purpose is to describe the data handling and EDA steps implemented in Python (Pandas / NumPy) and visualized with Matplotlib/Seaborn.

## Purpose & scope

- Provide a structured, reproducible EDA to inspect measurement variables (`mean`, `lower`, `upper`) and derived uncertainty metrics.
- Identify relationships (correlations), distribution shapes (skewness, kurtosis), and group-wise differences by disease, age group, sex, and location.
- Generate plots and summary statistics to guide follow-up modeling or policy-oriented storytelling.

## Student role & contributions

- Data handling: load, clean minimal transformations for analysis, compute uncertainty metrics.
- Visualizations: boxplots, violinplots, KDE/ridge plots, heatmaps.
- Basic interpretation and reporting of observed statistical properties (correlation, skewness, kurtosis).
- Prepared saved visual outputs to accompany the notebook.

## Data source & citation

Primary source (original dataset basis):  
Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2013 (GBD 2013) — HIV, Tuberculosis, and Malaria Incidence, Prevalence, and Mortality 1990–2013. Seattle, United States: Institute for Health Metrics and Evaluation (IHME). DOI: https://doi.org/10.6069/ZHGK-7F54

Data license: IHME Free-of-Charge Non-Commercial User Agreement — confirm permitted use before redistribution.

---

Detailed walkthrough (cell-by-cell explanation)
----------------------------------------------

1) Setup and imports
--------------------
The notebook begins by importing essential libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install openpyxl
import openpyxl
!pip install tabulate
import tabulate
```
- The notebook installs openpyxl and tabulate using pip inside the notebook. If you use a pre-configured environment, you can omit these pip install lines.
- Matplotlib & Seaborn are used for plotting.

2) Read and display the data
----------------------------
The notebook reads from an Excel file:
```python
df = pd.read_excel("C:/Users/USER/Downloads/Group_work_cleaned.xlsx")
df.head()
```
- Replace the path with your local path or a repo-relative path.
- df.head() prints the first rows to verify columns and types.

3) Correlation exploratory data analysis
----------------------------------------
The notebook computes correlations among three related numeric columns: mean, lower, upper.

Code:
```python
corr = df[["mean", "lower", "upper"]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlogram")
plt.show()
```

Interpretation in notebook:
- mean vs lower ≈ 0.98: Very strong positive correlation.
- mean vs upper ≈ 0.98: Very strong positive correlation.
- lower vs upper ≈ 0.93: Strong positive correlation.
- Conclusion: high multicollinearity — mean, lower, and upper convey highly overlapping information. If building models, include only one representative variable (or use transformations/PCA/regularization).

4) Pivot and metric-level correlation (Deaths vs Prevalence)
------------------------------------------------------------
This cell reshapes the dataset to have metrics as columns (Deaths, Prevalence) and computes correlation between them:
```python
pivot = df.pivot_table(
    index=["location_name","year","age_group_name","sex_name","cause_name"],
    columns="metric",
    values="mean"
).reset_index()
corr = pivot[["Deaths", "Prevalence"]].corr()
print(corr)
```
- The printed correlation matrix showed approximately 0.335 between Deaths and Prevalence.
- Interpretation: weak-to-moderate positive correlation — prevalence alone is a limited predictor of deaths; other confounders matter.

Important note: pivot_table will result in missing values if both metric columns aren't present for a given index. Drop or handle NaNs before computing correlations if necessary:
```python
pivot_clean = pivot.dropna(subset=["Deaths","Prevalence"])
```

5) Full numeric correlation heatmap
-----------------------------------
The notebook computes correlations across all numeric columns:
```python
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
- The notebook notes generally near-zero correlations for demographic/time identifiers vs core measurement variables, and a weak negative relationship for upper_deviation_pct with mean/lower/upper.

6) Distributions EDA (boxplots & violins per numeric variable)
--------------------------------------------------------------
The notebook selects numeric columns and plots boxplots and violin plots for each:
```python
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    sns.boxplot(x=df[col], ax=ax[0])
    ax[0].set_title(f"Boxplot: {col}")
    sns.violinplot(x=df[col], ax=ax[1])
    ax[1].set_title(f"Violin: {col}")
    plt.tight_layout()
    plt.show()
```
- This helps identify outliers, spread, and multimodality.
- Result: mean, lower, upper, and upper_deviation_pct are highly skewed (long positive tail).

7) Skewness and kurtosis calculations
-------------------------------------
The notebook computes skewness and kurtosis for numeric columns:
```python
df[num_cols].skew()
df[num_cols].kurt()
```
- Example output (from the notebook):
  - Skew:
    - location_id 3.63
    - mean 2.24, lower 2.44, upper 2.17, upper_deviation_pct 4.39
  - Kurtosis:
    - mean 4.84, lower 6.02, upper 4.48, upper_deviation_pct 23.03
- Interpretation:
  - Skew > 1: highly skewed (long-tail).
  - Kurtosis > 0: leptokurtic distribution — heavier tails and more outliers than normal.
- Recommendation: Consider log (or log1p), square root or Box-Cox transforms before using parametric models; use robust scaling or quantile transforms for ML algorithms that assume normal-like distributions.

8) Distribution by cause_name (violin & boxen plots)
----------------------------------------------------
The notebook creates violin plots for numeric columns grouped by cause_name:
```python
cat = "cause_name"
n_rows = 4
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18,20)) 
fig.suptitle(f"Distribution of Numerical Variables by {cat}", fontsize=16, y=1.02)
for i, col in enumerate(num_cols):
    row = i // n_cols
    col_idx = i % n_cols
    sns.violinplot(data=df, x=cat, y=col, ax=axes[row, col_idx], palette="viridis", inner="quartile", cut=0)
    axes[row, col_idx].set_title(f"Distribution of {col}", fontsize=12)
    axes[row, col_idx].set_xlabel(cat)
plt.tight_layout()
plt.show()
```
- A later cell uses boxenplot similarly.
- The notebook prints a FutureWarning: "Passing `palette` without assigning `hue` is deprecated..." — this warning is cosmetic; to silence it, remove palette or use hue explicitly, e.g. sns.violinplot(..., palette=None).

Interpretation from the notebook:
- HIV/AIDS shows larger magnitudes in mean / upper / lower and much higher uncertainty (upper_deviation_pct) compared to causes such as Tuberculosis.
- This suggests the HIV/AIDS group has extreme observations and larger relative uncertainty.

---

Interpretations & conclusions
-----------------------------
- mean, lower, and upper are essentially representing the same information (very high correlation). For modeling:
  - Avoid including all three simultaneously.
  - Use one (e.g., mean), or derive features (like width = upper - lower), or do dimensionality reduction (PCA), or apply regularized models (Ridge/Lasso).
- Deaths vs Prevalence correlation is weak-to-moderate (~0.33). Prevalence only partially explains variation in deaths.
- Several measurement variables are highly skewed and leptokurtic, indicating heavy tails and outliers. Consider transformations, robust statistics, or non-parametric methods.
- Certain causes (notably HIV/AIDS) drive extreme values and uncertainty; cause-specific modeling may be preferred.

Practical recommendations
-------------------------
- Data loading: use a repo-relative path and include the Excel in a data/ folder.
- If building models, derive features (e.g. relative uncertainty ratios) and use cross-validation.
- For variables with high skewness:
  - Try log1p: df['mean_log'] = np.log1p(df['mean'])
  - Use robust scalers (sklearn.preprocessing.RobustScaler) for models sensitive to outliers
- For multicollinearity:
  - Option A: Keep only one of correlated variables
  - Option B: Use PCA on mean/lower/upper to get orthogonal components
  - Option C: Use regularized regression (Ridge, Lasso, ElasticNet)
- To summarize large numbers of causes: consider grouping rare causes under "Other" or selecting top-k causes for plots to avoid overplotting.

Troubleshooting & common warnings
---------------------------------
- FileNotFoundError when reading Excel:
  - Ensure the path is correct and file exists. Use relative path or place the Excel in the repository.
- openpyxl or pandas read_excel error:
  - Ensure openpyxl is installed (pip install openpyxl).
- Seaborn FutureWarning about palette and hue:
  - To silence: remove palette or set hue explicitly. The plots still render.
- pivot_table producing NaN columns:
  - Many index combinations might not have both Deaths and Prevalence. Drop NaNs before correlation:
    pivot = pivot.dropna(subset=['Deaths','Prevalence'])
- Large dataset memory issues:
  - Use chunked reading or work with a sample when exploring.

Saving plots & outputs
----------------------
- To save figures:
```python
plt.savefig("figures/corr_heatmap.png", dpi=300, bbox_inches='tight')
```
- To save a processed DataFrame:
```python
df.to_csv("data/processed_dataset.csv", index=False)
```

Appendix: useful code snippets
------------------------------
Read Excel (example):
```python
import pandas as pd
df = pd.read_excel("data/Group_work_cleaned.xlsx", engine='openpyxl')
```

Compute correlation and plot heatmap:
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df[["mean","lower","upper"]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlogram")
plt.show()
```

Pivot to compare metrics:
```python
pivot = df.pivot_table(
    index=["location_name","year","age_group_name","sex_name","cause_name"],
    columns="metric",
    values="mean"
).reset_index()

pivot_clean = pivot.dropna(subset=["Deaths","Prevalence"])
pivot_clean[["Deaths","Prevalence"]].corr()
```

Log-transform highly skewed column:
```python
import numpy as np
df['mean_log1p'] = np.log1p(df['mean'])
```

Compute skewness & kurtosis:
```python
num_cols = df.select_dtypes(include='number').columns
skews = df[num_cols].skew()
kurts = df[num_cols].kurt()
print(skews)
print(kurts)
```

Plot grouped violin (avoid deprecation by not passing palette alone):
```python
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='cause_name', y='mean', inner='quartile')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Requirements snippet
--------------------
Add a file `requirements.txt` in the repo root with:
```
pandas
numpy
matplotlib
seaborn
openpyxl
tabulate
jupyterlab
```
---
##  Data Mining
**Team Member**: Aime

###  Dataset Overview

The dataset contains global health metrics for Tuberculosis and HIV/AIDS from 2000 to 2013.

| Column Name             | Description |
|------------------------|-------------|
| `location_id`          | Unique numeric identifier for each country or region |
| `location_name`        | Name of the country or region |
| `year`                 | Year of data collection (2000–2013) |
| `age_group_id`         | Numeric code for age group |
| `age_group_name`       | Label for age group (e.g., "Age-standardized") |
| `sex_id`               | Numeric code for sex category |
| `sex_name`             | Sex category: "Males", "Females", or "Both sexes" |
| `cause_name`           | Disease cause: "Tuberculosis" or "HIV/AIDS" |
| `metric`               | Type of health metric: "Deaths" or "Prevalence" |
| `unit`                 | Unit of measurement: "Rate per 100,000" |
| `mean`                 | Estimated central value of the metric |
| `lower`                | Lower bound of the 95% uncertainty interval |
| `upper`                | Upper bound of the 95% uncertainty interval |
| `upper_deviation_pct`  | Percent deviation from mean to upper bound |

> **Note**: The analysis focuses primarily on **death rates** (metric = "Deaths") with **age-standardized** values for fair cross-population comparison.

---

## Overview
Applied three complementary data mining techniques to uncover hidden patterns in global TB and HIV/AIDS disease burden data (2000-2013, ~78,000 records, 195 locations).

**Key Questions Answered:**
- **WHO?** Which countries share similar disease burden profiles? (Clustering)
- **WHAT?** What epidemiological patterns distinguish TB from HIV/AIDS? (Classification)
- **WHEN?** How have disease burdens changed over time and what are future projections? (Time Series)

---

## Methodology

### 1. K-Means Clustering
**Objective:** Group countries by disease burden similarity to identify regions requiring different intervention strategies.

**Features Engineered:**
- TB/HIV mean death rates (per 100,000)
- Temporal trends (deaths/year change)
- Variability (standard deviation)
- Disease ratio (HIV/TB burden)

**Process:**
1. Aggregated age-standardized death rates by location
2. Calculated trend slopes using linear regression
3. Standardized features using StandardScaler
4. Applied elbow method and silhouette analysis to determine optimal k
5. Performed K-means clustering with k=2

**Evaluation Metric:**
- Silhouette Score: 0.441 (moderate cluster separation)

<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/94981f77-d40d-476e-98c9-f26ec333a2ad" />
*Figure 1: Optimal k determination using elbow method (left) and silhouette score (right)*

<img width="1554" height="590" alt="image" src="https://github.com/user-attachments/assets/4d40d43e-43bc-4b8e-bd77-e811579e442b" />
*Figure 2: Country clusters by disease burden (left) and temporal trends (right)*

---

### 2. Decision Tree Classification
**Objective:** Predict disease type (TB vs HIV/AIDS) from epidemiological patterns to understand distinguishing characteristics.

**Features Used:**
- Age group (encoded)
- Sex (encoded)
- Death rate magnitude
- Uncertainty range (upper - lower bounds)
- Relative uncertainty (uncertainty/mean)
- Year (normalized 2000-2013)

**Process:**
1. Filtered death rate data (Rate per 100,000 unit)
2. Encoded categorical variables
3. Split data: 70% training (2,586 records), 30% testing (1,109 records)
4. Trained Decision Tree with constraints:
   - max_depth=3
 
5. Evaluated on test set

**Performance Metrics:**
- **Overall Accuracy:** 72.3%
- **Tuberculosis:** Precision=67%, Recall=91%, F1=77%
- **HIV/AIDS:** Precision=84%, Recall=53%, F1=65%

<img width="739" height="590" alt="image" src="https://github.com/user-attachments/assets/3b100c6d-8b03-4aa5-b08a-f25fcc4733fd" />
*Figure 3: Classification confusion matrix showing prediction patterns*

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/12824454-e839-4359-ad57-d819342793e5" />
*Figure 4: Feature importance - Death Rate (47%) is primary differentiator*

<img width="1990" height="1190" alt="image" src="https://github.com/user-attachments/assets/26bc1102-6954-463d-b57d-e25042111394" />
*Figure 5: Decision tree structure (top 3 levels) showing key decision rules*

---

### 3. Time Series Analysis
**Objective:** Identify temporal trends and forecast future disease burden for resource planning.

**Approach:**
1. **Trend Analysis:** Linear regression on global annual death totals
2. **Forecasting:** Exponential smoothing (Holt's method) for 3-year projections
3. **Evaluation:** Mean Absolute Percentage Error (MAPE)

**Data Preparation:**
- Aggregated all-ages, both-sexes death counts by year
- Separated TB and HIV/AIDS time series
- Calculated year-over-year percentage changes

**Forecast Performance:**
- **TB:** MAPE = 3.98% (excellent accuracy)
- **HIV/AIDS:** MAPE = 8.42% (good accuracy)

<img width="1389" height="990" alt="image" src="https://github.com/user-attachments/assets/04b6c18f-e6c1-4bcc-8428-86ff55910feb" />
*Figure 6: Historical trends showing TB stability vs HIV increase*

<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/cbeeb6bc-eeeb-4a30-821d-07649814f5db" />
*Figure 7: 3-year forecasts (2014-2016) with historical context*

---

## Key Findings

### 1. Clustering Results
**Two distinct country clusters identified:**

| Cluster | Countries (n) | TB Rate | HIV Rate | Trend | Examples |
|---------|---------------|---------|----------|-------|----------|
| 0 (Low-Moderate) | 32 | 59.4 | 76.1 | Declining | Tanzania, Cameroon |
| 1 (High Burden) | 17 | 134.5 | 394.2 | Steep decline | Zimbabwe, Lesotho |

**Insight:** Cluster 1 countries (mostly Southern Africa) have 5x higher HIV burden than Cluster 0, reflecting generalized HIV epidemics. Despite higher burden, these countries show steeper declining trends, suggesting effective intervention scale-up.

---

### 2. Classification Results
**Disease signatures learned by decision tree:**

**Most Important Features:**
1. **Death Rate (47%):** HIV/AIDS typically has higher rates in epidemic regions (threshold: ~215/100k)
2. **Relative Uncertainty (25%):** HIV data has higher uncertainty due to surveillance challenges
3. **Uncertainty Range (18%):** Reflects epidemiological confidence differences
4. **Year (8%):** Captures temporal changes in disease dynamics

**Model Behavior:**
- **Conservative on HIV prediction:** High precision (84%) but misses some cases (53% recall)
- **Aggressive on TB prediction:** Catches most cases (91% recall) but some false positives (67% precision)

**Insight:** The tree effectively learned that very high death rates (>215/100k) strongly indicate HIV/AIDS, particularly in Southern African epidemic settings.

---

### 3. Time Series Results
**Contrasting epidemic trajectories:**

| Disease | Trend | R² | 2000 Deaths | 2013 Deaths | 2016 Forecast |
|---------|-------|-----|-------------|-------------|---------------|
| **TB** | +68/year | 0.032 (weak) | 21,101 | 19,672 | 19,358 |
| **HIV/AIDS** | +321/year | 0.533 (strong) | 9,811 | 15,004 | 15,686 |

**Key Patterns:**
- **TB:** Stable endemic burden with fluctuations, no clear directional trend
- **HIV/AIDS:** Strong upward trend (53% increase 2000→2013) with high volatility
- **Forecasts:** TB expected to remain stable; HIV projected to continue rising without major interventions

**Insight:** TB represents a persistent, stable endemic problem requiring sustained control. HIV/AIDS showed ongoing epidemic expansion in the study period, necessitating aggressive intervention scale-up (which did occur post-2013 with treatment expansion).

---
## Insights & Storytelling
**Team Member**: Tumaini

### Overview
- This repository contains the Insights & Storytelling Notebook to analyze and visualize health services and disease burden data for Kenya, with a focus on HIV/AIDS and Tuberculosis (TB) from 2000–2013. The analysis leverages modeled estimates of mortality and uncertainty data from the Global Burden of Disease Study 2013, processed and cleaned for interactive exploration.

Built using Python and data visualization best practices, this notebook offers actionable insights for public health professionals, policymakers, and data analysts.

**How to Run the Analysis Code**:
- To reproduce the insights and visualizations from the Kenya health data analysis, follow the instructions below. The analysis uses Python with common data science libraries.

**Prerequisites**:
Ensure you have the following installed:

Python 3.8 or higher
Required libraries: pandas, numpy, matplotlib, seaborn, and openpyxl (for Excel file reading)
You can install them via pip:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```
**Step 1: Prepare the Data**

Make sure the cleaned dataset is saved as Group_work_cleaned.xlsx in your working directory. The file should contain a sheet with the following columns:

- `location_name`
- `year`
- `age_group_name`
- `sex_name`
- `cause_name`
- `metric` (e.g., "Deaths", "Incidence", "Prevalence")
- `mean`, `lower`, `upper`, `upper_deviation_pct`

**Step 2: Run the Analysis Script**

Copy and run the following Python code in a ipynb script:
```python
# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("Libraries loaded successfully.")

# Load the cleaned dataset
df = pd.read_excel("Group_work_cleaned.xlsx")

# Filter only Kenya
df_kenya = df[df['location_name'].str.lower() == 'kenya']
print("Full dataset rows:", len(df))
print("Kenya dataset rows:", len(df_kenya))

# Quick summary
df_kenya.info()

# Insight 1: Disease Death Trends Over Time
metric_to_plot = 'Deaths'
subset = df_kenya[df_kenya['metric'] == metric_to_plot]

plt.figure(figsize=(12,6))
sns.lineplot(data=subset, x='year', y='mean', hue='cause_name', marker='o')
plt.title(f"{metric_to_plot} Trends in Kenya (2000–2013)")
plt.xlabel("Year")
plt.ylabel(f"Mean {metric_to_plot}")
plt.legend(title="Disease")
plt.show()

print(subset.groupby('cause_name')['mean'].describe())

# Insight 2: Sex Differences in HIV/AIDS Deaths
hiv_deaths = df_kenya[
    (df_kenya['cause_name'] == 'HIV/AIDS') &
    (df_kenya['metric'] == 'Deaths')
]

sns.lineplot(data=hiv_deaths, x='year', y='mean', hue='sex_name', marker='o')
plt.title("HIV/AIDS Deaths by Sex (2000–2013)")
plt.xlabel("Year")
plt.ylabel("Mean HIV/AIDS Deaths")
plt.show()

print(hiv_deaths.groupby('sex_name')['mean'].describe())

# Insight 3: Age Group Heatmap (Year 2010)
year = 2010
subset = df_kenya[
    (df_kenya['year'] == year) &
    (df_kenya['metric'] == 'Deaths')
]

if subset.empty:
    print(f"No data available for year {year} and metric 'Deaths'.")
else:
    pivot = subset.pivot_table(
        index='age_group_name',
        columns='cause_name',
        values='mean',
        aggfunc='mean'
    )
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(f"Deaths Heatmap by Age Group & Disease ({year})")
    plt.show()


# Insight 4: Uncertainty Analysis
  
unc = df_kenya[['cause_name', 'year', 'upper_deviation_pct']]

sns.lineplot(data=unc, x='year', y='upper_deviation_pct', hue='cause_name', marker='o')
plt.title("Uncertainty (%) Over Time by Disease")
plt.xlabel("Year")
plt.ylabel("Upper Deviation (%)")
plt.show()

print(unc.groupby('cause_name')['upper_deviation_pct'].describe())
```
  

## Summary of Key Insights
### 1. Disease Burden Trends
- Overall, **HIV/AIDS** and **Tuberculosis (TB)** show significant mortality in Kenya from 2000 to 2013.
- HIV/AIDS deaths peaked around the mid-2000s, while TB shows more steady trends but also notable peaks.

### 2. Deaths Over Time by Disease
- **HIV/AIDS** remains the leading cause of death among the two diseases.
- **Tuberculosis** shows consistent but slightly lower death counts compared to HIV/AIDS.

### 3. Sex Differences (HIV/AIDS Deaths)
- For HIV/AIDS, **female deaths are slightly lower than male deaths**, with medians of ~752 (females) vs ~706 (males), but females show a wider range of extremes (~3051 vs ~3042 max).
- Both sexes combined reflect overall population burden, highlighting that public health strategies should address both male and female populations.

### 4. Age Group Vulnerabilities
- In 2010, heatmaps indicate **specific age groups** carry higher mortality for each disease.
- Typically, **older age groups and young adults** tend to exhibit higher death counts, consistent with HIV/AIDS burden among sexually active populations.

## Policy Implications

- Target **HIV/AIDS interventions** to high-burden age groups and both sexes.
- Maintain and scale **TB control programs**, especially in years/age groups with rising mortality.
- Use uncertainty metrics to prioritize data collection and refine surveillance systems.
- Public health strategies must continue monitoring trends to adapt interventions to shifts in disease burden.

---

## Limitations

- Data only covers 2000–2013.
- Only includes model-estimated deaths; raw incidence or prevalence data may differ.
- Incidence data is missing in this dataset, limiting insights into new infections or transmission rates.

---

## Expected Outputs by Team Role
### Mitchelle:
- Interactive visualizations (e.g., line charts, heatmaps) showing trends in disease metrics over time (2000–2013).
- Summary statistics comparing burden across age groups and sexes in Kenya.
- Identification of outliers or data anomalies requiring further investigation.
### Tumaini:
- A narrative report linking quantitative findings to Kenya’s public health policies during 2000–2013.
- Policy brief highlighting successes (e.g., scale-up of ART, ITN distribution) and persistent challenges (e.g., TB-HIV co-infection).
- Slide deck for stakeholder presentation with clear, compelling visuals and takeaways.
### Bophine:
- Final cleaned dataset (Group_work_cleaned.xlsx) optimized for analysis.
- Data dictionary mapping all fields to definitions and units.
- Data quality report documenting missingness, imputation methods, outlier thresholds, and limitations.
### Aime:
- Clustering models to identify high-risk demographic groups (e.g., young women with high HIV incidence).
- Anomaly detection to flag unusual reporting years or inconsistencies in trends.
- Forecasting prototypes (e.g., ARIMA, exponential smoothing) for short-term projections (not part of original data but for methodological exploration).
### Allan:
- Comprehensive project documentation including this README, Git commit log, and change history.
- Final integrated project report combining all team deliverables.
- Metadata compliance with IHME citation and license requirements.
