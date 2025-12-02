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

---

## Overview  

This project repo performs a comprehensive **Exploratory Data Analysis (EDA)** on a public health-related dataset titled `Group_work_cleaned.xlsx`. The analysis focuses on understanding the distribution, relationships, and uncertainty in key health metrics—particularly **Deaths** and **Prevalence**—associated with diseases such as **HIV/AIDS** and **Tuberculosis** across various demographics (age, sex), geographies (e.g., South Africa and other African nations), and time (years).

The goal is to:
- Load and validate data integrity
- Assess correlations among numerical variables
- Explore data distributions and identify skewness/kurtosis
- Visualize uncertainty in estimates
- Investigate disease patterns by age, sex, and location

All analysis is implemented using Python with core data science libraries.

---

## Tools & Libraries Used  

| Library       | Purpose |
|---------------|--------|
| `pandas`      | Data loading, cleaning, transformation, and manipulation |
| `numpy`       | Numerical operations and array handling |
| `matplotlib`  | Basic plotting and figure customization |
| `seaborn`     | Advanced statistical visualizations (heatmaps, violin plots, KDEs, etc.) |
| `openpyxl`    | Reading `.xlsx` Excel files |
| `tabulate`    | Formatted table printing (installed but not actively used) |

>  **Note**: Package installation commands (`!pip install`) confirm `openpyxl` and `tabulate` are available for file I/O and output formatting.

---

## Data Loading & Initial Inspection  

```python
df = pd.read_excel("C:/Users/USER/Downloads/Group_work_cleaned.xlsx")
df.head()
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
