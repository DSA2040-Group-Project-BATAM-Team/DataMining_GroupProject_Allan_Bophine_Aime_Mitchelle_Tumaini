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
