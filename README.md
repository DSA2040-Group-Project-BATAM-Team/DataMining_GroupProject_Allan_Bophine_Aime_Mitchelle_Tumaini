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

##  Setup & Dependencies

### Required Python Libraries
- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – Clustering and classification
- `mlxtend` – Association rule mining
- `warnings` – Suppress non-critical alerts

### Installation
Run the following in your terminal or notebook environment:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

 ## Methodology Breakdown
 ### Data Preparation
- Filter data into Deaths (63,742 records) and Prevalence (14,822 records)
- Create an **aggregated location-level dataset** by grouping on `location_id` and cause_name
- Compute average rates (`avg_mean`), uncertainty bounds, and record counts
- Pivot the data to create separate columns for TB and HIV/AIDS metrics per location
- Handle missing values by imputing 0 (indicating no reported cases/data)

Result: **49 locations** with numeric features suitable for clustering.

## Technique 1: K-Means Clustering
**Goal**: Discover natural groupings of countries based on disease burden profiles.

### Features Used
- 'avg_mean_HIV/AIDS', 'avg_mean_Tuberculosis'
- 'avg_upper_HIV/AIDS', 'avg_upper_Tuberculosis'
- 'record_count_HIV/AIDS', 'record_count_Tuberculosis'
  
### Workflow
1. Standardize all numeric features using StandardScaler
2. Evaluate cluster quality using:
- Elbow Method (inertia vs. k)
- Silhouette Score (cohesion vs. separation)
3. Select optimal k = 4 (highest silhouette score)
4. Fit KMeans(n_clusters=4) and assign cluster labels
5. Analyze and interpret each cluster

| Cluster | Average HIV Mean | Average TB Mean | Sample Countries |
|---------|------------------|-----------------|------------------|
| **0** | 1036.48 | 789.26 | Ethiopia, Kenya, South Africa, Zimbabwe |
| **1** | 360.86 | 254.76 | Algeria, Egypt, Angola, Morocco |
| **2** | 118.34 | 53.51 | Global, Libya, Tunisia, Seychelles |
| **3** | 346.74 | 337.04 | Sudan, South Sudan |

> **Insight**: Cluster 0 represents the high dual-burden region (Sub-Saharan Africa), while Cluster 2 includes low-burden or global aggregate entries.

### Visualization
- Scatter plot of TB vs. HIV mean rates, colored by cluster
- Clear separation between high, medium, and low burden regions

## Technique 2: Decision Tree Classification
**Goal**: Predict whether a location belongs to the high disease burden group and identify decisive factors.

### Target Variable Construction
- Compute overall `avg_death_rate` per location (age-standardized deaths/100k)
- Define **high burden** as values above the **75th percentile** (~182.39)
- Result: **12 high-burden**, **37 low-burden** locations
  
### Features
- `tb_death_rate`: Avg TB death rate
- `hiv_death_rate`: Avg HIV/AIDS death rate
- `uncertainty_range`: avg_upper_bound – avg_lower_bound
- `num_records`: Data completeness proxy
  
### Model Configuration
- `DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5)`
- Stratified 70/30 train-test split

### Performance
- **Test Accuracy: 100%** (due to small, well-separated classes)
- **Classification Report**: Perfect precision, recall, and F1 for both classes

| Feature | Importance |
|---------|------------------|
| `hiv_death_rate` | 1.00 | 
| All Others | 0.00 | 
> **Key Insight**: HIV/AIDS mortality **alone** is sufficient to distinguish high-burden locations — TB rates do not provide additional discriminative power in this context.

### Visualization
- Bar plot of feature importance
- Full decision tree diagram showing splits based exclusively on `hiv_death_rate`

## Technique 3: Association Rule Mining (Apriori)
**Goal**: Uncover frequent co-occurrences among disease, demographic, and temporal attributes.

### Transaction Construction
Each death record is converted into a market basket of categorical items:

- Disease: Tuberculosis or HIV_AIDS
- Sex: Sex_Males, Sex_Females, Sex_Both sexes
- Age: Age_Age-standardized
- Rate Category: Rate_Very_Low, Rate_Low, Rate_Medium, Rate_High (based on mean bins)
- Time Period: Period_2000-2005, Period_2006-2010, Period_2011-2013
Total transactions: 63,742
Unique items: 31

### Algorithm Settings
- min_support = 0.05 → rules must appear in ≥5% of transactions
- min_confidence = 0.6 → rules must be ≥60% reliable

### Key Results
- 105 frequent itemsets found
- 6 strong association rules met confidence threshold

**Top Association Rules (by Lift)**
IF Rate_Medium → THEN Tuberculosis  
  Support: 0.199 | Confidence: 0.619 | Lift: 1.14

IF Sex_Males, Rate_Medium → THEN Tuberculosis  
  Support: 0.073 | Confidence: 0.633 | Lift: 1.16

IF Period_2000-2005, Rate_Medium → THEN Tuberculosis  
  Support: 0.087 | Confidence: 0.630 | Lift: 1.16
> **Interpretation**: Tuberculosis is strongly associated with medium mortality rates, particularly among males and in the early 2000s. No strong rules linked HIV/AIDS to specific demographic or temporal patterns, suggesting its burden is more uniformly distributed or extreme (high/low only).

### Visualization
- Scatter plots of **Support vs. Confidence** and **Support vs. Lift**
- Color gradients show rule strength and reliability

## Key Insights Summary
1. **Epidemiological Clusters Exist**: Countries naturally group into distinct burden profiles — especially a high dual-burden cluster in Sub-Saharan Africa.
2. **HIV Drives High Burden Classification**: In this dataset, HIV/AIDS mortality is the sole predictor of whether a location is classified as "high burden."
3. **TB Shows Endemic Patterns**: Tuberculosis consistently appears in medium-rate contexts across genders and time periods, indicating persistent transmission even without epidemic-level spikes.
4. **Data Completeness Varies**: Record counts differ widely by country and disease, which was accounted for in feature engineering.



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
