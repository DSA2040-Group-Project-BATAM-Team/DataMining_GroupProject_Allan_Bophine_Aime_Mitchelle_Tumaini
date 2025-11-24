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
- **Standardization**: Cleaned text casing, ensured integer years

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

- **Environment**: Python 3.13  
- **Key Libraries**: `pandas`, `openpyxl`, `numpy`  
- **Dependencies Installed**:
  ```bash
  pip install openpyxl et-xmlfile
- Script: `1_extract_transform.ipynb`
- Workflow: Load → Filter → Clean → Enrich → Export


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
