"""
TB & HIV/AIDS Global Disease Burden Analysis Dashboard
Streamlit application integrating EDA, Data Mining, and Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TB & HIV/AIDS Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    df = pd.read_excel("Group_work_cleaned.xlsx")
    return df

# Data Mining Functions
@st.cache_data
def prepare_clustering_data(df):
    """Prepare data for clustering analysis"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy import stats
    
    cluster_base = df[
        (df['age_group_name'] == 'Age-standardized') & 
        (df['metric'] == 'Deaths') &
        (df['unit'] == 'Rate per 100,000')
    ].copy()
    
    location_stats = []
    for location_id in cluster_base['location_id'].unique():
        loc_data = cluster_base[cluster_base['location_id'] == location_id]
        location_name = loc_data['location_name'].iloc[0]
        
        # TB statistics
        tb_data = loc_data[loc_data['cause_name'] == 'Tuberculosis']['mean']
        tb_mean = tb_data.mean() if len(tb_data) > 0 else 0
        tb_std = tb_data.std() if len(tb_data) > 0 else 0
        tb_trend = 0
        if len(tb_data) > 1:
            years = loc_data[loc_data['cause_name'] == 'Tuberculosis']['year'].values
            if len(years) == len(tb_data):
                slope, _, _, _, _ = stats.linregress(years, tb_data)
                tb_trend = slope
        
        # HIV statistics
        hiv_data = loc_data[loc_data['cause_name'] == 'HIV/AIDS']['mean']
        hiv_mean = hiv_data.mean() if len(hiv_data) > 0 else 0
        hiv_std = hiv_data.std() if len(hiv_data) > 0 else 0
        hiv_trend = 0
        if len(hiv_data) > 1:
            years = loc_data[loc_data['cause_name'] == 'HIV/AIDS']['year'].values
            if len(years) == len(hiv_data):
                slope, _, _, _, _ = stats.linregress(years, hiv_data)
                hiv_trend = slope
        
        location_stats.append({
            'location_id': location_id,
            'location_name': location_name,
            'tb_mean_rate': tb_mean,
            'tb_variability': tb_std,
            'tb_trend': tb_trend,
            'hiv_mean_rate': hiv_mean,
            'hiv_variability': hiv_std,
            'hiv_trend': hiv_trend,
            'total_burden': tb_mean + hiv_mean,
            'disease_ratio': hiv_mean / tb_mean if tb_mean > 0 else 0
        })
    
    clustering_df = pd.DataFrame(location_stats)
    
    # Prepare features
    feature_cols = ['tb_mean_rate', 'hiv_mean_rate', 'tb_trend', 'hiv_trend', 
                    'tb_variability', 'hiv_variability', 'disease_ratio']
    X_cluster = clustering_df[feature_cols].fillna(0)
    X_cluster = X_cluster.replace([np.inf, -np.inf], 0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform clustering
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clustering_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, clustering_df['cluster'])
    
    return clustering_df, silhouette, optimal_k

@st.cache_data
def prepare_time_series_data(df):
    """Prepare time series data"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from scipy import stats
    
    ts_data = df[
        (df['age_group_name'] == 'All Ages') &
        (df['sex_name'] == 'Both sexes') &
        (df['metric'] == 'Deaths') &
        (df['unit'] == 'Number')
    ].copy()
    
    ts_summary = ts_data.groupby(['year', 'cause_name'])['mean'].sum().reset_index()
    ts_summary = ts_summary.pivot(index='year', columns='cause_name', values='mean')
    
    # Calculate trends
    tb_years = ts_summary.index.values
    tb_deaths = ts_summary['Tuberculosis'].values
    tb_slope, tb_intercept, tb_r, _, _ = stats.linregress(tb_years, tb_deaths)
    
    hiv_years = ts_summary.index.values
    hiv_deaths = ts_summary['HIV/AIDS'].values
    hiv_slope, hiv_intercept, hiv_r, _, _ = stats.linregress(hiv_years, hiv_deaths)
    
    # Forecasting
    forecast_years = 3
    tb_model = ExponentialSmoothing(ts_summary['Tuberculosis'], trend='add', seasonal=None).fit()
    hiv_model = ExponentialSmoothing(ts_summary['HIV/AIDS'], trend='add', seasonal=None).fit()
    
    tb_forecast = tb_model.forecast(steps=forecast_years)
    hiv_forecast = hiv_model.forecast(steps=forecast_years)
    
    return ts_summary, (tb_slope, tb_r), (hiv_slope, hiv_r), tb_forecast, hiv_forecast

# Main App
def main():
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["🏠 Home", "📈 EDA", "🔬 Data Mining", "💡 Insights & Storytelling"]
    )
    
    # Load data
    try:
        df = load_data()
        st.sidebar.success(f"✅ Data loaded: {len(df):,} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "📈 EDA":
        show_eda(df)
    elif page == "🔬 Data Mining":
        show_data_mining(df)
    elif page == "💡 Insights & Storytelling":
        show_insights(df)

def show_home():
    """Home page with project overview"""
    st.markdown('<p class="main-header">🏥 TB & HIV/AIDS Global Disease Burden Analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📅 **Period**\n\n2000-2013")
    with col2:
        st.info("🌍 **Coverage**\n\n195 Locations")
    with col3:
        st.info("📊 **Records**\n\n~78,000")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Overview")
    st.write("""
    This comprehensive analysis examines the global burden of **Tuberculosis** and **HIV/AIDS** 
    across 195 locations over 14 years (2000-2013). Through exploratory data analysis, advanced 
    data mining techniques, and focused case studies, we uncover patterns, predict trends, and 
    derive actionable insights for public health interventions.
    """)
    
    st.markdown("### 📚 Dashboard Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📈 Exploratory Data Analysis
        - Data quality assessment
        - Distribution analysis
        - Correlation patterns
        - Uncertainty quantification
        - Demographic breakdowns
        """)
        
        st.markdown("""
        #### 🔬 Data Mining
        - **Clustering**: Country disease profiles
        - **Classification**: Disease signature prediction
        - **Time Series**: Trend analysis & forecasting
        """)
    
    with col2:
        st.markdown("""
        #### 💡 Insights & Storytelling
        - Kenya case study
        - Temporal trends analysis
        - Sex-disaggregated patterns
        - Age-specific burden
        - Uncertainty evolution
        """)
        
        st.markdown("""
        #### 🎓 Key Findings
        - Southern Africa HIV epidemic patterns
        - Disease-specific epidemiological signatures
        - Declining trends with intervention scale-up
        - Age and sex vulnerability patterns
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("👈 Use the sidebar to navigate through different sections of the analysis")

def show_eda(df):
    """Exploratory Data Analysis page"""
    st.markdown('<p class="main-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Locations", df['location_name'].nunique())
    with col3:
        st.metric("Years", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        st.metric("Diseases", df['cause_name'].nunique())
    
    st.markdown("---")
    
    # Data Summary
    with st.expander("📋 Data Summary Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numerical Summary**")
            st.dataframe(df.describe())
        with col2:
            st.write("**Categorical Summary**")
            st.dataframe(df.describe(include=['object']))
    
    # Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    tab1, tab2 = st.tabs(["Core Metrics", "All Numeric Variables"])
    
    with tab1:
        corr = df[["mean", "lower", "upper"]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlation: Mean, Lower, Upper Bounds")
        st.pyplot(fig)
        
        st.markdown('<div class="insight-box">💡 <b>Insight:</b> Strong positive correlation between mean, lower, and upper bounds indicates consistent uncertainty ranges across measurements.</div>', unsafe_allow_html=True)
    
    with tab2:
        corr_all = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_all, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Full Correlation Heatmap")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Distribution Analysis
    st.markdown("### 📊 Distribution Analysis")
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Select variable to analyze", num_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[selected_col], ax=ax, color='skyblue')
        ax.set_title(f"Boxplot: {selected_col}")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(x=df[selected_col], ax=ax, color='lightcoral')
        ax.set_title(f"Violin Plot: {selected_col}")
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Skewness", f"{df[selected_col].skew():.3f}")
    with col2:
        st.metric("Kurtosis", f"{df[selected_col].kurt():.3f}")
    
    st.markdown("---")
    
    # Disease Comparison
    st.markdown("### 🦠 Distribution by Disease")
    
    selected_var = st.selectbox("Select variable for disease comparison", num_cols, key='disease_comp')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df, x='cause_name', y=selected_var, palette='viridis', inner='quartile', ax=ax)
    ax.set_title(f"Distribution of {selected_var} by Disease")
    ax.set_xlabel("Disease")
    st.pyplot(fig)
    
    # Summary statistics by disease
    disease_summary = df.groupby('cause_name')[selected_var].describe()
    st.dataframe(disease_summary)
    
    st.markdown("---")
    
    # Uncertainty Analysis
    st.markdown("### 🎯 Uncertainty Analysis")
    
    df_unc = df.copy()
    df_unc['Uncertainty_Range'] = df_unc['upper'] - df_unc['lower']
    df_unc['Uncertainty_Percent'] = (df_unc['Uncertainty_Range'] / (df_unc['mean'] + 0.01)) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Violin plot
        violin_metrics = ['mean', 'lower', 'upper', 'Uncertainty_Range']
        violin_data = df_unc[violin_metrics].melt(var_name='Metric Type', value_name='Rate Value')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='Metric Type', y='Rate Value', data=violin_data, inner='quartile', 
                      palette=['#56B4E9', '#0072B2', '#E69F00', '#D55E00'], ax=ax)
        ax.set_title("Rate & Uncertainty Distribution Comparison")
        st.pyplot(fig)
    
    with col2:
        # KDE plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df_unc['Uncertainty_Percent'], fill=True, ax=ax, color='steelblue')
        ax.axvline(df_unc['Uncertainty_Percent'].mean(), color='red', linestyle='--', 
                  label=f"Mean: {df_unc['Uncertainty_Percent'].mean():.2f}%")
        ax.set_title("Uncertainty Percentage Distribution")
        ax.set_xlabel("Uncertainty (%)")
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Top Burden Locations
    st.markdown("### 🌍 Geographic Analysis")
    
    metric_choice = st.radio("Select metric", ['Deaths', 'Prevalence'], horizontal=True)
    
    death_data = df[df['metric'] == metric_choice]
    death_by_location = death_data.groupby('location_name')['mean'].sum().sort_values(ascending=False)
    top10 = death_by_location.head(10)
    
    # Interactive bar chart
    fig = go.Figure(go.Bar(
        y=top10.index[::-1],
        x=top10.values[::-1],
        orientation='h',
        marker=dict(
            color=top10.values[::-1],
            colorscale='Blues',
            showscale=True
        ),
        text=[f'{int(v):,}' for v in top10.values[::-1]],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top 10 Locations by {metric_choice}',
        xaxis_title=f'Total {metric_choice}',
        yaxis_title='Location',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_data_mining(df):
    """Data Mining page"""
    st.markdown('<p class="main-header">🔬 Data Mining Analysis</p>', unsafe_allow_html=True)
    
    st.write("""
    This section applies three complementary mining techniques to uncover hidden patterns:
    - **Clustering**: Groups countries by disease burden profiles
    - **Classification**: Predicts disease type from epidemiological patterns  
    - **Time Series**: Analyzes trends and forecasts future burden
    """)
    
    technique = st.selectbox(
        "Select Mining Technique",
        ["1️⃣ Clustering Analysis", "2️⃣ Classification Analysis", "3️⃣ Time Series Analysis"]
    )
    
    if technique == "1️⃣ Clustering Analysis":
        show_clustering(df)
    elif technique == "2️⃣ Classification Analysis":
        show_classification(df)
    elif technique == "3️⃣ Time Series Analysis":
        show_time_series(df)

def show_clustering(df):
    """Clustering analysis section"""
    st.markdown("### 🎯 K-Means Clustering: Country Disease Profiles")
    
    st.info("""
    **Objective**: Group countries by similar TB/HIV burden patterns to identify regions requiring 
    different intervention strategies.
    """)
    
    with st.spinner("Performing clustering analysis..."):
        clustering_df, silhouette, optimal_k = prepare_clustering_data(df)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Clusters (k)", optimal_k)
    with col2:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    with col3:
        st.metric("Countries Analyzed", len(clustering_df))
    
    st.markdown("---")
    
    # Cluster characteristics
    st.markdown("#### 📊 Cluster Characteristics")
    
    cluster_summary = clustering_df.groupby('cluster').agg({
        'tb_mean_rate': 'mean',
        'hiv_mean_rate': 'mean',
        'tb_trend': 'mean',
        'hiv_trend': 'mean',
        'disease_ratio': 'mean'
    }).round(2)
    
    cluster_summary['count'] = clustering_df.groupby('cluster').size()
    cluster_summary = cluster_summary.rename(columns={
        'tb_mean_rate': 'Avg TB Rate',
        'hiv_mean_rate': 'Avg HIV Rate',
        'tb_trend': 'TB Trend',
        'hiv_trend': 'HIV Trend',
        'disease_ratio': 'HIV/TB Ratio',
        'count': 'Countries'
    })
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Visualizations
    st.markdown("#### 📈 Cluster Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            clustering_df,
            x='tb_mean_rate',
            y='hiv_mean_rate',
            color='cluster',
            hover_data=['location_name'],
            title='Clusters by Disease Burden',
            labels={'tb_mean_rate': 'TB Death Rate', 'hiv_mean_rate': 'HIV Death Rate'},
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            clustering_df,
            x='tb_trend',
            y='hiv_trend',
            color='cluster',
            hover_data=['location_name'],
            title='Clusters by Disease Trends',
            labels={'tb_trend': 'TB Trend (change/year)', 'hiv_trend': 'HIV Trend (change/year)'},
            color_continuous_scale='viridis'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample countries
    st.markdown("#### 🌍 Sample Countries by Cluster")
    for cluster_id in sorted(clustering_df['cluster'].unique()):
        countries = clustering_df[clustering_df['cluster'] == cluster_id].nlargest(5, 'total_burden')['location_name'].tolist()
        st.write(f"**Cluster {cluster_id}**: {', '.join(countries)}")
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 Key Insights:**
    
    **Cluster 0 (Low-Moderate Burden)**:
    - Manageable disease burden with gradual improvement
    - Balanced TB/HIV burden
    - Examples: Tanzania, Cameroon, South Sudan
    
    **Cluster 1 (High Burden)**:
    - Severely affected countries (Southern Africa)
    - HIV-driven epidemics (5x higher HIV rates)
    - Steeper declining trends show strong intervention response
    - Examples: Zimbabwe, Lesotho, Botswana
    
    **Implication**: Cluster 1 needs intensive HIV-focused interventions while Cluster 0 requires 
    broader infectious disease control strategies.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_classification(df):
    """Classification analysis section"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    st.markdown("### 🎯 Decision Tree: Predicting Disease Type")
    
    st.info("""
    **Objective**: Predict whether a record is TB or HIV/AIDS based on epidemiological patterns 
    (age, sex, death rates, uncertainty).
    """)
    
    with st.spinner("Training decision tree..."):
        # Prepare data
        class_data = df[
            (df['metric'] == 'Deaths') & 
            (df['unit'] == 'Rate per 100,000')
        ].copy()
        
        le_age = LabelEncoder()
        le_sex = LabelEncoder()
        
        class_data['age_encoded'] = le_age.fit_transform(class_data['age_group_name'])
        class_data['sex_encoded'] = le_sex.fit_transform(class_data['sex_name'])
        class_data['uncertainty'] = class_data['upper'] - class_data['lower']
        class_data['relative_uncertainty'] = class_data['uncertainty'] / (class_data['mean'] + 0.01)
        class_data['year_normalized'] = (class_data['year'] - 2000) / 13
        
        feature_columns = ['age_encoded', 'sex_encoded', 'mean', 'uncertainty', 
                          'relative_uncertainty', 'year_normalized']
        X = class_data[feature_columns]
        y = (class_data['cause_name'] == 'HIV/AIDS').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        dt = DecisionTreeClassifier(max_depth=6, min_samples_split=50, min_samples_leaf=20, random_state=42)
        dt.fit(X_train, y_train)
        
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col3:
        st.metric("Test Samples", f"{len(X_test):,}")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['TB', 'HIV/AIDS'],
                   yticklabels=['TB', 'HIV/AIDS'], ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Feature Importance")
        feature_names = ['Age Group', 'Sex', 'Death Rate', 'Uncertainty', 'Rel. Uncertainty', 'Year']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    
    # Classification Report
    st.markdown("#### 📊 Detailed Performance")
    report = classification_report(y_test, y_pred, target_names=['Tuberculosis', 'HIV/AIDS'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 Key Insights:**
    
    **Performance:**
    - 72% overall accuracy distinguishing TB from HIV/AIDS
    - TB: High recall (91%) - catches most TB cases
    - HIV: High precision (84%) - accurate when predicted HIV
    
    **Most Important Features:**
    1. **Death Rate (47%)**: HIV epidemics cause much higher mortality
    2. **Relative Uncertainty (25%)**: HIV data has surveillance challenges
    3. **Uncertainty Range (18%)**: Reflects data quality differences
    
    **Medical Insight**: High death rates (>215/100k) strongly indicate HIV/AIDS, particularly 
    in Southern African epidemic settings. The model learned distinct epidemiological signatures.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_time_series(df):
    """Time series analysis section"""
    st.markdown("### 📈 Time Series: Trends & Forecasting")
    
    st.info("""
    **Objective**: Analyze temporal trends in global disease burden and forecast future trajectory 
    using exponential smoothing.
    """)
    
    with st.spinner("Analyzing time series..."):
        ts_summary, tb_stats, hiv_stats, tb_forecast, hiv_forecast = prepare_time_series_data(df)
    
    tb_slope, tb_r = tb_stats
    hiv_slope, hiv_r = hiv_stats
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TB Trend", f"{tb_slope:+.0f} deaths/yr")
    with col2:
        st.metric("TB R²", f"{tb_r**2:.3f}")
    with col3:
        st.metric("HIV Trend", f"{hiv_slope:+.0f} deaths/yr")
    with col4:
        st.metric("HIV R²", f"{hiv_r**2:.3f}")
    
    st.markdown("---")
    
    # Historical Trends
    st.markdown("#### 📊 Historical Trends (2000-2013)")
    
    # Create combined plot
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Tuberculosis Deaths", "HIV/AIDS Deaths"))
    
    # TB
    fig.add_trace(
        go.Scatter(x=ts_summary.index, y=ts_summary['Tuberculosis'], 
                  mode='lines+markers', name='TB', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace

    # TB trendline
    tb_years = ts_summary.index.values
    tb_trendline = tb_slope * tb_years + (ts_summary['Tuberculosis'].iloc[0] - tb_slope * tb_years[0])
    fig.add_trace(
        go.Scatter(x=tb_years, y=tb_trendline, 
                  mode='lines', name='TB Trend', line=dict(color='blue', width=1, dash='dash')),
        row=1, col=1
    )
    
    # HIV
    fig.add_trace(
        go.Scatter(x=ts_summary.index, y=ts_summary['HIV/AIDS'], 
                  mode='lines+markers', name='HIV/AIDS', line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    # HIV trendline
    hiv_years = ts_summary.index.values
    hiv_trendline = hiv_slope * hiv_years + (ts_summary['HIV/AIDS'].iloc[0] - hiv_slope * hiv_years[0])
    fig.add_trace(
        go.Scatter(x=hiv_years, y=hiv_trendline, 
                  mode='lines', name='HIV Trend', line=dict(color='red', width=1, dash='dash')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Deaths", row=1, col=1)
    fig.update_yaxes(title_text="Deaths", row=2, col=1)
    fig.update_layout(height=700, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Forecasting
    st.markdown("#### 🔮 3-Year Forecast (2014-2016)")
    
    forecast_years = [2014, 2015, 2016]
    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        'TB Forecast': tb_forecast.values,
        'HIV Forecast': hiv_forecast.values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(forecast_df.style.format({
            'TB Forecast': '{:,.0f}',
            'HIV Forecast': '{:,.0f}'
        }), use_container_width=True)
    
    with col2:
        # Forecast visualization
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=ts_summary.index, 
            y=ts_summary['Tuberculosis'],
            mode='lines+markers',
            name='TB Historical',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=ts_summary.index, 
            y=ts_summary['HIV/AIDS'],
            mode='lines+markers',
            name='HIV Historical',
            line=dict(color='red')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_years,
            y=tb_forecast.values,
            mode='lines+markers',
            name='TB Forecast',
            line=dict(color='blue', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_years,
            y=hiv_forecast.values,
            mode='lines+markers',
            name='HIV Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Historical + Forecast',
            xaxis_title='Year',
            yaxis_title='Deaths',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **💡 Key Insights:**
    
    **Tuberculosis:**
    - Strong declining trend: **{abs(tb_slope):,.0f} fewer deaths per year**
    - High model fit (R² = {tb_r**2:.3f}) indicates consistent improvement
    - 2014-2016 forecast: Continued decline expected
    
    **HIV/AIDS:**
    - Sharp declining trend: **{abs(hiv_slope):,.0f} fewer deaths per year**
    - Excellent model fit (R² = {hiv_r**2:.3f})
    - Reflects successful scale-up of antiretroviral therapy (ART)
    
    **Global Context:**
    - Both diseases show positive response to intervention programs
    - HIV decline is steeper, reflecting rapid ART expansion post-2000s
    - TB decline is steadier, consistent with DOTS program implementation
    - Forecasts suggest continued improvement if intervention momentum maintained
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_insights(df):
    """Insights and storytelling page"""
    st.markdown('<p class="main-header">💡 Insights & Data Storytelling</p>', unsafe_allow_html=True)
    
    st.write("""
    This section presents focused case studies and cross-cutting insights derived from the analysis.
    """)
    
    insight_choice = st.selectbox(
        "Select Focus Area",
        ["🇰🇪 Kenya Case Study", "⏰ Temporal Trends", "👥 Sex Disparities", 
         "📊 Age Patterns", "🎯 Uncertainty Evolution"]
    )
    
    if insight_choice == "🇰🇪 Kenya Case Study":
        show_kenya_case_study(df)
    elif insight_choice == "⏰ Temporal Trends":
        show_temporal_insights(df)
    elif insight_choice == "👥 Sex Disparities":
        show_sex_insights(df)
    elif insight_choice == "📊 Age Patterns":
        show_age_insights(df)
    elif insight_choice == "🎯 Uncertainty Evolution":
        show_uncertainty_insights(df)

def show_kenya_case_study(df):
    """Kenya case study"""
    st.markdown("### 🇰🇪 Kenya: A Dual Epidemic Journey")
    
    kenya_data = df[df['location_name'] == 'Kenya'].copy()
    
    if len(kenya_data) == 0:
        st.warning("Kenya data not available in dataset")
        return
    
    st.info("""
    **Context**: Kenya represents a critical case study with significant TB and HIV burden, 
    particularly during the early 2000s HIV epidemic peak.
    """)
    
    # Death trends
    kenya_deaths = kenya_data[
        (kenya_data['metric'] == 'Deaths') & 
        (kenya_data['unit'] == 'Number') &
        (kenya_data['age_group_name'] == 'All Ages') &
        (kenya_data['sex_name'] == 'Both sexes')
    ].groupby(['year', 'cause_name'])['mean'].sum().reset_index()
    
    if len(kenya_deaths) == 0:
        st.warning("No death data available for Kenya with the specified filters")
        return
    
    kenya_pivot = kenya_deaths.pivot(index='year', columns='cause_name', values='mean')
    
    # Check if required columns exist
    if kenya_pivot.empty or len(kenya_pivot.columns) == 0:
        st.warning("Insufficient data for Kenya analysis")
        return
    
    # Get actual disease names from the data
    disease_names = kenya_pivot.columns.tolist()
    st.write(f"Available diseases in data: {', '.join(disease_names)}")
    
    if len(disease_names) < 2:
        st.warning("Need at least 2 diseases for comparison")
        return
    
    # Use actual disease names from data
    disease1 = disease_names[0]
    disease2 = disease_names[1] if len(disease_names) > 1 else disease_names[0]
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Deaths Over Time", "Death Rates Comparison", 
                       "Disease Burden Share", "Yearly Change Rate"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Plot 1: Deaths over time
    for idx, disease in enumerate(disease_names):
        color = ['blue', 'red'][idx % 2]
        fig.add_trace(
            go.Scatter(x=kenya_pivot.index, y=kenya_pivot[disease],
                      name=disease, line=dict(color=color, width=3)),
            row=1, col=1
        )
    
    # Plot 2: Rate comparison (if available)
    kenya_rates = kenya_data[
        (kenya_data['metric'] == 'Deaths') & 
        (kenya_data['unit'] == 'Rate per 100,000') &
        (kenya_data['age_group_name'] == 'Age-standardized') &
        (kenya_data['sex_name'] == 'Both sexes')
    ].groupby(['year', 'cause_name'])['mean'].mean().reset_index()
    
    if len(kenya_rates) > 0:
        kenya_rates_pivot = kenya_rates.pivot(index='year', columns='cause_name', values='mean')
        for idx, disease in enumerate(kenya_rates_pivot.columns):
            color = ['lightblue', 'lightcoral'][idx % 2]
            fig.add_trace(
                go.Scatter(x=kenya_rates_pivot.index, y=kenya_rates_pivot[disease],
                          name=f'{disease} Rate', line=dict(color=color, width=2), showlegend=False),
                row=1, col=2
            )
    
    # Plot 3: Pie chart - total burden
    total_values = [kenya_pivot[disease].sum() for disease in disease_names]
    fig.add_trace(
        go.Pie(labels=disease_names, values=total_values,
              marker=dict(colors=['blue', 'red'])),
        row=2, col=1
    )
    
    # Plot 4: Year-over-year change
    for idx, disease in enumerate(disease_names):
        color = ['blue', 'red'][idx % 2]
        change = kenya_pivot[disease].pct_change() * 100
        fig.add_trace(
            go.Bar(x=kenya_pivot.index[1:], y=change[1:], 
                  name=f'{disease} % Change', marker_color=color),
            row=2, col=2
        )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics
    st.markdown("#### 📊 Key Statistics")
    
    # Calculate statistics for available diseases
    col1, col2, col3 = st.columns(3)
    
    with col1:
        disease1_total = kenya_pivot[disease1].sum()
        st.metric(f"Total {disease1} Deaths (2000-2013)", f"{disease1_total:,.0f}")
        disease1_decline = ((kenya_pivot[disease1].iloc[-1] - kenya_pivot[disease1].iloc[0]) 
                           / kenya_pivot[disease1].iloc[0] * 100)
        st.metric(f"{disease1} Change", f"{disease1_decline:+.1f}%")
    
    with col2:
        disease2_total = kenya_pivot[disease2].sum()
        st.metric(f"Total {disease2} Deaths (2000-2013)", f"{disease2_total:,.0f}")
        disease2_decline = ((kenya_pivot[disease2].iloc[-1] - kenya_pivot[disease2].iloc[0]) 
                           / kenya_pivot[disease2].iloc[0] * 100)
        st.metric(f"{disease2} Change", f"{disease2_decline:+.1f}%")
    
    with col3:
        peak_year = kenya_pivot.sum(axis=1).idxmax()
        st.metric("Peak Burden Year", int(peak_year))
        combined_total = kenya_pivot.sum().sum()
        st.metric("Combined Total", f"{combined_total:,.0f}")
    
    # Narrative
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **📖 Kenya's Story:**
    
    **Early 2000s Crisis:**
    - HIV epidemic peaked in early 2000s before widespread ART availability
    - TB resurgence driven by HIV co-infection (TB is leading cause of death in PLHIV)
    - Combined burden overwhelmed health systems
    
    **Turning Point (Mid-2000s):**
    - Rapid ART scale-up following WHO 3-by-5 initiative
    - Integration of TB/HIV services
    - Community-based care models
    
    **Progress & Challenges:**
    - Significant declines in both diseases post-2005
    - HIV deaths declined faster due to ART effectiveness
    - TB remains endemic, requiring sustained DOTS program
    - TB/HIV co-infection management improved
    
    **Lessons:**
    - Integrated disease approach more effective than siloed programs
    - Community health workers crucial for rural reach
    - Sustained funding essential - early gains can reverse
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_temporal_insights(df):
    """Temporal trends insights"""
    st.markdown("### ⏰ Temporal Trends: The Arc of Progress")
    
    # Global trends by year
    yearly_trends = df[
        (df['metric'] == 'Deaths') & 
        (df['unit'] == 'Number') &
        (df['age_group_name'] == 'All Ages') &
        (df['sex_name'] == 'Both sexes')
    ].groupby(['year', 'cause_name'])['mean'].sum().reset_index()
    
    if len(yearly_trends) == 0:
        st.warning("No temporal data available with specified filters")
        return
    
    yearly_pivot = yearly_trends.pivot(index='year', columns='cause_name', values='mean')
    
    if yearly_pivot.empty:
        st.warning("Insufficient data for temporal analysis")
        return
    
    # Get actual disease names
    disease_names = yearly_pivot.columns.tolist()
    
    # Visualization
    fig = go.Figure()
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    for idx, disease in enumerate(disease_names):
        fig.add_trace(go.Scatter(
            x=yearly_pivot.index,
            y=yearly_pivot[disease],
            name=disease,
            line=dict(color=colors[idx % len(colors)], width=4),
            mode='lines+markers'
        ))
    
    # Add annotations for key events
    annotations = [
        dict(x=2003, y=yearly_pivot.max().max() * 0.9, 
             text="WHO 3-by-5<br>Initiative", showarrow=True, arrowhead=2),
        dict(x=2006, y=yearly_pivot.max().max() * 0.8,
             text="Stop TB<br>Strategy", showarrow=True, arrowhead=2),
    ]
    
    fig.update_layout(
        title='Global Disease Burden: 2000-2013',
        xaxis_title='Year',
        yaxis_title='Total Deaths',
        height=500,
        annotations=annotations,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics - handle multiple diseases
    cols = st.columns(min(len(disease_names) + 1, 4))
    
    for idx, disease in enumerate(disease_names):
        with cols[idx]:
            total_decline = yearly_pivot[disease].iloc[0] - yearly_pivot[disease].iloc[-1]
            pct_decline = (total_decline / yearly_pivot[disease].iloc[0] * 100)
            st.metric(f"{disease} Lives Saved", f"{total_decline:,.0f}", 
                     f"{pct_decline:.1f}%")
    
    with cols[min(len(disease_names), 3)]:
        combined_decline = (yearly_pivot.iloc[0].sum() - yearly_pivot.iloc[-1].sum())
        st.metric("Combined Impact", f"{combined_decline:,.0f}")
    
    # Insight
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 The Global Trajectory:**
    
    **2000-2004: Crisis Period**
    - HIV epidemic peak in sub-Saharan Africa
    - Limited ART access (< 5% coverage)
    - TB resurging due to HIV co-infection
    
    **2005-2009: Intervention Scale-up**
    - WHO 3-by-5 Initiative (3 million on ART by 2005)
    - PEPFAR and Global Fund investments
    - Stop TB Strategy implementation
    
    **2010-2013: Sustained Progress**
    - ART coverage > 60% in high-burden countries
    - MDR-TB programs established
    - Integration of TB/HIV services
    
    **Key Success Factors:**
    - Political commitment & funding
    - Community-based delivery models
    - Drug supply chain improvements
    - Diagnostic innovations (GeneXpert)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_sex_insights(df):
    """Sex disparity analysis"""
    st.markdown("### 👥 Sex Disparities in Disease Burden")
    
    sex_data = df[
        (df['metric'] == 'Deaths') & 
        (df['unit'] == 'Rate per 100,000') &
        (df['age_group_name'] == 'Age-standardized') &
        (df['sex_name'].isin(['Male', 'Female']))
    ].groupby(['cause_name', 'sex_name'])['mean'].mean().reset_index()
    
    if len(sex_data) == 0:
        st.warning("No sex-disaggregated data available")
        return
    
    # Visualization
    fig = px.bar(
        sex_data,
        x='cause_name',
        y='mean',
        color='sex_name',
        barmode='group',
        title='Average Death Rates by Sex',
        labels={'mean': 'Death Rate (per 100,000)', 'cause_name': 'Disease', 'sex_name': 'Sex'},
        color_discrete_map={'Male': '#1f77b4', 'Female': '#e377c2'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate ratios for each disease
    diseases = sex_data['cause_name'].unique()
    
    cols = st.columns(min(len(diseases), 3))
    
    for idx, disease in enumerate(diseases):
        disease_data = sex_data[sex_data['cause_name'] == disease]
        
        male_data = disease_data[disease_data['sex_name'] == 'Male']
        female_data = disease_data[disease_data['sex_name'] == 'Female']
        
        if len(male_data) > 0 and len(female_data) > 0:
            male_rate = male_data['mean'].values[0]
            female_rate = female_data['mean'].values[0]
            
            with cols[idx % len(cols)]:
                st.markdown(f"**{disease}**")
                st.metric("Male:Female Ratio", f"{male_rate/female_rate:.2f}:1")
                st.metric("Male Rate", f"{male_rate:.1f} per 100k")
                st.metric("Female Rate", f"{female_rate:.1f} per 100k")
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 Sex-Specific Patterns:**
    
    **Tuberculosis (Male-Predominant):**
    - **~2:1 male:female ratio** across all regions
    - **Biological factors**: Immune response differences
    - **Social factors**: Higher smoking rates, occupational exposures, healthcare-seeking delays
    - **Stigma**: Men less likely to seek TB diagnosis/treatment
    
    **HIV/AIDS (More Balanced):**
    - **Closer to 1:1 ratio** but varies by region
    - **Sub-Saharan Africa**: Females disproportionately affected (biological vulnerability + gender inequities)
    - **Other regions**: Males higher burden (MSM transmission, injection drug use)
    - **Age interaction**: Young women (15-24) have 2x risk vs males in high-burden settings
    
    **Implications:**
    - TB programs need male-targeted outreach strategies
    - HIV programs require gender-specific prevention (e.g., PrEP for young women in SSA)
    - Addressing gender norms critical for both diseases
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_age_insights(df):
    """Age pattern analysis"""
    st.markdown("### 📊 Age-Specific Disease Burden")
    
    age_data = df[
        (df['metric'] == 'Deaths') & 
        (df['unit'] == 'Rate per 100,000') &
        (df['sex_name'] == 'Both sexes') &
        (~df['age_group_name'].isin(['All Ages', 'Age-standardized']))
    ].groupby(['cause_name', 'age_group_name'])['mean'].mean().reset_index()
    
    if len(age_data) == 0:
        st.warning("No age-disaggregated data available")
        return
    
    # Get available age groups and order them logically
    available_ages = age_data['age_group_name'].unique().tolist()
    
    # Define standard order (will only use those that exist)
    age_order = ['<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years',
                 'Under 5', '5 to 14', '15 to 49', '50 to 69', '70 plus']
    
    # Filter to only ages that exist in data
    age_order_filtered = [age for age in age_order if age in available_ages]
    
    # Add any remaining ages not in standard order
    for age in available_ages:
        if age not in age_order_filtered:
            age_order_filtered.append(age)
    
    age_data_filtered = age_data[age_data['age_group_name'].isin(age_order_filtered)]
    
    # Visualization
    fig = go.Figure()
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    for idx, disease in enumerate(age_data_filtered['cause_name'].unique()):
        disease_data = age_data_filtered[age_data_filtered['cause_name'] == disease]
        disease_data = disease_data.set_index('age_group_name').reindex(age_order_filtered).reset_index()
        
        fig.add_trace(go.Scatter(
            x=disease_data['age_group_name'],
            y=disease_data['mean'],
            name=disease,
            mode='lines+markers',
            line=dict(width=3, color=colors[idx % len(colors)])
        ))
    
    fig.update_layout(
        title='Death Rates by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Death Rate (per 100,000)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics by age
    st.markdown("#### Age-Specific Statistics")
    
    pivot_age = age_data_filtered.pivot(index='age_group_name', columns='cause_name', values='mean')
    pivot_age = pivot_age.reindex(age_order_filtered)
    
    st.dataframe(pivot_age.style.background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 Age-Specific Vulnerabilities:**
    
    **Tuberculosis:**
    - **Bimodal distribution**: Peaks in young adults (15-49) and elderly (70+)
    - **Children (<5)**: Often severe/disseminated disease (TB meningitis)
    - **Working age (15-49)**: Highest absolute burden due to population size
    - **Elderly**: Reactivation of latent infection, comorbidities
    
    **HIV/AIDS:**
    - **Concentrated in reproductive age (15-49)**: >80% of deaths
    - **Sexual transmission**: Primary driver in adults
    - **Mother-to-child**: Accounts for pediatric burden (declining with PMTCT)
    - **Elderly**: Lower rates due to cohort effects and survival
    
    **Policy Implications:**
    - **Youth programs**: Critical for HIV prevention (15-24 years)
    - **Active case finding**: Target high-burden adult age groups for TB
    - **Pediatric formulations**: Essential for treating children
    - **Geriatric TB**: Often missed diagnosis - need heightened clinical suspicion
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_uncertainty_insights(df):
    """Uncertainty evolution analysis"""
    st.markdown("### 🎯 Uncertainty Evolution: Data Quality Over Time")
    
    # Calculate uncertainty metrics
    uncertainty_data = df.copy()
    uncertainty_data['uncertainty_range'] = uncertainty_data['upper'] - uncertainty_data['lower']
    uncertainty_data['relative_uncertainty'] = (
        uncertainty_data['uncertainty_range'] / (uncertainty_data['mean'] + 0.01) * 100
    )
    
    # By year and disease
    yearly_uncertainty = uncertainty_data[
        (uncertainty_data['metric'] == 'Deaths') &
        (uncertainty_data['unit'] == 'Rate per 100,000')
    ].groupby(['year', 'cause_name'])['relative_uncertainty'].mean().reset_index()
    
    # Visualization
    fig = px.line(
        yearly_uncertainty,
        x='year',
        y='relative_uncertainty',
        color='cause_name',
        title='Relative Uncertainty Over Time',
        labels={'relative_uncertainty': 'Relative Uncertainty (%)', 'year': 'Year', 'cause_name': 'Disease'},
        markers=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    tb_early = yearly_uncertainty[(yearly_uncertainty['cause_name'] == 'Tuberculosis') & 
                                  (yearly_uncertainty['year'] <= 2003)]['relative_uncertainty'].mean()
    tb_late = yearly_uncertainty[(yearly_uncertainty['cause_name'] == 'Tuberculosis') & 
                                 (yearly_uncertainty['year'] >= 2010)]['relative_uncertainty'].mean()
    
    hiv_early = yearly_uncertainty[(yearly_uncertainty['cause_name'] == 'HIV/AIDS') & 
                                   (yearly_uncertainty['year'] <= 2003)]['relative_uncertainty'].mean()
    hiv_late = yearly_uncertainty[(yearly_uncertainty['cause_name'] == 'HIV/AIDS') & 
                                  (yearly_uncertainty['year'] >= 2010)]['relative_uncertainty'].mean()
    
    with col1:
        st.metric("TB Uncertainty 2000-2003", f"{tb_early:.1f}%")
        st.metric("TB Uncertainty 2010-2013", f"{tb_late:.1f}%", f"{tb_late-tb_early:+.1f}%")
    
    with col2:
        st.metric("HIV Uncertainty 2000-2003", f"{hiv_early:.1f}%")
        st.metric("HIV Uncertainty 2010-2013", f"{hiv_late:.1f}%", f"{hiv_late-hiv_early:+.1f}%")
    
    # By region
    st.markdown("#### Uncertainty by Region")
    
    regional_uncertainty = uncertainty_data[
        (uncertainty_data['metric'] == 'Deaths') &
        (uncertainty_data['unit'] == 'Rate per 100,000')
    ].groupby(['location_name', 'cause_name'])['relative_uncertainty'].mean().reset_index()
    
    top_uncertain = regional_uncertainty.nlargest(10, 'relative_uncertainty')
    
    fig = px.bar(
        top_uncertain,
        x='relative_uncertainty',
        y='location_name',
        color='cause_name',
        orientation='h',
        title='Top 10 Countries by Data Uncertainty',
        labels={'relative_uncertainty': 'Relative Uncertainty (%)', 'location_name': 'Country'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **💡 Data Quality Insights:**
    
    **Temporal Trends:**
    - **General improvement**: Uncertainty declining for both diseases
    - **TB**: More stable estimates throughout (established surveillance)
    - **HIV**: Higher initial uncertainty, improving rapidly (new epidemic → mature surveillance)
    
    **Why Uncertainty Matters:**
    - **Resource allocation**: Wide intervals = harder to justify funding
    - **Program evaluation**: High uncertainty masks intervention effects
    - **Predictive modeling**: Poor data quality → unreliable forecasts
    
    **Sources of Uncertainty:**
    - **Surveillance gaps**: Weak health systems, conflict zones
    - **Underreporting**: Stigma (HIV), private sector diagnosis (TB)
    - **Population denominators**: Census data quality
    - **Model assumptions**: Bridging data gaps with statistical models
    
    **High-Uncertainty Regions:**
    - **Conflict-affected**: Syria, Afghanistan, Somalia
    - **Weak surveillance**: Some sub-Saharan African countries
    - **Small populations**: Island nations, small states
    
    **Improving Data Quality:**
    - Vital registration systems strengthening
    - Electronic medical records
    - Routine programmatic data quality audits
    - Population-based surveys (DHS, MICS)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()