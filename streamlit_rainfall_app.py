

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from rainfall_forecast_module import RainfallForecaster
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Philippines Rainfall Predictor",
    page_icon="🌧️",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess all data (cached for performance)
    """
    with st.spinner("🔄 Loading and preprocessing data..."):
        # Load datasets
        df_cities = pd.read_csv('datasets/cities.csv')
        df_daily = pd.read_csv('datasets/daily_data_combined_2020_to_2023.csv')
        
        # Try to load hourly data for REAL humidity and pressure
        has_hourly_data = False
        try:
            st.info("📊 Loading hourly data (this takes 2-3 minutes the first time)...")
            df_hourly = pd.read_csv(
                'datasets/hourly_data_combined_2020_to_2023.csv',
                usecols=['city_name', 'datetime', 'relative_humidity_2m', 'pressure_msl']
            )
            
            # Process hourly data
            df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
            df_hourly['date'] = df_hourly['datetime'].dt.date
            
            # Aggregate to daily
            hourly_daily = df_hourly.groupby(['city_name', 'date']).agg({
                'relative_humidity_2m': 'mean',
                'pressure_msl': 'mean'
            }).reset_index()
            
            hourly_daily.rename(columns={
                'relative_humidity_2m': 'humidity',
                'pressure_msl': 'air_pressure'
            }, inplace=True)
            
            has_hourly_data = True
            st.success("✅ Loaded real humidity and pressure data!")
            
        except FileNotFoundError:
            st.warning("⚠️ Hourly data not found - using estimated values (lower accuracy)")
            hourly_daily = None
        except Exception as e:
            st.warning(f"⚠️ Could not load hourly data: {str(e)} - using estimated values")
            hourly_daily = None
        
        # Process daily data
        df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
        df_daily['date'] = df_daily['datetime'].dt.date
        df_daily['year'] = df_daily['datetime'].dt.year
        df_daily['month'] = df_daily['datetime'].dt.month
        
        # Merge hourly data if available
        if has_hourly_data and hourly_daily is not None:
            df_daily = df_daily.merge(
                hourly_daily[['city_name', 'date', 'humidity', 'air_pressure']],
                on=['city_name', 'date'],
                how='left'
            )
        
        # Fill missing with estimates if hourly data not available or incomplete
        if 'humidity' not in df_daily.columns or df_daily['humidity'].isna().any():
            if 'humidity' not in df_daily.columns:
                df_daily['humidity'] = 75.0  # Philippines average
            else:
                df_daily['humidity'] = df_daily['humidity'].fillna(75.0)
        
        if 'air_pressure' not in df_daily.columns or df_daily['air_pressure'].isna().any():
            if 'air_pressure' not in df_daily.columns:
                df_daily['air_pressure'] = 1011.0  # Tropical average
            else:
                df_daily['air_pressure'] = df_daily['air_pressure'].fillna(1011.0)
        
        # Aggregate to monthly
        agg_dict = {
            'temperature_2m_mean': 'mean',
            'precipitation_sum': 'sum',
        }
        
        # Add optional columns if they exist
        if 'humidity' in df_daily.columns:
            agg_dict['humidity'] = 'mean'
        if 'air_pressure' in df_daily.columns:
            agg_dict['air_pressure'] = 'mean'
        if 'rain_sum' in df_daily.columns:
            agg_dict['rain_sum'] = 'sum'
        if 'wind_speed_10m_max' in df_daily.columns:
            agg_dict['wind_speed_10m_max'] = 'mean'
        
        monthly_agg = df_daily.groupby(['city_name', 'year', 'month']).agg(agg_dict).reset_index()
        
        monthly_agg.rename(columns={
            'temperature_2m_mean': 'temperature',
            'precipitation_sum': 'monthly_rainfall'
        }, inplace=True)
        
        # Merge with city coordinates
        df_monthly = monthly_agg.merge(df_cities, on='city_name', how='inner')
        
        # Add ENSO indices
        oni_data = {
            2020: {1: 0.5, 2: 0.6, 3: 0.4, 4: 0.2, 5: -0.1, 6: -0.3, 
                   7: -0.4, 8: -0.6, 9: -0.9, 10: -1.2, 11: -1.3, 12: -1.3},
            2021: {1: -1.4, 2: -1.1, 3: -0.8, 4: -0.6, 5: -0.5, 6: -0.4,
                   7: -0.5, 8: -0.7, 9: -0.9, 10: -1.1, 11: -1.1, 12: -1.0},
            2022: {1: -0.9, 2: -0.8, 3: -0.6, 4: -0.4, 5: -0.2, 6: 0.1,
                   7: 0.3, 8: 0.5, 9: 0.8, 10: 1.0, 11: 1.1, 12: 1.0},
            2023: {1: -0.7, 2: -0.4, 3: -0.1, 4: 0.2, 5: 0.5, 6: 0.8,
                   7: 1.1, 8: 1.3, 9: 1.6, 10: 1.8, 11: 1.9, 12: 2.0}
        }
        
        def get_oni(row):
            year, month = int(row['year']), int(row['month'])
            if year in oni_data and month in oni_data[year]:
                return oni_data[year][month]
            return 0.0
        
        df_monthly['oni_index'] = df_monthly.apply(get_oni, axis=1)
        df_monthly['el_nino'] = (df_monthly['oni_index'] > 0.5).astype(int)
        df_monthly['la_nina'] = (df_monthly['oni_index'] < -0.5).astype(int)
        
        rows_before = len(df_monthly)
        
        # STRICT DATA QUALITY: Drop incomplete rows
        df_monthly = df_monthly.dropna()
        
        # Only keep cities with complete 48 months (4 years * 12 months)
        city_month_counts = df_monthly.groupby('city_name').size()
        complete_cities = city_month_counts[city_month_counts == 48].index
        df_monthly = df_monthly[df_monthly['city_name'].isin(complete_cities)]
        
        rows_after = len(df_monthly)
        cities_count = df_monthly['city_name'].nunique()
        
        # Store metadata
        df_monthly.attrs['has_hourly_data'] = has_hourly_data
        df_monthly.attrs['rows_dropped'] = rows_before - rows_after
        df_monthly.attrs['cities_count'] = cities_count
        
        return df_monthly


@st.cache_resource
def train_models(df_monthly):
    """
    Train all three models (cached to avoid retraining)
    - XGBoost: Best performance (RMSE: 69.96mm, R²: 0.78)
    - RBF: SVR with RBF kernel (RMSE: 101mm, R²: 0.55)
    - Polynomial: SVR with polynomial kernel (RMSE: 108mm, R²: 0.49)
    Uses cyclical time encoding for better seasonality capture
    """
    with st.spinner("🤖 Training XGBoost & SVR models..."):
        # Add cyclical time encoding for months (better than linear)
        df_monthly = df_monthly.copy()
        df_monthly['month_sin'] = np.sin(2 * np.pi * df_monthly['month'] / 12)
        df_monthly['month_cos'] = np.cos(2 * np.pi * df_monthly['month'] / 12)
        
        feature_columns = [
            'month_sin', 'month_cos', 'latitude', 'longitude', 'temperature',
            'humidity', 'air_pressure', 'oni_index', 'el_nino', 'la_nina'
        ]
        
        X = df_monthly[feature_columns].values
        y = df_monthly['monthly_rainfall'].values
        
        # Scaler for SVR models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Three models: XGBoost (best) + SVR variants
        models = {
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                random_state=42,
                n_jobs=-1
            ),
            'RBF': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            ),
            'Polynomial': SVR(
                kernel='poly',
                C=100,
                degree=3,
                gamma='scale',
                epsilon=0.1
            )
        }
        
        # Train models (XGBoost with raw data, SVR with scaled data)
        for name, model in models.items():
            if 'XGBoost' in name:
                model.fit(X, y)  # Tree-based doesn't need scaling
            else:
                model.fit(X_scaled, y)  # SVR needs scaling
        
        return models, scaler, feature_columns


def get_predictions_for_month_year(df_monthly, models, scaler, feature_columns, 
                                   selected_year, selected_month, model_name):
    """
    Filter data and compute predictions for specific month/year
    """
    # Filter data for selected month/year
    df_filtered = df_monthly[
        (df_monthly['year'] == selected_year) & 
        (df_monthly['month'] == selected_month)
    ].copy()
    
    if len(df_filtered) == 0:
        return None, None, None, None
    
    # Add cyclical encoding if not already present
    if 'month_sin' not in df_filtered.columns:
        df_filtered['month_sin'] = np.sin(2 * np.pi * df_filtered['month'] / 12)
        df_filtered['month_cos'] = np.cos(2 * np.pi * df_filtered['month'] / 12)
    
    # Prepare features
    X_filtered = df_filtered[feature_columns].values
    
    # Get predictions (XGBoost uses raw data, SVR uses scaled data)
    model = models[model_name]
    if 'XGBoost' in model_name:
        predictions = model.predict(X_filtered)
    else:
        X_filtered_scaled = scaler.transform(X_filtered)
        predictions = model.predict(X_filtered_scaled)
    
    # Clip predictions to minimum of 0 (no negative rainfall)
    predictions = np.maximum(predictions, 0)
    
    # Get actual values if available
    actual_values = df_filtered['monthly_rainfall'].values
    
    # Calculate metrics (RMSE only)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    
    return df_filtered, predictions, actual_values, rmse


def create_interactive_map(df_filtered, predictions):
    """
    Create interactive Plotly map with rainfall predictions
    """
    fig = go.Figure()
    
    # Color scale based on predictions
    min_val = predictions.min()
    max_val = predictions.max()
    
    # Add scatter mapbox for cities
    fig.add_trace(go.Scattermapbox(
        lat=df_filtered['latitude'],
        lon=df_filtered['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            color=predictions,
            colorscale='YlGnBu',
            showscale=True,
            colorbar=dict(
                title="Rainfall<br>(mm)",
                thickness=15,
                len=0.7
            ),
            cmin=min_val,
            cmax=max_val
        ),
        text=[f"{city}<br>Predicted: {pred:.1f} mm<br>Lat: {lat:.2f}<br>Lon: {lon:.2f}" 
              for city, pred, lat, lon in zip(
                  df_filtered['city_name'], 
                  predictions,
                  df_filtered['latitude'],
                  df_filtered['longitude']
              )],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Predictions'
    ))
    
    # Update layout for map
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=12.8797, lon=121.7740),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig


def create_heatmap_density(df_filtered, predictions):
    """
    Create density heatmap using Plotly
    """
    fig = go.Figure(go.Densitymapbox(
        lat=df_filtered['latitude'],
        lon=df_filtered['longitude'],
        z=predictions,
        radius=30,
        colorscale='YlGnBu',
        showscale=True,
        colorbar=dict(title="Rainfall (mm)")
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=12.8797, lon=121.7740),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def create_comparison_chart(actual_values, predictions, city_names):
    """
    Create comparison bar chart
    """
    df_compare = pd.DataFrame({
        'City': city_names[:20],  # Show top 20 cities
        'Actual': actual_values[:20],
        'Predicted': predictions[:20]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Actual',
        x=df_compare['City'],
        y=df_compare['Actual'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Predicted',
        x=df_compare['City'],
        y=df_compare['Predicted'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Rainfall (Top 20 Cities)',
        xaxis_title='City',
        yaxis_title='Rainfall (mm)',
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


@st.cache_data
def generate_forecasts(_df_monthly, _model, _scaler, _feature_columns, future_years, model_name):
    """Generate forecast scenarios (cached by model)"""
    with st.spinner(f"🔮 Generating forecasts for {future_years[0]}-{future_years[-1]} ({model_name})..."):
        forecaster = RainfallForecaster(_df_monthly, _model, _scaler, _feature_columns)
        scenarios = forecaster.generate_all_scenarios(future_years)
        return scenarios


def main():
    """
    Main Streamlit app
    """
    # Header
    st.markdown('<p class="main-header">🌧️ Philippines Rainfall Prediction System</p>', 
                unsafe_allow_html=True)
    st.markdown("**Historical Analysis (2020-2023) + Future Forecasts (2024-2030)**")
    st.markdown("---")
    
    # Load data (cached)
    df_monthly = load_and_preprocess_data()
    
    # Train models (cached)
    models, scaler, feature_columns = train_models(df_monthly)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("---")
    
    # MODE SELECTION - NEW FEATURE
    mode = st.sidebar.radio(
        "📊 Analysis Mode:",
        options=['📜 Historical (2020-2023)', '🔮 Forecast (2024-2030)'],
        help="Historical: Analyze actual past data\nForecast: Predict future rainfall"
    )
    
    is_forecast_mode = 'Forecast' in mode
    
    st.sidebar.markdown("---")
    
    # Month names dictionary
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    # Year and Month selection (different for historical vs forecast)
    if is_forecast_mode:
        # FORECAST MODE
        available_years = list(range(2024, 2031))  # 2024-2030
        selected_year = st.sidebar.selectbox(
            "📅 Forecast Year",
            options=available_years,
            index=0
        )
        
        available_months = list(range(1, 13))  # All months
        selected_month = st.sidebar.selectbox(
            "📆 Month",
            options=available_months,
            format_func=lambda x: month_names[x],
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # ENSO Scenario selection (only for forecast)
        scenario = st.sidebar.selectbox(
            "🌡️ Climate Scenario:",
            options=['neutral', 'el_nino', 'la_nina'],
            format_func=lambda x: {
                'neutral': '⚪ Neutral (Normal)',
                'el_nino': '🔴 El Niño (Warmer/Drier)',
                'la_nina': '🔵 La Niña (Cooler/Wetter)'
            }[x],
            help="ENSO phase affects rainfall patterns"
        )
    else:
        # HISTORICAL MODE
        available_years = sorted(df_monthly['year'].unique())
        selected_year = st.sidebar.selectbox(
            "📅 Select Year",
            options=available_years,
            index=len(available_years) - 1
        )
        
        available_months = sorted(
            df_monthly[df_monthly['year'] == selected_year]['month'].unique()
        )
        
        selected_month = st.sidebar.selectbox(
            "📆 Select Month",
            options=available_months,
            format_func=lambda x: month_names[x],
            index=0
        )
    
    # Kernel selection
    st.sidebar.markdown("---")
    model_name = st.sidebar.radio(
        "🚀 ML Algorithm",
        options=['XGBoost', 'RBF', 'Polynomial'],
        index=0,
        help="XGBoost: Best accuracy (RMSE: 69.96mm, R²: 0.78)\nRBF: SVR with RBF kernel (RMSE: 101mm, R²: 0.55)\nPolynomial: SVR with polynomial kernel (RMSE: 108mm, R²: 0.49)"
    )
    
    # Visualization type
    st.sidebar.markdown("---")
    viz_type = st.sidebar.radio(
        "🗺️ Map Type",
        options=['Scatter Map', 'Density Heatmap'],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    if is_forecast_mode:
        st.sidebar.info("""
        **Forecast Mode:**
        1. Select future year (2024-2030)
        2. Choose month
        3. Pick climate scenario
        4. Explore forecasts!
        
        Uses trained model to predict future rainfall based on climate scenarios.
        """)
    else:
        st.sidebar.info("""
        **Historical Mode:**
        1. Select year and month
        2. Choose ML algorithm
        3. Pick visualization type
        4. Explore the predictions!
        
        Analyzes actual past data (2020-2023).
        """)
    
    # Get predictions or forecasts
    if is_forecast_mode:
        # FORECAST MODE - Generate predictions for future
        with st.spinner(f"🔮 Generating forecast for {month_names[selected_month]} {selected_year}..."):
            # Generate forecasts for selected year
            model = models[model_name]
            years_to_forecast = list(range(2024, selected_year + 1))
            scenarios = generate_forecasts(df_monthly, model, scaler, feature_columns, years_to_forecast, model_name)
            
            # Get forecast for selected month/year/scenario
            forecast_df = scenarios[scenario]
            df_filtered = forecast_df[
                (forecast_df['year'] == selected_year) & 
                (forecast_df['month'] == selected_month)
            ].copy()
            
            predictions = df_filtered['predicted_rainfall'].values
            actual_values = None  # No actual values for future
            rmse = None
    else:
        # HISTORICAL MODE - Analyze past data
        with st.spinner(f"🔮 Computing predictions for {month_names[selected_month]} {selected_year}..."):
            result = get_predictions_for_month_year(
                df_monthly, models, scaler, feature_columns,
                selected_year, selected_month, model_name
            )
            df_filtered, predictions, actual_values, rmse = result
    
    if df_filtered is None or len(df_filtered) == 0:
        st.error(f"❌ No data available for {month_names[selected_month]} {selected_year}")
        return
    
    # Display metrics
    if is_forecast_mode:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📍 Cities",
                value=len(df_filtered),
                help="Number of cities with data"
            )
        
        with col2:
            st.metric(
                label="🔮 Scenario",
                value=scenario.replace('_', ' ').title(),
                help="Climate scenario for forecast"
            )
        
        with col3:
            rainfall_range = f"{predictions.min():.0f}-{predictions.max():.0f}"
            st.metric(
                label="📊 Range",
                value=f"{rainfall_range} mm",
                help="Min-Max forecast range"
            )
    else:
        # Historical mode - only show Cities and Avg Rainfall (no RMSE)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="📍 Cities",
                value=len(df_filtered),
                help="Number of cities with data"
            )
        
        with col2:
            avg_rainfall = predictions.mean()
            st.metric(
                label="🌧️ Avg Rainfall",
                value=f"{avg_rainfall:.1f} mm",
                help="Average predicted rainfall"
            )
    
    st.markdown("---")
    
    # Status indicator
    if is_forecast_mode:
        st.warning(f"🔮 **FORECAST MODE** - Predicting {month_names[selected_month]} {selected_year} ({scenario.replace('_', ' ').title()} scenario)")
    else:
        st.info(f"📜 **HISTORICAL MODE** - Analyzing actual data for {month_names[selected_month]} {selected_year}")
    
    # Main visualization
    map_title = f"🗺️ {month_names[selected_month]} {selected_year} - {model_name}"
    if is_forecast_mode:
        map_title += f" ({scenario.replace('_', ' ').title()})"
    st.subheader(map_title)
    
    if viz_type == 'Scatter Map':
        fig_map = create_interactive_map(df_filtered, predictions)
    else:
        fig_map = create_heatmap_density(df_filtered, predictions)
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Additional visualizations
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        if is_forecast_mode:
            # For forecast mode, show scenario comparison if we have all scenarios
            st.subheader("📊 Scenario Comparison")
            
            # Show comparison for top cities
            top_cities = df_filtered.nlargest(10, 'predicted_rainfall')['city_name'].values
            
            fig_scenarios = go.Figure()
            
            for scenario_name in ['neutral', 'el_nino', 'la_nina']:
                scenario_df = scenarios[scenario_name]
                scenario_month = scenario_df[
                    (scenario_df['year'] == selected_year) & 
                    (scenario_df['month'] == selected_month)
                ]
                
                # Get predictions for top cities
                city_preds = []
                for city in top_cities:
                    pred = scenario_month[scenario_month['city_name'] == city]['predicted_rainfall'].values
                    city_preds.append(pred[0] if len(pred) > 0 else 0)
                
                fig_scenarios.add_trace(go.Bar(
                    name=scenario_name.replace('_', ' ').title(),
                    x=top_cities,
                    y=city_preds
                ))
            
            fig_scenarios.update_layout(
                title='Top 10 Cities - All Scenarios',
                xaxis_title='City',
                yaxis_title='Predicted Rainfall (mm)',
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_scenarios, use_container_width=True)
        else:
            # For historical mode, show actual vs predicted
            st.subheader("📊 Actual vs Predicted")
            fig_compare = create_comparison_chart(
                actual_values, 
                predictions, 
                df_filtered['city_name'].values
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    
    with col_right:
        st.subheader("📈 Statistical Summary")
        
        if is_forecast_mode:
            # For forecast mode, show forecast statistics only
            summary_data = {
                'Metric': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
                'Forecast': [
                    f"{predictions.min():.1f}",
                    f"{predictions.max():.1f}",
                    f"{predictions.mean():.1f}",
                    f"{np.median(predictions):.1f}",
                    f"{predictions.std():.1f}"
                ]
            }
        else:
            # For historical mode, show both actual and predicted
            summary_data = {
                'Metric': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
                'Actual': [
                    f"{actual_values.min():.1f}",
                    f"{actual_values.max():.1f}",
                    f"{actual_values.mean():.1f}",
                    f"{np.median(actual_values):.1f}",
                    f"{actual_values.std():.1f}"
                ],
                'Predicted': [
                    f"{predictions.min():.1f}",
                    f"{predictions.max():.1f}",
                    f"{predictions.mean():.1f}",
                    f"{np.median(predictions):.1f}",
                    f"{predictions.std():.1f}"
                ]
            }
        
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True
        )
        
        # ENSO info
        if is_forecast_mode:
            # Show scenario info for forecast
            oni_value = df_filtered['oni_index'].iloc[0]
            if scenario == 'el_nino':
                enso_status = "🔴 El Niño Scenario"
                enso_color = "#ff4444"
                enso_desc = "Warmer, drier conditions"
            elif scenario == 'la_nina':
                enso_status = "🔵 La Niña Scenario"
                enso_color = "#4444ff"
                enso_desc = "Cooler, wetter conditions"
            else:
                enso_status = "⚪ Neutral Scenario"
                enso_color = "#888888"
                enso_desc = "Normal conditions"
            
            st.markdown(f"""
            <div style='padding: 1rem; background-color: {enso_color}22; 
                        border-left: 4px solid {enso_color}; border-radius: 5px; margin-top: 1rem;'>
                <h4 style='margin: 0;'>Climate Scenario</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
                    <b>{enso_status}</b><br>
                    <span style='font-size: 0.9rem;'>{enso_desc}</span><br>
                    <span style='font-size: 0.8rem; color: #666;'>ONI: {oni_value:.2f}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show actual ENSO status for historical
            oni_value = df_filtered['oni_index'].iloc[0]
            if oni_value > 0.5:
                enso_status = "🔴 El Niño"
                enso_color = "#ff4444"
            elif oni_value < -0.5:
                enso_status = "🔵 La Niña"
                enso_color = "#4444ff"
            else:
                enso_status = "⚪ Neutral"
                enso_color = "#888888"
            
            st.markdown(f"""
            <div style='padding: 1rem; background-color: {enso_color}22; 
                        border-left: 4px solid {enso_color}; border-radius: 5px; margin-top: 1rem;'>
                <h4 style='margin: 0;'>ENSO Status</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
                    <b>{enso_status}</b> (ONI: {oni_value:.2f})
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data table
    st.markdown("---")
    with st.expander("📋 View Detailed Data Table"):
        if is_forecast_mode:
            # Forecast mode - show forecast only
            display_df = df_filtered[['city_name', 'latitude', 'longitude', 
                                       'temperature', 'humidity', 'air_pressure']].copy()
            display_df['forecast_rainfall'] = predictions
            
            st.dataframe(
                display_df.sort_values('forecast_rainfall', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            # Historical mode - show actual vs predicted
            display_df = df_filtered[['city_name', 'latitude', 'longitude', 
                                       'temperature', 'humidity', 'air_pressure', 
                                       'monthly_rainfall']].copy()
            display_df['predicted_rainfall'] = predictions
            display_df['error'] = display_df['monthly_rainfall'] - predictions
            display_df['abs_error'] = np.abs(display_df['error'])
            
            st.dataframe(
                display_df.sort_values('abs_error', ascending=False),
                use_container_width=True,
                hide_index=True
            )
    
    # Footer
    st.markdown("---")
    
    if is_forecast_mode:
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><b>Philippines Rainfall Prediction & Forecast System</b></p>
            <p>Using Support Vector Regression</p>
            <p>Historical Data: 2020-2023 | Forecast: 2024-2030 | Models: RBF, Polynomial, Sigmoid</p>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
                ⚠️ Forecasts are scenario-based projections. Uncertainty increases with time horizon.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><b>Philippines Rainfall Prediction System</b></p>
            <p>Using Support Vector Regression with Actual Humidity & Pressure Data</p>
            <p>Historical Data: 2020-2023 | Models: RBF, Polynomial, Sigmoid</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

