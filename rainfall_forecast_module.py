"""
Philippines Rainfall Forecasting Module
Predicts future rainfall (2024-2030) based on historical patterns (2020-2023)

Uses two-stage approach:
1. Forecast meteorological features (temperature, humidity, pressure)
2. Apply trained SVR model to predict rainfall

Author: Data Science Project
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RainfallForecaster:
    """
    Forecasts rainfall for future years using historical patterns
    """
    
    def __init__(self, historical_df, trained_model, scaler, feature_columns):
        """
        Initialize forecaster with historical data and trained model
        
        Parameters:
        -----------
        historical_df : DataFrame
            Historical monthly data (2020-2023)
        trained_model : SVR
            Trained SVR model
        scaler : StandardScaler
            Fitted scaler from training
        feature_columns : list
            List of feature column names
        """
        self.historical_df = historical_df
        self.trained_model = trained_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def forecast_enso_scenarios(self, future_years):
        """
        Generate ENSO (El Niño/La Niña) scenarios for future years
        
        Returns three scenarios: neutral, El Niño, and La Niña
        """
        scenarios = {}
        
        for year in future_years:
            # Scenario 1: Neutral conditions
            scenarios[f'{year}_neutral'] = {
                month: 0.0 for month in range(1, 13)
            }
            
            # Scenario 2: El Niño conditions
            scenarios[f'{year}_el_nino'] = {
                1: 0.8, 2: 1.0, 3: 1.2, 4: 1.3, 5: 1.2, 6: 1.0,
                7: 0.8, 8: 0.6, 9: 0.5, 10: 0.6, 11: 0.8, 12: 1.0
            }
            
            # Scenario 3: La Niña conditions
            scenarios[f'{year}_la_nina'] = {
                1: -1.0, 2: -1.2, 3: -1.3, 4: -1.2, 5: -1.0, 6: -0.8,
                7: -0.7, 8: -0.8, 9: -1.0, 10: -1.1, 11: -1.2, 12: -1.1
            }
        
        return scenarios
    
    def forecast_temperature(self, city_name, future_years, trend_adjustment=0.02):
        """
        Forecast temperature using historical average + linear trend
        
        Parameters:
        -----------
        city_name : str
            City name
        future_years : list
            Years to forecast
        trend_adjustment : float
            Annual warming trend (°C per year), default 0.02°C/year
        
        Returns:
        --------
        dict : {(year, month): temperature}
        """
        city_data = self.historical_df[self.historical_df['city_name'] == city_name]
        
        if len(city_data) == 0:
            # Use overall average if city not found
            city_data = self.historical_df
        
        forecasts = {}
        
        # Calculate monthly averages from historical data
        monthly_avg = city_data.groupby('month')['temperature'].mean()
        
        # Apply trend for each future year
        base_year = self.historical_df['year'].max()
        
        for year in future_years:
            years_ahead = year - base_year
            trend_offset = trend_adjustment * years_ahead
            
            for month in range(1, 13):
                if month in monthly_avg.index:
                    forecasts[(year, month)] = monthly_avg[month] + trend_offset
                else:
                    # Use overall mean if month missing
                    forecasts[(year, month)] = city_data['temperature'].mean() + trend_offset
        
        return forecasts
    
    def forecast_humidity(self, city_name, future_years, temperature_forecasts):
        """
        Forecast humidity based on temperature relationship
        (Warmer temperatures → slightly lower relative humidity)
        """
        city_data = self.historical_df[self.historical_df['city_name'] == city_name]
        
        if len(city_data) == 0:
            city_data = self.historical_df
        
        forecasts = {}
        monthly_avg = city_data.groupby('month')['humidity'].mean()
        
        for (year, month), temp in temperature_forecasts.items():
            base_humidity = monthly_avg[month] if month in monthly_avg.index else city_data['humidity'].mean()
            
            # Adjust humidity based on temperature anomaly
            historical_temp = city_data[city_data['month'] == month]['temperature'].mean()
            temp_anomaly = temp - historical_temp
            
            # Rough relationship: +1°C → -2% relative humidity
            humidity_adjustment = -2 * temp_anomaly
            
            forecasts[(year, month)] = np.clip(base_humidity + humidity_adjustment, 30, 100)
        
        return forecasts
    
    def forecast_pressure(self, city_name, future_years):
        """
        Forecast air pressure using historical monthly averages
        (Pressure is relatively stable, use historical mean)
        """
        city_data = self.historical_df[self.historical_df['city_name'] == city_name]
        
        if len(city_data) == 0:
            city_data = self.historical_df
        
        forecasts = {}
        monthly_avg = city_data.groupby('month')['air_pressure'].mean()
        
        for year in future_years:
            for month in range(1, 13):
                if month in monthly_avg.index:
                    forecasts[(year, month)] = monthly_avg[month]
                else:
                    forecasts[(year, month)] = 1013.25  # Standard sea level
        
        return forecasts
    
    def create_forecast_dataframe(self, future_years, enso_scenario='neutral'):
        """
        Create complete forecast dataframe with all features
        
        Parameters:
        -----------
        future_years : list
            Years to forecast (e.g., [2024, 2025, 2026])
        enso_scenario : str
            'neutral', 'el_nino', or 'la_nina'
        
        Returns:
        --------
        DataFrame with forecasted features for all cities
        """
        # Get ENSO scenarios
        all_enso = self.forecast_enso_scenarios(future_years)
        
        # Get unique cities
        cities = self.historical_df['city_name'].unique()
        
        forecast_rows = []
        
        for city in cities:
            # Get city coordinates
            city_data = self.historical_df[self.historical_df['city_name'] == city].iloc[0]
            lat = city_data['latitude']
            lon = city_data['longitude']
            
            # Forecast features
            temp_forecasts = self.forecast_temperature(city, future_years)
            humidity_forecasts = self.forecast_humidity(city, future_years, temp_forecasts)
            pressure_forecasts = self.forecast_pressure(city, future_years)
            
            # Build forecast dataframe
            for year in future_years:
                for month in range(1, 13):
                    # Get ENSO index for this scenario
                    scenario_key = f'{year}_{enso_scenario}'
                    oni_index = all_enso.get(scenario_key, {}).get(month, 0.0)
                    
                    row = {
                        'city_name': city,
                        'year': year,
                        'month': month,
                        'latitude': lat,
                        'longitude': lon,
                        'temperature': temp_forecasts.get((year, month), city_data.get('temperature', 27.0)),
                        'humidity': humidity_forecasts.get((year, month), 70.0),
                        'air_pressure': pressure_forecasts.get((year, month), 1013.25),
                        'oni_index': oni_index,
                        'el_nino': 1 if oni_index > 0.5 else 0,
                        'la_nina': 1 if oni_index < -0.5 else 0,
                        'scenario': enso_scenario
                    }
                    
                    forecast_rows.append(row)
        
        return pd.DataFrame(forecast_rows)
    
    def predict_rainfall(self, forecast_df):
        """
        Use trained SVR model to predict rainfall for forecasted features
        
        Parameters:
        -----------
        forecast_df : DataFrame
            Dataframe with forecasted features
        
        Returns:
        --------
        DataFrame with rainfall predictions added
        """
        # Prepare features in same order as training
        X_forecast = forecast_df[self.feature_columns].values
        
        # Scale features
        X_forecast_scaled = self.scaler.transform(X_forecast)
        
        # Predict rainfall
        predictions = self.trained_model.predict(X_forecast_scaled)
        
        # Add predictions to dataframe
        forecast_df['predicted_rainfall'] = predictions
        
        return forecast_df
    
    def generate_all_scenarios(self, future_years):
        """
        Generate forecasts for all ENSO scenarios
        
        Returns:
        --------
        dict : {scenario_name: forecast_dataframe}
        """
        scenarios = {}
        
        for scenario_name in ['neutral', 'el_nino', 'la_nina']:
            print(f"Generating {scenario_name} scenario forecast...")
            
            # Create forecast dataframe
            forecast_df = self.create_forecast_dataframe(future_years, scenario_name)
            
            # Predict rainfall
            forecast_df = self.predict_rainfall(forecast_df)
            
            scenarios[scenario_name] = forecast_df
        
        return scenarios
    
    def get_city_forecast(self, city_name, year, scenarios_dict):
        """
        Get forecast for specific city and year across all scenarios
        """
        results = {}
        
        for scenario_name, forecast_df in scenarios_dict.items():
            city_year = forecast_df[
                (forecast_df['city_name'] == city_name) & 
                (forecast_df['year'] == year)
            ].copy()
            
            results[scenario_name] = city_year
        
        return results
    
    def calculate_uncertainty(self, scenarios_dict, year, month):
        """
        Calculate prediction uncertainty across scenarios
        
        Returns mean, min, max predictions for given year/month
        """
        predictions = []
        
        for scenario_df in scenarios_dict.values():
            month_data = scenario_df[
                (scenario_df['year'] == year) & 
                (scenario_df['month'] == month)
            ]['predicted_rainfall'].values
            
            predictions.append(month_data)
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(axis=0),
            'min': predictions.min(axis=0),
            'max': predictions.max(axis=0),
            'std': predictions.std(axis=0)
        }


def example_usage():
    """
    Example of how to use the forecaster
    """
    print("Example: Forecasting rainfall for 2024-2026")
    print("="*70)
    
    # This would normally come from your trained model
    # For demo purposes, showing the structure
    
    print("\nSteps:")
    print("1. Load historical data (2020-2023)")
    print("2. Train SVR model")
    print("3. Create forecaster with trained model")
    print("4. Generate scenarios (Neutral, El Niño, La Niña)")
    print("5. Predict rainfall for each scenario")
    print("6. Visualize results with uncertainty bands")
    
    print("\nScenarios:")
    print("  - Neutral: Normal conditions (ONI ≈ 0)")
    print("  - El Niño: Warmer, drier conditions (ONI > 0.5)")
    print("  - La Niña: Cooler, wetter conditions (ONI < -0.5)")
    
    print("\nUncertainty:")
    print("  - Min/Max across scenarios shows forecast range")
    print("  - Mean gives best estimate")
    print("  - Standard deviation shows variability")


if __name__ == "__main__":
    example_usage()

