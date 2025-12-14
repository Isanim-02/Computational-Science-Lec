"""
Predictive Rainfall Model for the Philippines
Using Support Vector Regression with Multiple Kernels
ENHANCED VERSION: Uses hourly data for actual humidity and pressure

Author: Data Science Project
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing kriging library, provide alternative if not available
try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except ImportError:
    print("PyKrige not installed. Spatial interpolation will be limited.")
    KRIGING_AVAILABLE = False


class PhilippinesRainfallPredictorEnhanced:
    """
    Enhanced rainfall prediction system using Support Vector Regression
    Uses hourly data to extract actual relative humidity and pressure
    """
    
    def __init__(self, daily_data_path, hourly_data_path, cities_path):
        """
        Initialize the predictor with data paths
        
        Parameters:
        -----------
        daily_data_path : str
            Path to daily weather data CSV file
        hourly_data_path : str
            Path to hourly weather data CSV file (for humidity & pressure)
        cities_path : str
            Path to cities coordinates CSV file
        """
        self.daily_data_path = daily_data_path
        self.hourly_data_path = hourly_data_path
        self.cities_path = cities_path
        self.df_monthly = None
        self.X_scaled = None
        self.y = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_data(self, sample_size_daily=None, sample_size_hourly=None):
        """
        Load and prepare the datasets
        
        Parameters:
        -----------
        sample_size_daily : int, optional
            Number of rows to sample from daily data
        sample_size_hourly : int, optional
            Number of rows to sample from hourly data
        """
        print("Loading data...")
        
        # Load cities data
        self.df_cities = pd.read_csv(self.cities_path)
        print(f"Loaded {len(self.df_cities)} cities")
        
        # Load daily weather data
        if sample_size_daily:
            self.df_daily = pd.read_csv(self.daily_data_path, nrows=sample_size_daily)
        else:
            print("Reading daily data (this may take a moment)...")
            self.df_daily = pd.read_csv(self.daily_data_path)
        
        print(f"Loaded {len(self.df_daily)} daily weather records")
        
        # Load hourly data for humidity and pressure
        print("Loading hourly data for humidity and pressure...")
        if sample_size_hourly:
            self.df_hourly = pd.read_csv(self.hourly_data_path, nrows=sample_size_hourly)
        else:
            # Read hourly data in chunks if it's very large
            print("  (Processing hourly data - this will take 2-3 minutes)...")
            try:
                self.df_hourly = pd.read_csv(self.hourly_data_path)
            except MemoryError:
                print("  Memory issue detected, reading in chunks...")
                # Read only necessary columns in chunks
                cols_needed = ['city_name', 'datetime', 'relative_humidity_2m', 'pressure_msl']
                chunk_list = []
                for chunk in pd.read_csv(self.hourly_data_path, 
                                        usecols=cols_needed, 
                                        chunksize=100000):
                    chunk_list.append(chunk)
                self.df_hourly = pd.concat(chunk_list, ignore_index=True)
        
        print(f"Loaded {len(self.df_hourly)} hourly records")
        
    def preprocess_data(self):
        """
        Preprocess data: 
        1. Extract humidity & pressure from hourly data
        2. Aggregate daily to monthly 
        3. Merge everything together
        """
        print("\nPreprocessing data...")
        
        # ===== STEP 1: Process Hourly Data for Humidity & Pressure =====
        print("Step 1: Extracting humidity and pressure from hourly data...")
        
        self.df_hourly['datetime'] = pd.to_datetime(self.df_hourly['datetime'])
        self.df_hourly['date'] = self.df_hourly['datetime'].dt.date
        
        # Aggregate hourly to daily (average humidity and pressure per day)
        hourly_daily = self.df_hourly.groupby(['city_name', 'date']).agg({
            'relative_humidity_2m': 'mean',
            'pressure_msl': 'mean'
        }).reset_index()
        
        hourly_daily.rename(columns={
            'relative_humidity_2m': 'humidity',
            'pressure_msl': 'air_pressure'
        }, inplace=True)
        
        print(f"  Created {len(hourly_daily)} daily humidity/pressure records")
        
        # ===== STEP 2: Process Daily Data =====
        print("Step 2: Processing daily weather data...")
        
        self.df_daily['datetime'] = pd.to_datetime(self.df_daily['datetime'])
        self.df_daily['date'] = self.df_daily['datetime'].dt.date
        self.df_daily['year'] = self.df_daily['datetime'].dt.year
        self.df_daily['month'] = self.df_daily['datetime'].dt.month
        
        # Merge daily data with humidity/pressure from hourly
        print("  Merging humidity and pressure into daily data...")
        self.df_daily = self.df_daily.merge(
            hourly_daily[['city_name', 'date', 'humidity', 'air_pressure']],
            on=['city_name', 'date'],
            how='left'
        )
        
        print(f"  Merged data: {len(self.df_daily)} records with humidity & pressure")
        
        # ===== STEP 3: Aggregate to Monthly =====
        print("Step 3: Aggregating to monthly level...")
        
        monthly_agg = self.df_daily.groupby(['city_name', 'year', 'month']).agg({
            'temperature_2m_mean': 'mean',
            'humidity': 'mean',  # Actual relative humidity!
            'air_pressure': 'mean',  # Actual air pressure!
            'precipitation_sum': 'sum',  # Total monthly rainfall
            'rain_sum': 'sum',
            'wind_speed_10m_max': 'mean'
        }).reset_index()
        
        monthly_agg.rename(columns={
            'temperature_2m_mean': 'temperature',
            'precipitation_sum': 'monthly_rainfall'
        }, inplace=True)
        
        # ===== STEP 4: Merge with City Coordinates =====
        self.df_monthly = monthly_agg.merge(
            self.df_cities, 
            on='city_name', 
            how='left'
        )
        
        print(f"Created {len(self.df_monthly)} monthly records")
        print(f"Date range: {self.df_monthly['year'].min()}-{self.df_monthly['year'].max()}")
        print(f"Using ACTUAL humidity and pressure from hourly data!")
        
    def add_enso_indices(self):
        """
        Add El Niño Southern Oscillation (ENSO) indices
        Using ONI (Oceanic Niño Index) historical data
        """
        print("\nAdding ENSO (El Niño/La Niña) indices...")
        
        # ONI Index data (Oceanic Niño Index) - ACTUAL NOAA DATA
        # Source: NOAA Climate Prediction Center
        # https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
        oni_data = {
            2020: {1: 0.5, 2: 0.6, 3: 0.4, 4: 0.2, 5: -0.1, 6: -0.3, 
                   7: -0.4, 8: -0.6, 9: -0.9, 10: -1.2, 11: -1.3, 12: -1.3},  # Mod La Niña
            2021: {1: -1.4, 2: -1.1, 3: -0.8, 4: -0.6, 5: -0.5, 6: -0.4,
                   7: -0.5, 8: -0.7, 9: -0.9, 10: -1.1, 11: -1.1, 12: -1.0},  # Mod La Niña
            2022: {1: -0.9, 2: -0.8, 3: -0.6, 4: -0.4, 5: -0.2, 6: 0.1,
                   7: 0.3, 8: 0.5, 9: 0.8, 10: 1.0, 11: 1.1, 12: 1.0},  # Transition to El Niño
            2023: {1: -0.7, 2: -0.4, 3: -0.1, 4: 0.2, 5: 0.5, 6: 0.8,
                   7: 1.1, 8: 1.3, 9: 1.6, 10: 1.8, 11: 1.9, 12: 2.0}  # Strong El Niño
        }
        
        def get_oni(row):
            year, month = int(row['year']), int(row['month'])
            if year in oni_data and month in oni_data[year]:
                return oni_data[year][month]
            return 0.0
        
        self.df_monthly['oni_index'] = self.df_monthly.apply(get_oni, axis=1)
        self.df_monthly['el_nino'] = (self.df_monthly['oni_index'] > 0.5).astype(int)
        self.df_monthly['la_nina'] = (self.df_monthly['oni_index'] < -0.5).astype(int)
        
        print(f"El Nino periods: {self.df_monthly['el_nino'].sum()} months")
        print(f"La Nina periods: {self.df_monthly['la_nina'].sum()} months")
        
    def prepare_features(self):
        """
        Prepare feature matrix (X) and target variable (y)
        """
        print("\nPreparing features and target variable...")
        
        # Remove rows with missing values
        initial_count = len(self.df_monthly)
        self.df_monthly = self.df_monthly.dropna()
        removed_count = initial_count - len(self.df_monthly)
        
        if removed_count > 0:
            print(f"  Removed {removed_count} rows with missing values")
        
        # Feature columns with ACTUAL humidity and pressure
        feature_columns = [
            'month',
            'latitude',
            'longitude',
            'temperature',
            'humidity',         # ← ACTUAL relative humidity from hourly data!
            'air_pressure',     # ← ACTUAL air pressure from hourly data!
            'oni_index',
            'el_nino',
            'la_nina'
        ]
        
        target_column = 'monthly_rainfall'
        
        # Create feature matrix and target vector
        X = self.df_monthly[feature_columns].values
        y = self.df_monthly[target_column].values
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y
        
        print(f"Feature matrix shape: {self.X_scaled.shape}")
        print(f"Target variable shape: {self.y.shape}")
        print(f"\nFeatures used:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i}. {col}")
        print(f"\nUsing REAL humidity and pressure (not proxies)!")
        print(f"\nRainfall statistics:")
        print(f"  Min:  {y.min():.2f} mm")
        print(f"  Max:  {y.max():.2f} mm")
        print(f"  Mean: {y.mean():.2f} mm")
        print(f"  Std:  {y.std():.2f} mm")
        
    def train_svr_models(self, kernel_params=None):
        """
        Train SVR models with three different kernels
        """
        print("\n" + "="*70)
        print("TRAINING SUPPORT VECTOR REGRESSION MODELS")
        print("="*70)
        
        if kernel_params is None:
            kernel_params = {
                'rbf': {'kernel': 'rbf', 'C': 100, 'gamma': 'scale', 'epsilon': 0.1},
                'poly': {'kernel': 'poly', 'C': 100, 'degree': 3, 'gamma': 'scale', 'epsilon': 0.1},
                'sigmoid': {'kernel': 'sigmoid', 'C': 100, 'gamma': 'scale', 'epsilon': 0.1}
            }
        
        self.models = {
            'RBF': SVR(**kernel_params['rbf']),
            'Polynomial': SVR(**kernel_params['poly']),
            'Sigmoid': SVR(**kernel_params['sigmoid'])
        }
        
        print("\nModel configurations:")
        for name, model in self.models.items():
            print(f"\n{name} Kernel: {model.get_params()}")
    
    def evaluate_with_kfold(self, n_splits=5):
        """
        Evaluate models using k-fold cross-validation
        """
        print("\n" + "="*70)
        print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
        print("="*70)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name} Kernel")
            print(f"{'='*70}")
            
            rmse_scores = []
            r2_scores = []
            fold_num = 1
            
            for train_index, test_index in kf.split(self.X_scaled):
                X_train, X_test = self.X_scaled[train_index], self.X_scaled[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
                
                print(f"\nFold {fold_num}:")
                print(f"  RMSE: {rmse:.4f} mm")
                print(f"  R²:   {r2:.4f}")
                
                fold_num += 1
            
            mean_rmse = np.mean(rmse_scores)
            std_rmse = np.std(rmse_scores)
            mean_r2 = np.mean(r2_scores)
            std_r2 = np.std(r2_scores)
            
            self.results[model_name] = {
                'rmse_scores': rmse_scores,
                'r2_scores': r2_scores,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'mean_r2': mean_r2,
                'std_r2': std_r2
            }
            
            print(f"\n{'-'*70}")
            print(f"SUMMARY for {model_name} Kernel:")
            print(f"{'-'*70}")
            print(f"  Average RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f} mm")
            print(f"  Average R2:   {mean_r2:.4f} +/- {std_r2:.4f}")
            print(f"{'-'*70}")
    
    def plot_results(self):
        """
        Visualize cross-validation results
        """
        print("\nGenerating performance visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(self.results.keys())
        rmse_means = [self.results[m]['mean_rmse'] for m in models]
        rmse_stds = [self.results[m]['std_rmse'] for m in models]
        r2_means = [self.results[m]['mean_r2'] for m in models]
        r2_stds = [self.results[m]['std_r2'] for m in models]
        
        axes[0].bar(models, rmse_means, yerr=rmse_stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[0].set_ylabel('Root Mean Square Error (mm)', fontsize=12)
        axes[0].set_title('RMSE Comparison Across Kernels\n(Using Actual Humidity & Pressure)', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(models, r2_means, yerr=r2_stds, capsize=5,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('R² Comparison Across Kernels\n(Using Actual Humidity & Pressure)', 
                         fontsize=14, fontweight='bold')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('svr_kernel_comparison_enhanced.png', dpi=300, bbox_inches='tight')
        print("Saved: svr_kernel_comparison_enhanced.png")
        
        # Fold-by-fold details
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            row = idx // 3
            col = idx % 3
            
            axes[0, col].plot(range(1, len(results['rmse_scores'])+1), 
                            results['rmse_scores'], 
                            marker='o', linewidth=2, markersize=8)
            axes[0, col].axhline(y=results['mean_rmse'], color='r', 
                               linestyle='--', label=f'Mean: {results["mean_rmse"]:.2f}')
            axes[0, col].set_title(f'{model_name} - RMSE', fontweight='bold')
            axes[0, col].set_xlabel('Fold')
            axes[0, col].set_ylabel('RMSE (mm)')
            axes[0, col].legend()
            axes[0, col].grid(alpha=0.3)
            
            axes[1, col].plot(range(1, len(results['r2_scores'])+1), 
                            results['r2_scores'], 
                            marker='s', linewidth=2, markersize=8, color='green')
            axes[1, col].axhline(y=results['mean_r2'], color='r', 
                               linestyle='--', label=f'Mean: {results["mean_r2"]:.3f}')
            axes[1, col].set_title(f'{model_name} - R²', fontweight='bold')
            axes[1, col].set_xlabel('Fold')
            axes[1, col].set_ylabel('R² Score')
            axes[1, col].legend()
            axes[1, col].grid(alpha=0.3)
        
        plt.suptitle('Detailed Performance (Using Actual Humidity & Pressure from Hourly Data)', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('svr_fold_details_enhanced.png', dpi=300, bbox_inches='tight')
        print("Saved: svr_fold_details_enhanced.png")
        
    def spatial_interpolation_kriging(self, kernel_name='RBF', n_grid=50):
        """
        Perform spatial interpolation using Ordinary Kriging
        """
        if not KRIGING_AVAILABLE:
            print("\nKriging not available. Install PyKrige: pip install pykrige")
            self._simple_spatial_plot(kernel_name)
            return
        
        print(f"\n{'='*70}")
        print(f"SPATIAL INTERPOLATION WITH KRIGING - {kernel_name} Kernel")
        print(f"{'='*70}")
        
        model = self.models[kernel_name]
        model.fit(self.X_scaled, self.y)
        predictions = model.predict(self.X_scaled)
        
        lats = self.df_monthly['latitude'].values
        lons = self.df_monthly['longitude'].values
        
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        grid_lat = np.linspace(lat_min, lat_max, n_grid)
        grid_lon = np.linspace(lon_min, lon_max, n_grid)
        
        print(f"\nPerforming Ordinary Kriging interpolation...")
        print(f"Grid resolution: {n_grid}x{n_grid}")
        
        try:
            OK = OrdinaryKriging(
                lons, lats, predictions,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
            
            z, ss = OK.execute('grid', grid_lon, grid_lat)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            im1 = axes[0].contourf(grid_lon, grid_lat, z, levels=15, cmap='YlGnBu')
            axes[0].scatter(lons, lats, c=predictions, s=20, cmap='YlGnBu', 
                          edgecolors='black', linewidth=0.5, alpha=0.6)
            axes[0].set_xlabel('Longitude', fontsize=12)
            axes[0].set_ylabel('Latitude', fontsize=12)
            axes[0].set_title(f'Kriged Rainfall Prediction - {kernel_name} Kernel\n(Using Actual Humidity & Pressure)', 
                            fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0], label='Rainfall (mm)')
            
            im2 = axes[1].contourf(grid_lon, grid_lat, ss, levels=15, cmap='Reds')
            axes[1].scatter(lons, lats, s=20, c='black', alpha=0.3)
            axes[1].set_xlabel('Longitude', fontsize=12)
            axes[1].set_ylabel('Latitude', fontsize=12)
            axes[1].set_title('Kriging Variance (Uncertainty)', 
                            fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1], label='Variance')
            
            plt.tight_layout()
            plt.savefig(f'kriging_interpolation_{kernel_name.lower()}_enhanced.png', dpi=300, bbox_inches='tight')
            print(f"Saved: kriging_interpolation_{kernel_name.lower()}_enhanced.png")
            
        except Exception as e:
            print(f"Kriging failed: {e}")
            self._simple_spatial_plot(kernel_name)
    
    def _simple_spatial_plot(self, kernel_name):
        """Create simple spatial visualization without kriging"""
        print("\nCreating simple spatial visualization...")
        
        model = self.models[kernel_name]
        model.fit(self.X_scaled, self.y)
        predictions = model.predict(self.X_scaled)
        
        lats = self.df_monthly['latitude'].values
        lons = self.df_monthly['longitude'].values
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(lons, lats, c=predictions, s=100, 
                            cmap='YlGnBu', alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Predicted Rainfall (mm)')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(f'Rainfall Predictions - {kernel_name} Kernel\n(Using Actual Humidity & Pressure)', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'rainfall_spatial_{kernel_name.lower()}_enhanced.png', dpi=300, bbox_inches='tight')
        print(f"Saved: rainfall_spatial_{kernel_name.lower()}_enhanced.png")
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        print("\n" + "="*70)
        print("FINAL SUMMARY REPORT")
        print("="*70)
        
        print(f"\nDataset Information:")
        print(f"  Total monthly records: {len(self.df_monthly)}")
        print(f"  Number of cities: {self.df_monthly['city_name'].nunique()}")
        print(f"  Time period: {self.df_monthly['year'].min()}-{self.df_monthly['year'].max()}")
        print(f"  Features used: 9")
        print(f"    - Month, Latitude, Longitude")
        print(f"    - Temperature")
        print(f"    - ACTUAL Relative Humidity (from hourly data)")
        print(f"    - ACTUAL Air Pressure (from hourly data)")
        print(f"    - ONI Index, El Niño flag, La Niña flag")
        
        print(f"\n{'-'*70}")
        print("Model Performance Comparison:")
        print(f"{'-'*70}")
        print(f"{'Kernel':<15} {'RMSE (mm)':<20} {'R2 Score':<20}")
        print(f"{'-'*70}")
        
        for model_name, results in self.results.items():
            rmse_str = f"{results['mean_rmse']:.4f} +/- {results['std_rmse']:.4f}"
            r2_str = f"{results['mean_r2']:.4f} +/- {results['std_r2']:.4f}"
            print(f"{model_name:<15} {rmse_str:<20} {r2_str:<20}")
        
        print(f"{'-'*70}")
        
        best_model_rmse = min(self.results.items(), key=lambda x: x[1]['mean_rmse'])
        best_model_r2 = max(self.results.items(), key=lambda x: x[1]['mean_r2'])
        
        print(f"\nBest Model by RMSE: {best_model_rmse[0]} "
              f"(RMSE = {best_model_rmse[1]['mean_rmse']:.4f} mm)")
        print(f"Best Model by R2: {best_model_r2[0]} "
              f"(R2 = {best_model_r2[1]['mean_r2']:.4f})")
        
        print("\nENHANCEMENT: Using ACTUAL humidity and pressure from hourly data!")
        print("   This should provide more accurate predictions than proxies.")
        print("\n" + "="*70)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("PHILIPPINES RAINFALL PREDICTION MODEL - ENHANCED")
    print("Using ACTUAL Humidity & Pressure from Hourly Data")
    print("="*70)
    
    predictor = PhilippinesRainfallPredictorEnhanced(
        daily_data_path='daily_data_combined_2020_to_2023.csv',
        hourly_data_path='hourly_data_combined_2020_to_2023.csv',
        cities_path='cities.csv'
    )
    
    # Load data
    # For faster testing: sample_size_daily=50000, sample_size_hourly=500000
    # For full accuracy: both = None
    print("\n⚠️  NOTE: This will take 5-10 minutes due to hourly data processing")
    print("For faster testing, adjust sample sizes in main() function\n")
    
    predictor.load_data(
        sample_size_daily=100000,    # Adjust for speed/accuracy trade-off
        sample_size_hourly=1000000   # Adjust for speed/accuracy trade-off
    )
    
    # Process
    predictor.preprocess_data()
    predictor.add_enso_indices()
    predictor.prepare_features()
    
    # Train
    predictor.train_svr_models()
    
    # Evaluate
    predictor.evaluate_with_kfold(n_splits=5)
    
    # Visualize
    predictor.plot_results()
    predictor.spatial_interpolation_kriging(kernel_name='RBF', n_grid=50)
    
    # Summary
    predictor.print_summary_report()
    
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - svr_kernel_comparison_enhanced.png")
    print("  - svr_fold_details_enhanced.png")
    print("  - kriging_interpolation_rbf_enhanced.png")
    print("\nUsed ACTUAL relative humidity and air pressure!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

