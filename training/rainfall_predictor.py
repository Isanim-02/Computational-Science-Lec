"""
Python class for the rainfall prediction model in the Philippines.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from models import MODELS_TO_TEST 
from sklearn.pipeline import Pipeline

class RainfallPredictor:
    def __init__(self, monthly_data_path, feature_columns=None):
        """
        Initialize the predictor with data paths
        """
        self.monthly_data_path = monthly_data_path
        self.df_monthly = None
        self.models = {}

        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [
                'month_sin', 'month_cos', 'latitude', 'longitude', 'temperature',
                'humidity', 'air_pressure', 'oni_index', 'el_nino', 'la_nina',
                'monthly_rainfall_lag_1',
            ]
        
        # Load data upon initialization
        self._load_data()
        
    def _preprocess_data(self):
        """Removes rows with missing data"""        
        print("\nPreprocessing data...")
        
        # Sort to ensure proper temporal order
        self.df_monthly = self.df_monthly.sort_values(
            by=['city', 'year', 'month']
        )

        INCLUDE_LAGGED_RAINFALL = 'monthly_rainfall_lag_1' in self.feature_columns

        # If lagged rainfall is not included, drop the column
        if not INCLUDE_LAGGED_RAINFALL and 'monthly_rainfall_lag_1' in self.df_monthly.columns:
            self.df_monthly = self.df_monthly.drop(columns=['monthly_rainfall_lag_1'])

        # Drop incomplete rows
        self.df_monthly = self.df_monthly.dropna()

        # Only keep cities with complete number of monthly data
        months_per_city = self.df_monthly.groupby('city').size()
        num_months = months_per_city.max()

        complete_cities = months_per_city[months_per_city == num_months].index
        self.df_monthly = self.df_monthly[self.df_monthly['city'].isin(complete_cities)]

        print(f"Monthly records: {len(self.df_monthly)}")
        print(f"Cities with complete data: {len(complete_cities)}")
        print(f"Features: {len(self.df_monthly.columns) - 4}")


    def _load_data(self):
        """Load all datasets"""
        # Load monthly data, if available
        if not os.path.exists(self.monthly_data_path):
            raise FileNotFoundError(f"Monthly data file not found: {self.monthly_data_path}")
        
        self.df_monthly = pd.read_csv(self.monthly_data_path)
        self.df_monthly.rename(columns={'city_name': 'city'}, inplace=True)
        print(f"Monthly records loaded: {len(self.df_monthly)}")

        self._preprocess_data()

        # Prepare features
        self.X = self.df_monthly[self.feature_columns].values
        self.y = self.df_monthly['monthly_rainfall'].values
        print(f"\nDataset: {len(self.X)} samples, {self.X.shape[1]} features")
        print(f"Features: {', '.join(self.feature_columns)}")

        return 


    def cross_validate(self, cv_method='time_series'):
        """Cross-validate the models."""
        print("\n" + "="*70)
        print("TRAINING AND EVALUATION (Cross-Validation)")
        print("="*70)
                
        # Time-Series CV
        if cv_method == 'time_series':
            print("\nUsing Time-Series Cross-Validation")

            years = self.df_monthly['year'].values
            unique_years = np.unique(years)
            results = {}
            
            for model_name, model in MODELS_TO_TEST.items():
                print(f"\n{model_name}")
                print("-" * 70)
                
                rmse_scores = []
                r2_scores = []
                mae_scores = []
                
                for i in range(len(unique_years) - 1):
                    # Train on all years <= unique_years[i]
                    train_idx = np.where(years <= unique_years[i])[0]
                    # Validate on next year
                    test_idx = np.where(years == unique_years[i+1])[0]
                    
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    rmse_scores.append(rmse)
                    r2_scores.append(r2)
                    mae_scores.append(mae)

                    print(f"Fold {i+1}: Train <= {unique_years[i]}, Test = {unique_years[i+1]}")                
                    print(f"   RMSE = {rmse:>7.4f} mm | R² = {r2:.4f} | MAE = {mae:>6.4f} mm")

                # Store results
                results[model_name] = {
                    'rmse_mean': np.mean(rmse_scores),
                    'rmse_std': np.std(rmse_scores),
                    'r2_mean': np.mean(r2_scores),
                    'r2_std': np.std(r2_scores),
                    'mae_mean': np.mean(mae_scores),
                    'mae_std': np.std(mae_scores)
                }
                
                print(f"  Average: RMSE = {results[model_name]['rmse_mean']:.2f} +/- {results[model_name]['rmse_std']:.2f} mm")
                print(f"           R² = {results[model_name]['r2_mean']:.4f} +/- {results[model_name]['r2_std']:.4f}")
        elif cv_method == 'k_fold':
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            results = {}
            
            for model_name, model in MODELS_TO_TEST.items():
                print(f"\n{model_name}")
                print("-" * 70)
                
                rmse_scores = []
                r2_scores = []
                mae_scores = []

                for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X), 1):
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    rmse_scores.append(rmse)
                    r2_scores.append(r2)
                    mae_scores.append(mae)
                    
                    print(f"  Fold {fold}: RMSE = {rmse:>7.2f} mm | R² = {r2:.4f} | MAE = {mae:>6.2f} mm")
                
                # Store results
                results[model_name] = {
                    'rmse_mean': np.mean(rmse_scores),
                    'rmse_std': np.std(rmse_scores),
                    'r2_mean': np.mean(r2_scores),
                    'r2_std': np.std(r2_scores),
                    'mae_mean': np.mean(mae_scores),
                    'mae_std': np.std(mae_scores)
                }
                
                print(f"  Average: RMSE = {results[model_name]['rmse_mean']:.2f} +/- {results[model_name]['rmse_std']:.2f} mm")
                print(f"           R² = {results[model_name]['r2_mean']:.4f} +/- {results[model_name]['r2_std']:.4f}")


        # Train final models on full dataset
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS ON FULL DATASET")
        print("="*70)
        
        return results
    

    def train_final_models(self, model_output_dir=None):
        """Train new models on the full dataset."""
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS ON FULL DATASET")
        print("="*70)
        
        self.models = {}
        
        for model_name, model in MODELS_TO_TEST.items():
            print(f"\nTraining {model_name}...")
            model.fit(self.X, self.y)
            self.models[model_name] = model
            print(f"Finished training {model_name}")

            if model_output_dir is not None:
                model_filename = f"{model_output_dir}/{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '') }_model.pkl"
                joblib.dump(model, model_filename)
                print(f"Saved: {model_filename}")
        
        return self.models
    
    
    def plot_results(self, results, results_dir=None):
        """Generate visualization plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Create comparison plot
        _, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        models = list(results.keys())
        rmse_means = [results[m]['rmse_mean'] for m in models]
        rmse_stds = [results[m]['rmse_std'] for m in models]
        
        axes[0].bar(range(len(models)), rmse_means, yerr=rmse_stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=15, ha='right')
        axes[0].set_ylabel('RMSE (mm)', fontsize=12)
        axes[0].set_title('Root Mean Square Error Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
            axes[0].text(i, mean + std + 2, f'{mean:.1f}', ha='center', fontsize=10)
        
        # R² comparison
        r2_means = [results[m]['r2_mean'] for m in models]
        r2_stds = [results[m]['r2_std'] for m in models]
        
        axes[1].bar(range(len(models)), r2_means, yerr=r2_stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=15, ha='right')
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Add values on bars
        for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
            axes[1].text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=10)

        plt.tight_layout()

        plt_filename = 'model_comparison.png'
        if results_dir is not None:
            plt_filename = os.path.join(results_dir, plt_filename)            

        plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plt_filename}")


    def plot_feature_importance(self, results_dir=None):
        """Plot feature importance for each model."""
        # Feature importance for each model
        for model_name, model in self.models.items():
             # If it's a tree-based model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            # If it's a pipeline with a tree-based model as last step
            elif isinstance(model, Pipeline) and hasattr(model.steps[-1][1], 'feature_importances_'):
                importances = model.steps[-1][1].feature_importances_
            else:
                print(f"Skipping feature importance for {model_name} (not supported)")
                continue

            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[indices], color='#1f77b4')
            plt.xticks(range(len(importances)), [self.feature_columns[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Feature Importance', fontsize=12)
            plt.title(f'{model_name} Feature Importance', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            filename = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_feature_importance.png"
            if results_dir is not None:
                filename = os.path.join(results_dir, filename)

            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            plt.close('all')
