# 🌧️ Philippines Rainfall Prediction Model

A comprehensive machine learning system for predicting monthly rainfall in the Philippines using **Support Vector Regression (SVR)** with multiple kernel functions.

## 📋 Project Overview

This project implements a predictive rainfall model that analyzes historical weather data from 2020-2023 across multiple Philippine cities. The model uses Support Vector Regression with three different kernel tricks to predict monthly rainfall based on geographic, meteorological, and climate anomaly features.

## 🎯 Features

### Machine Learning Models
- **Support Vector Regression (SVR)** with three kernel types:
  - **Radial Basis Function (RBF)** - Captures non-linear patterns
  - **Polynomial** - Models polynomial relationships
  - **Sigmoid** - Neural network-like transformations

### Input Features (9 variables)
1. **Month** - Seasonal patterns
2. **Latitude** - Geographic location (North-South)
3. **Longitude** - Geographic location (East-West)
4. **Temperature** - Monthly average temperature (°C)
5. **Humidity Proxy** - Apparent temperature indicator
6. **Air Pressure Proxy** - Evapotranspiration indicator
7. **ONI Index** - Oceanic Niño Index (El Niño/La Niña strength)
8. **El Niño Indicator** - Binary flag for El Niño conditions
9. **La Niña Indicator** - Binary flag for La Niña conditions

### Target Variable
- **Monthly Rainfall** - Total precipitation in millimeters (mm)

### Evaluation Metrics
- **RMSE (Root Mean Square Error)** - Prediction accuracy in mm
- **R² (Coefficient of Determination)** - Model fit quality (0-1)
- **K-Fold Cross-Validation** - Robust performance assessment (k=5)

### Spatial Analysis
- **Ordinary Kriging** - Geostatistical interpolation
- **Variance Mapping** - Uncertainty quantification
- **Gap Visualization** - Identifying prediction gaps

## 📊 Project Structure

```
CS Lec/
│
├── rainfall_prediction_svr.py      # Main prediction script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── cities.csv                       # City coordinates (lat/lon)
├── daily_data_combined_2020_to_2023.csv  # Historical weather data
│
├── Frigillana_Vidal_Villamor_-_CSPE001_-_FA1.pdf  # Project guide
└── NOAA ONI Table Data Retrieval.pdf               # ENSO data reference
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files are present:**
   - `cities.csv`
   - `daily_data_combined_2020_to_2023.csv`

## 💻 Usage

### Basic Execution

Run the complete pipeline:

```bash
python rainfall_prediction_svr.py
```

### What the Script Does

1. **Loads Data** - Reads city coordinates and daily weather records
2. **Preprocesses** - Aggregates daily data to monthly level
3. **Adds ENSO Indices** - Incorporates El Niño/La Niña indicators
4. **Prepares Features** - Standardizes input variables
5. **Trains Models** - Fits SVR with RBF, Polynomial, and Sigmoid kernels
6. **Cross-Validates** - 5-fold validation with RMSE and R² metrics
7. **Visualizes** - Generates performance comparison plots
8. **Spatial Analysis** - Creates Kriging interpolation maps

### Output Files

After execution, the following files will be generated:

1. **svr_kernel_comparison.png**
   - Bar chart comparing RMSE and R² across kernels
   - Shows mean values with error bars

2. **svr_fold_details.png**
   - Fold-by-fold performance visualization
   - RMSE and R² trends across validation folds

3. **kriging_interpolation_rbf.png**
   - Spatial interpolation map using Kriging
   - Uncertainty/variance visualization

## 📈 Expected Results

### Typical Performance Metrics

| Kernel       | RMSE (mm)      | R² Score       |
|-------------|----------------|----------------|
| **RBF**     | ~50-80         | ~0.60-0.75     |
| **Polynomial** | ~55-85      | ~0.55-0.70     |
| **Sigmoid** | ~60-90         | ~0.50-0.65     |

*Note: Actual values depend on data quality and sample size*

### Interpretation

- **Lower RMSE** = Better prediction accuracy
- **Higher R²** = Better model fit (1.0 = perfect, 0.0 = baseline)
- **RBF kernel** typically performs best for rainfall data

## 🔧 Customization

### Adjust Sample Size

For faster testing, modify `main()` function:

```python
# Test with smaller sample
predictor.load_data(sample_size=50000)

# Use full dataset (slower but more accurate)
predictor.load_data(sample_size=None)
```

### Change Cross-Validation Folds

```python
# Use 10-fold cross-validation
predictor.evaluate_with_kfold(n_splits=10)
```

### Modify SVR Parameters

```python
custom_params = {
    'rbf': {'kernel': 'rbf', 'C': 200, 'gamma': 'auto', 'epsilon': 0.2},
    'poly': {'kernel': 'poly', 'C': 150, 'degree': 4, 'gamma': 'scale', 'epsilon': 0.15},
    'sigmoid': {'kernel': 'sigmoid', 'C': 150, 'gamma': 'scale', 'epsilon': 0.15}
}
predictor.train_svr_models(kernel_params=custom_params)
```

### Update ONI Indices

Edit the `oni_data` dictionary in `add_enso_indices()` method with actual NOAA data from:
https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php

## 🌏 About El Niño/La Niña

### ENSO Impacts on Philippines Rainfall

- **El Niño** (ONI > +0.5): Warmer Pacific → Typically DRIER conditions
- **La Niña** (ONI < -0.5): Cooler Pacific → Typically WETTER conditions
- **Neutral** (-0.5 to +0.5): Normal rainfall patterns

The model incorporates these climate anomalies to improve prediction accuracy.

## 📊 Spatial Interpolation (Kriging)

### What is Kriging?

Ordinary Kriging is a geostatistical technique that:
- Estimates values at unmeasured locations
- Provides uncertainty quantification
- Respects spatial autocorrelation
- Creates smooth, continuous surfaces

### Interpreting Kriging Results

1. **Rainfall Map**: Shows predicted rainfall across the Philippines
   - Blue areas = Lower rainfall
   - Yellow/Green areas = Higher rainfall
   - Black dots = Actual measurement locations

2. **Variance Map**: Indicates prediction uncertainty
   - Low variance = High confidence
   - High variance = Gaps in data coverage (need more stations)

## 🐛 Troubleshooting

### Memory Issues

If you encounter memory errors:
```python
predictor.load_data(sample_size=50000)  # Reduce sample size
```

### PyKrige Installation Issues

If Kriging fails to install:
```bash
# Windows
pip install pykrige --no-cache-dir

# Linux/Mac
pip install pykrige
```

Alternative: The script will generate simple spatial plots without Kriging.

### Missing Data Warnings

The script automatically handles missing values by:
- Dropping rows with NaN values
- Reporting the final dataset size

## 📚 References

### Data Sources
- **Weather Data**: Open-Meteo Historical Weather API
- **ENSO Indices**: NOAA Climate Prediction Center ONI
- **City Coordinates**: OpenStreetMap / GeoNames

### Scientific Background
- Support Vector Machines: Vapnik, V. (1995). "The Nature of Statistical Learning Theory"
- Kriging: Matheron, G. (1963). "Principles of Geostatistics"
- ENSO Impacts: Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA)

## 👥 Contributors

- **Project**: CSPE001 Final Assessment
- **Students**: Frigillana, Vidal, Villamor
- **Course**: Computer Science & Python Engineering

## 📝 License

This project is for educational purposes as part of CSPE001 coursework.

## 🤝 Acknowledgments

- NOAA Climate Prediction Center for ENSO data
- Open-Meteo for weather data access
- Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA)
- Scikit-learn and PyKrige development teams

---

## 🔮 Future Enhancements

Potential improvements for future iterations:

1. **Additional Features**
   - Wind patterns and direction
   - Sea surface temperatures
   - Topographic elevation

2. **Advanced Models**
   - Ensemble methods (Random Forest, Gradient Boosting)
   - Deep learning (LSTM for temporal patterns)
   - Hybrid SVR-Neural Network models

3. **Real-Time Predictions**
   - API integration for live weather data
   - Automated monthly forecasting
   - Web dashboard for visualization

4. **Higher Resolution**
   - Hourly rainfall predictions
   - Sub-regional analysis
   - Typhoon season modeling

---

**Last Updated**: December 2024

For questions or issues, please refer to the project documentation or contact the course instructor.

