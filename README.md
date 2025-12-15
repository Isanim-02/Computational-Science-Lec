# 🌧️ Philippines Rainfall Prediction System

A comprehensive rainfall prediction and forecasting system for the Philippines using Support Vector Regression (SVR) with machine learning.

## 🎯 Features

### 📜 Historical Analysis (2020-2023)
- Analyze actual rainfall data from 2020-2023
- Compare predictions with actual measurements
- Evaluate model performance with RMSE
- View 141 Philippine cities

### 🔮 Future Forecasts (2024-2030)
- Predict rainfall for years 2024-2030
- Three climate scenarios:
  - ⚪ **Neutral** - Normal conditions
  - 🔴 **El Niño** - Warmer/Drier conditions
  - 🔵 **La Niña** - Cooler/Wetter conditions
- Scenario comparison charts
- Uncertainty quantification

### 🗺️ Interactive Visualizations
- Interactive maps with Plotly
- City-specific predictions
- Scenario comparisons
- Monthly trends

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required data files:
  - `daily_data_combined_2020_to_2023.csv`
  - `cities.csv`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the App

**Windows:**
```bash
# Double-click START_DYNAMIC_APP.bat
# OR run:
streamlit run streamlit_rainfall_app.py
```

**Linux/Mac:**
```bash
# Run the launcher:
./start_dynamic_app.sh
# OR directly:
streamlit run streamlit_rainfall_app.py
```

The app will automatically open in your web browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Choose Mode
- **📜 Historical (2020-2023)** - Analyze past data
- **🔮 Forecast (2024-2030)** - Predict future rainfall

### Step 2: Select Parameters

**Historical Mode:**
- Year: 2020-2023
- Month: January-December
- Kernel: RBF, Polynomial, or Sigmoid
- Map Type: Scatter or Density Heatmap

**Forecast Mode:**
- Year: 2024-2030
- Month: January-December
- Climate Scenario: Neutral, El Niño, or La Niña
- Kernel: RBF, Polynomial, or Sigmoid
- Map Type: Scatter or Density Heatmap

### Step 3: Explore Results
- View interactive map
- Check metrics (cities, RMSE, rainfall)
- Compare scenarios
- Analyze trends

## 📊 Technical Details

### Machine Learning Model
- **Algorithm:** Support Vector Regression (SVR)
- **Kernels:** RBF, Polynomial, Sigmoid
- **Features:** Month, Location, Temperature, Humidity, Pressure, ENSO Index
- **Target:** Monthly Rainfall (mm)

### Forecasting Approach
**Two-Stage Process:**

1. **Stage 1:** Forecast meteorological features
   - Temperature (with warming trend)
   - Humidity (temperature-adjusted)
   - Air Pressure (historical average)
   - ENSO Index (scenario-based)

2. **Stage 2:** Apply trained SVR model
   - Use forecasted features → Predict rainfall

### Climate Scenarios
- **Neutral:** ONI Index ≈ 0 (normal conditions)
- **El Niño:** ONI Index > +0.5 (warmer, drier)
- **La Niña:** ONI Index < -0.5 (cooler, wetter)

## 📁 Project Structure

```
project/
├── streamlit_rainfall_app.py          # Main web application
├── rainfall_forecast_module.py        # Forecasting engine
├── rainfall_prediction_svr_with_hourly.py  # Core ML model
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── START_DYNAMIC_APP.bat             # Windows launcher
├── start_dynamic_app.sh              # Linux/Mac launcher
├── daily_data_combined_2020_to_2023.csv  # Historical data
└── cities.csv                        # City coordinates
```

## 📈 Example Use Cases

### 1. Agricultural Planning
Check rainfall forecasts for planting season to select appropriate crops.

### 2. Water Resource Management
Predict dry season rainfall to plan reservoir levels.

### 3. Flood Risk Assessment
Evaluate La Niña scenario during monsoon season for flood preparedness.

### 4. Infrastructure Planning
Design stormwater systems based on maximum rainfall across all scenarios.

## ⚠️ Important Notes

### Forecast Limitations
- Forecasts are **projections**, not exact predictions
- Uncertainty **increases** with time horizon
- Based on **2020-2023 patterns** (may not hold if climate shifts)
- **Scenario-based** (actual ENSO phase unknown)

### Best Practices
- Use forecast **ranges** (compare all 3 scenarios)
- Update forecasts **annually** with new data
- Combine with **expert knowledge**
- Plan for **multiple scenarios**

## 🔧 Dependencies

Core libraries:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `plotly` - Interactive visualizations

See `requirements.txt` for complete list.

## 📝 Data Requirements

### Input Data Format

**Daily Weather Data:**
- Columns: `city_name`, `datetime`, `temperature_2m_mean`, `precipitation_sum`
- Optional: `relative_humidity_2m`, `pressure_msl`

**Cities Data:**
- Columns: `city_name`, `latitude`, `longitude`

## 🎓 Citation

If you use this system in your research, please cite:

```
Philippines Rainfall Prediction System (2024)
Support Vector Regression with Climate Scenarios
Historical Analysis: 2020-2023 | Forecasts: 2024-2030
```

## 📞 Support

For questions or issues:
1. Check the in-app help text
2. Review this README
3. Examine the code comments

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_rainfall_app.py

# Stop the app
# Press Ctrl+C in the terminal
```

## 🎉 Summary

**What You Get:**
- ✅ Historical analysis tool (2020-2023)
- ✅ Future forecasting system (2024-2030)
- ✅ Three climate scenarios
- ✅ Interactive web interface
- ✅ All 141 Philippine cities
- ✅ Monthly predictions
- ✅ Easy to use

**Perfect For:**
- Agricultural planning
- Water resource management
- Climate research
- Policy making
- Infrastructure planning

---

**Ready to explore? Run the app now!** 🚀

```bash
streamlit run streamlit_rainfall_app.py
```
