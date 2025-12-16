# 📊 Data Setup Instructions

## Large File Not Included

The `hourly_data_combined_2020_to_2023.csv` file (1.15 GB) is **not included** in this repository because it exceeds GitHub's 100 MB file size limit.

## ✅ How to Run the App

### Option 1: Without Hourly Data (Lower Accuracy)
The app will work without the hourly file, but with estimated humidity/pressure values and lower accuracy.

Just run:
```bash
streamlit run streamlit_rainfall_app.py
```

### Option 2: With Hourly Data (Best Accuracy - Recommended)
For best results (RMSE: 69.96mm), you need the hourly data:

1. **Download the hourly CSV file** from:
   - Your original data source
   - Or contact the project team

2. **Place it in the project root:**
   ```
   Computational-Science-Lec/
   ├── hourly_data_combined_2020_to_2023.csv  ← Add here
   ├── streamlit_rainfall_app.py
   └── ...
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_rainfall_app.py
   ```

## 📈 Performance Difference

| Mode | RMSE | R² | Cities |
|------|------|-----|--------|
| **With Hourly Data** | 69.96 mm | 0.78 | 133 |
| **Without Hourly Data** | ~120+ mm | ~0.45 | Variable |

**Recommendation:** Use hourly data for production/research purposes!

## 🔧 Technical Details

The hourly file provides:
- **Real humidity data** (relative_humidity_2m)
- **Real air pressure data** (surface_pressure)
- **Complete 2020-2023 coverage**

Without it, the app estimates these values with reduced accuracy.

---

**Need help?** Check the repository issues or contact the maintainers.

