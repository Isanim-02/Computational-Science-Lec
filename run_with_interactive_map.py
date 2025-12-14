"""
Run Philippines Rainfall Prediction with INTERACTIVE MAPS
Creates zoomable, clickable HTML maps using Folium
"""

from rainfall_prediction_svr_with_hourly import PhilippinesRainfallPredictorEnhanced
from philippines_interactive_map import create_interactive_visualizations

def main():
    """
    Run complete analysis with interactive Folium maps
    """
    print("\n" + "="*70)
    print("PHILIPPINES RAINFALL PREDICTION")
    print("INTERACTIVE MAPS (Folium)")
    print("="*70)
    print("\nThis creates HTML maps you can:")
    print("   - Zoom and pan")
    print("   - Click markers for details")
    print("   - Toggle different layers")
    print("   - View in any web browser")
    print("="*70)
    
    # Initialize
    predictor = PhilippinesRainfallPredictorEnhanced(
        daily_data_path='daily_data_combined_2020_to_2023.csv',
        hourly_data_path='hourly_data_combined_2020_to_2023.csv',
        cities_path='cities.csv'
    )
    
    # Load data
    print("\nLoading data...")
    predictor.load_data(
        sample_size_daily=100000,
        sample_size_hourly=1000000
    )
    
    # Process
    print("\nPreprocessing...")
    predictor.preprocess_data()
    predictor.add_enso_indices()
    predictor.prepare_features()
    
    # Train
    print("\nTraining SVR models...")
    predictor.train_svr_models()
    
    # Evaluate
    print("\nCross-validation...")
    predictor.evaluate_with_kfold(n_splits=5)
    
    # Standard plots
    print("\nCreating performance plots...")
    predictor.plot_results()
    
    # Interactive maps
    print("\nCreating INTERACTIVE maps...")
    create_interactive_visualizations(predictor)
    
    # Summary
    predictor.print_summary_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE WITH INTERACTIVE MAPS!")
    print("="*70)
    print("\nGenerated files:")
    print("\nPerformance Plots:")
    print("  - svr_kernel_comparison_enhanced.png")
    print("  - svr_fold_details_enhanced.png")
    print("\nINTERACTIVE HTML Maps:")
    print("  - philippines_interactive_rbf.html")
    print("  - philippines_interactive_polynomial.html")
    print("  - philippines_interactive_sigmoid.html")
    print("  - philippines_kernel_comparison_interactive.html")
    print("\nHow to use:")
    print("  1. Double-click any .html file")
    print("  2. Opens in your web browser")
    print("  3. Zoom, pan, click markers!")
    print("  4. Toggle layers to compare")
    print("\nInteractive maps can be:")
    print("  - Shared with others (just send the HTML file)")
    print("  - Embedded in presentations")
    print("  - Included in reports")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

