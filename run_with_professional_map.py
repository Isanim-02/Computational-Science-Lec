"""
Run Philippines Rainfall Prediction with PROFESSIONAL MAPS
Enhanced styling similar to official Philippines regional maps
"""

from rainfall_prediction_svr_with_hourly import PhilippinesRainfallPredictorEnhanced
from philippines_professional_map import create_professional_visualizations

def main():
    """
    Run complete analysis with professional Philippines maps
    """
    print("\n" + "="*70)
    print("PHILIPPINES RAINFALL PREDICTION")
    print("PROFESSIONAL MAPS with Regional Boundaries")
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
    
    # Professional maps
    print("\nCreating PROFESSIONAL maps...")
    create_professional_visualizations(predictor)
    
    # Summary
    predictor.print_summary_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE WITH PROFESSIONAL MAPS!")
    print("="*70)
    print("\nGenerated files:")
    print("\nPerformance Plots:")
    print("  - svr_kernel_comparison_enhanced.png")
    print("  - svr_fold_details_enhanced.png")
    print("\nPROFESSIONAL Maps:")
    print("  - philippines_professional_kriging_rbf.png")
    print("  - philippines_professional_kriging_polynomial.png")
    print("  - philippines_professional_kriging_sigmoid.png")
    print("\nMaps include:")
    print("  - Regional boundaries (Luzon, Visayas, Mindanao)")
    print("  - Region labels & colors")
    print("  - Major cities labeled")
    print("  - Compass rose")
    print("  - Professional styling")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

