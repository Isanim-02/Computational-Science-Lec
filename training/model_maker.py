"""
    Trains the final models and then stores them in the models folder.
"""

from rainfall_predictor import RainfallPredictor

MODELS_DIR = 'models'

if __name__ == "__main__":
    predictor = RainfallPredictor(monthly_data_path='datasets/monthly.csv')
    predictor.train_final_models(model_output_dir=MODELS_DIR)
    print(f"Trained models saved to {MODELS_DIR}")