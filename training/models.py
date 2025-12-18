"""
    Contains the machine learning models to be tested for rainfall prediction.
"""

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS_TO_TEST = {
    'XGBoost (Optimized)': XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        # max_depth=6,
        # min_child_weight=3,
        subsample=0.9,
        reg_lambda=2,
        reg_alpha=0.01,
        min_child_weight=7,
        max_depth=4,
        gamma=0.1,
        colsample_bytree=0.9,

        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        # n_estimators=200,
        # learning_rate=0.1,
        # max_depth=5,
        # min_samples_split=5,
        subsample=0.7,
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=None,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ),
    'XGBoost (Fast)': XGBRegressor(
        n_estimators=200,
        learning_rate=0.15,
        # max_depth=5,
        subsample=0.9,
        reg_lambda=2,
        reg_alpha=0.01,
        min_child_weight=7,
        max_depth=4,
        gamma=0.1,
        colsample_bytree=0.9,

        random_state=42,
        n_jobs=-1
    ),
    'SVR (RBF)': make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf')
    ),
    'SVR (POLY)': make_pipeline(
        StandardScaler(),  
            SVR(kernel='poly')
    ),
}