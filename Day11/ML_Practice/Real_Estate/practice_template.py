"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - ML Practice 5
DOMAIN: Real Estate Valuation & Regression Pipeline (Linear Regression)
================================================================================
PROBLEM STATEMENT (PS):
Forecast residential property prices (price_usd) using an end-to-end regression
pipeline. Engineer sqft_per_room, one-hot encode location_type, split 80/20, 
scale features using StandardScaler, fit Linear Regression, and evaluate 
R-squared, MAE, and RMSE.

Fill in all the `___` blanks below to complete the implementation.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "housing_ml_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(csv_path)
    
    # 1. Drop identifier and target
    X = df.drop(columns=['house_id', 'price_usd'])
    
    # 2. Isolate continuous price target
    y = df['price_usd']
    
    return X, y                     # 3. Return X, y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Spatial Feature Engineering & Encoding
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    X = X.copy()
    
    # 4. Compute sqft per room
    X['sqft_per_room'] = X['square_feet'] / X['num_bedrooms']
    
    # 5. One-hot encode location_type
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded                          # 6. Return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 7. 80/20 train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 8. Fit on train, transform on test
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Multi-Metric Regression Evaluation
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 9. Fit LinearRegression
    model = LinearRegression().fit(X_train_scaled, y_train)
    
    # 10. Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # 11. Compute metrics
    r2 = model.score(X_test_scaled, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    
    return model                          # 12. Return model


def run_housing_pipeline():
    print("=" * 65)
    print("STARTING HOUSING VALUATION PIPELINE (Linear Regression)")
    print("=" * 65)
    
    X, y = phase_1_discovery_and_target_identification()
    X_enc = phase_2_feature_engineering_and_encoding(X)
    X_tr_s, X_te_s, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_enc, y)
    model = phase_4_model_training_and_testing(X_tr_s, X_te_s, y_tr, y_te)


if __name__ == "__main__":
    run_housing_pipeline()
