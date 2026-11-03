"""
================================================================================
PRACTICE TEMPLATE: Day 11 - ML Practice 5 (Real Estate Valuation Pipeline)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: housing_ml_data.csv
TARGET:  'price_usd'
KEY TASKS:
1. Feature Engineering: sqft_per_room = sqft / (bedrooms + bathrooms)
2. Categorical Encoding: pd.get_dummies(..., drop_first=True) on 'location_type'
3. Train/Test Split (80/20) and StandardScaler on train
4. LinearRegression fit, evaluate R2, MAE, RMSE
5. Persist model & scaler with joblib
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import joblib

csv_path = Path(__file__).resolve().parent / "housing_ml_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. X = df.drop(columns=['house_id', 'price_usd'])
    3. y = df['price_usd']
    4. Return X, y!
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print(f"Dataset Shape: {df.shape}")
    # X = df.drop(columns=['house_id', 'price_usd'])
    # y = df['price_usd']
    # return X, y
    pass


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Feature Engineering & Encoding
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X = X.copy()
    2. X['sqft_per_room'] = X['sqft'] / (X['bedrooms'] + X['bathrooms'])
    3. X_encoded = pd.get_dummies(X, drop_first=True)
    4. Return X_encoded!
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = X.copy()
    # X['sqft_per_room'] = X['sqft'] / (X['bedrooms'] + X['bathrooms'])
    # X_encoded = pd.get_dummies(X, drop_first=True)
    # return X_encoded
    pass


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    2. scaler = StandardScaler()
    3. X_tr_scaled = scaler.fit_transform(X_tr)
    4. X_te_scaled = scaler.transform(X_te)
    5. Return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler!
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    # scaler = StandardScaler()
    # X_tr_scaled = scaler.fit_transform(X_tr)
    # X_te_scaled = scaler.transform(X_te)
    # return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler
    pass


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Model Induction with Linear Regression
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. model = LinearRegression().fit(X_train_scaled, y_train)
    2. y_pred = model.predict(X_test_scaled)
    3. r2 = model.score(X_test_scaled, y_test)
    4. mae = mean_absolute_error(y_test, y_pred)
    5. rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    6. Return model!
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # model = LinearRegression().fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)
    # r2 = model.score(X_test_scaled, y_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"R-squared Score: {r2:.4f}")
    # print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    # print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    # return model
    pass


def run_housing_pipeline():
    print("=" * 65)
    print("STARTING HOUSING VALUATION PIPELINE (Linear Regression)")
    print("=" * 65)
    
    # X, y = phase_1_discovery_and_target_identification()
    # X_enc = phase_2_feature_engineering_and_encoding(X)
    # X_tr_s, X_te_s, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_enc, y)
    # model = phase_4_model_training_and_testing(X_tr_s, X_te_s, y_tr, y_te)


if __name__ == "__main__":
    run_housing_pipeline()
