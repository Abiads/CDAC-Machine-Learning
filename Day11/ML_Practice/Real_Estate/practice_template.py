"""
================================================================================
EXAM PRACTICE: Day 11 - ML Practice 5
DOMAIN: Real Estate Valuation & Regression Pipeline (Linear Regression)
================================================================================
PROBLEM STATEMENT (PS):
A property analytics platform requires an automated appraisal pipeline to forecast 
residential home sale prices (price_usd) from structural attributes and geographic 
location types (Urban, Suburban, Rural). The training dataset contains 25 verified 
recent home transactions.

VALUATION QUESTIONS TO SOLVE:
1. How do you engineer a structural room density metric (sqft / (bedrooms + bathrooms))?
2. How do you convert the categorical 'location_type' feature into one-hot numeric columns?
3. How do you split the dataset 80/20 and apply StandardScaler without leaking test statistics?
4. How do you fit an ordinary least squares Linear Regression model and compute R-squared, 
   Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)?

DATASET FILE: housing_ml_data.csv
AVAILABLE COLUMNS:
  - house_id (identifier)
  - sqft, bedrooms, bathrooms, age_years, garage_spaces, location_type
  - price_usd (continuous target)
================================================================================
"""

from pathlib import Path
# TODO: Import required libraries (Numpy, Pandas, train_test_split, StandardScaler, LinearRegression, metrics, joblib)

csv_path = Path(__file__).resolve().parent / "housing_ml_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    - Ingest housing_ml_data.csv.
    - Inspect dimensions and isolate feature matrix X (drop house_id and price_usd) and target y.
    - Return X, y.
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Spatial Feature Engineering & Encoding
    - Safely copy X.
    - Engineer 'sqft_per_room' = sqft / (bedrooms + bathrooms).
    - One-hot encode 'location_type' using pd.get_dummies(..., drop_first=True).
    - Return transformed X_encoded.
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    - Split data 80/20 with random_state=42.
    - Standardize features with StandardScaler (fit on train, transform test).
    - Return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Multi-Metric Regression Evaluation
    - Fit LinearRegression on X_train_scaled.
    - Predict on X_test_scaled.
    - Compute and print R-squared score, MAE ($), and RMSE ($).
    - Return the trained model.
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    # TODO: Write your implementation here
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
