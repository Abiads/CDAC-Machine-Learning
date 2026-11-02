"""
Title: End-to-End Supervised Machine Learning Pipeline 
Description: Simple, built-in function driven Python script for real estate price regression.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

csv_path = Path(__file__).resolve().parent / "housing_ml_data.csv"


def phase_1_discovery_and_target_identification():
    """Phase 1: Ingest housing data and separate target price from features."""
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['house_id', 'price_usd'])
    y = df['price_usd']
    print(f"X Shape: {X.shape}, y Shape: {y.shape}")
    print(X.head(2))
    return X, y


def phase_2_feature_engineering_and_encoding(X):
    """Phase 2: Engineer sqft_per_room and one-hot encode location & garage."""
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    X = X.copy()
    X['sqft_per_room'] = X['square_feet'] / X['num_bedrooms']
    X_encoded = pd.get_dummies(X, drop_first=True)
    print("Transformed Features Head:\n", X_encoded.head(2))
    return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """Phase 3: 80/20 train-test split and standard scaling."""
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """Phase 4: Train Linear Regression model and compute MAE, RMSE, and R2."""
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """Phase 5: Serialize regression pipeline artifacts and test sample inference."""
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "housing_model.joblib")
    joblib.dump(scaler, base_dir / "housing_scaler.joblib")
    print("Artifacts saved: housing_model.joblib, housing_scaler.joblib")

    # Sample property inference: 2000 sqft, 3 beds, 10 yrs, 666 sqft/rm, Suburban, Garage=Yes
    sample = [[2000, 3, 10, 666.7, True, False, True]]
    sample_scaled = scaler.transform(sample)
    pred_price = model.predict(sample_scaled)
    print(f"Sample Property Valuation: ${pred_price[0]:,.2f}")


def run_ml_pipeline():
    """Facade function coordinating the entire machine learning pipeline."""
    print("=" * 65)
    print("STARTING REAL ESTATE REGRESSION LIFECYCLE")
    print("=" * 65)

    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())

    print("\n" + "=" * 65)
    print("REAL ESTATE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_ml_pipeline()
