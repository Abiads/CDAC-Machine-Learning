"""
Title: Supervised Classification Pipeline with Random Forest 
Description: Simple, built-in function driven Python script for customer churn prediction.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

csv_path = Path(__file__).resolve().parent / "customer_churn_data.csv"


def phase_1_discovery_and_target_identification():
    """Phase 1: Ingest churn data and separate target from predictors."""
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape}")
    print("Class Balance (churn_flag):\n", df['churn_flag'].value_counts())
    X = df.drop(columns=['customer_id', 'churn_flag'])
    y = df['churn_flag']
    return X, y


def phase_2_feature_engineering_and_encoding(X):
    """Phase 2: Calculate average monthly spend and encode categorical columns."""
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    X = X.copy()
    X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months']
    X_encoded = pd.get_dummies(X, drop_first=True)
    print("Transformed Feature Matrix Head:\n", X_encoded.head(2))
    return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """Phase 3: Stratified 80/20 train-test split and standard scaling."""
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """Phase 4: Train Random Forest classifier and evaluate test metrics."""
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """Phase 5: Serialize Random Forest pipeline artifacts and test sample inference."""
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "churn_rf_model.joblib")
    joblib.dump(scaler, base_dir / "churn_scaler.joblib")
    print("Artifacts saved: churn_rf_model.joblib, churn_scaler.joblib")

    # Sample inference
    sample_profile = [[2, 70.0, 140.0, 3, 70.0, False, False, True]]
    sample_scaled = scaler.transform(sample_profile)
    pred = model.predict(sample_scaled)
    print(f"Sample Subscriber Churn Prediction (0=Retain, 1=Churn Risk): {pred[0]}")


def run_ml_pipeline():
    """Facade function coordinating the entire classification pipeline."""
    print("=" * 65)
    print("STARTING SUPERVISED CLASSIFICATION LIFECYCLE (Random Forest)")
    print("=" * 65)

    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())

    print("\n" + "=" * 65)
    print("CUSTOMER CHURN PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_ml_pipeline()
