"""
Title: Supervised Classification Pipeline for Loan Default Risk 
Description: Simple, built-in function driven Python script for loan default prediction.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

csv_path = Path(__file__).resolve().parent / "loan_risk_data.csv"


def phase_1_discovery_and_target_identification():
    """Phase 1: Ingest loan applicant data and separate target from predictors."""
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape}")
    print("Class Distribution (default_flag):\n", df['default_flag'].value_counts())
    X = df.drop(columns=['applicant_id', 'default_flag'])
    y = df['default_flag']
    return X, y


def phase_2_feature_engineering_and_encoding(X):
    """Phase 2: Construct loan-to-income debt ratio and one-hot encode housing status."""
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    X = X.copy()
    X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']
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
    """Phase 4: Train Logistic Regression model and evaluate on unseen test data."""
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    base_dir = Path(__file__).resolve().parent
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Approved (0)', 'Default (1)'],
                yticklabels=['Approved (0)', 'Default (1)'])
    plt.title("Confusion Matrix - Loan Default Risk")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(base_dir / "loan_risk_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved visualization: 'loan_risk_confusion_matrix.png'")
    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """Phase 5: Serialize model and scaler artifacts and test sample inference."""
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "loan_risk_model.joblib")
    joblib.dump(scaler, base_dir / "loan_risk_scaler.joblib")
    print("Artifacts saved: loan_risk_model.joblib, loan_risk_scaler.joblib")

    # Verify reload & test inference
    loaded_model = joblib.load(base_dir / "loan_risk_model.joblib")
    sample_pred = loaded_model.predict([[30, 50, 650, 4, 20, 0.4, False, True]])
    print(f"Sample Applicant Inference (0=Approved, 1=Default Risk): {sample_pred[0]}")


def run_ml_pipeline():
    """Facade function coordinating the entire loan risk classification pipeline."""
    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE (Logistic Regression)")
    print("=" * 65)

    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())

    print("\n" + "=" * 65)
    print("LOAN RISK LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_ml_pipeline()
