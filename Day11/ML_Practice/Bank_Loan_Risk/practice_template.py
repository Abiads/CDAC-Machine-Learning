"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - ML Practice 1
DOMAIN: Banking Credit Risk (Supervised Classification Pipeline)
================================================================================
PROBLEM STATEMENT (PS):
A commercial bank requires an automated credit evaluation pipeline to classify
loan applicants into low-risk (0) and high-risk default (1).
You must engineer a debt ratio, one-hot encode housing status, perform stratified 
splitting, scale features without data leakage, fit Logistic Regression, 
and persist the pipeline.

Fill in all the `___` blanks below to complete the implementation.
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "loan_risk_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    - Ingest loan_risk_data.csv.
    - Separate feature matrix X (drop applicant_id and default_flag) and target y.
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape}")
    print("Class Balance:\n", df['default_flag'].value_counts())
    
    # 1. Drop non-predictive ID and the target column
    X = df.drop(columns=['applicant_id', 'default_flag'])
    
    # 2. Isolate the target column
    y = df['default_flag']
    
    return X, y                     # 3. Return X, y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Feature Engineering & Categorical Encoding
    - Construct loan_to_income_ratio and one-hot encode.
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 4. Prevent SettingWithCopyWarning
    X = X.copy()
    
    # 5. Calculate loan to income ratio
    X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']
    
    # 6. Apply get_dummies dropping first level
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded                          # 7. Return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    - Stratified train/test split and StandardScaler.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 8. Train/test split with stratification on target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 9. Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    
    # 10. Transform test data ONLY
    X_te_scaled = scaler.transform(X_te)
    
    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Model Induction with Logistic Regression
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 11. Fit LogisticRegression
    model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    
    # 12. Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    return model                          # 13. Return model


def phase_5_model_persistence_and_inference(model, scaler, feature_columns):
    """
    [Phase 5] Persistence with Joblib
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "loan_risk_model.joblib")
    joblib.dump(scaler, base_dir / "loan_risk_scaler.joblib")
    print("Saved model and scaler artifacts.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_ml_pipeline():
    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE (Logistic Regression)")
    print("=" * 65)
    
    X, y = phase_1_discovery_and_target_identification()
    X_enc = phase_2_feature_engineering_and_encoding(X)
    X_tr_s, X_te_s, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_enc, y)
    model = phase_4_model_training_and_testing(X_tr_s, X_te_s, y_tr, y_te)
    phase_5_model_persistence_and_inference(model, scaler, X_enc.columns.tolist())


if __name__ == "__main__":
    run_ml_pipeline()
