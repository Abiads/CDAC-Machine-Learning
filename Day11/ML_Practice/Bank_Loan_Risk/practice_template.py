"""
================================================================================
PRACTICE TEMPLATE: Day 11 - ML Practice 1 (Bank Loan Default Risk Pipeline)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: loan_risk_data.csv
TARGET:  'default_flag' (0 = Approved, 1 = Default Risk)
KEY TASKS:
1. Feature Engineering: loan_to_income_ratio = loan_amount_k / annual_income_k
2. Categorical Encoding: pd.get_dummies(..., drop_first=True)
3. Stratified Split: train_test_split(..., stratify=y)
4. Fit StandardScaler on TRAIN ONLY, transform test
5. Fit LogisticRegression, evaluate accuracy, confusion matrix, classification_report
6. Persist model & scaler with joblib
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import joblib

csv_path = Path(__file__).resolve().parent / "loan_risk_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. Drop applicant_id and target:
       X = df.drop(columns=['applicant_id', 'default_flag'])
       y = df['default_flag']
    3. Return X, y!
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print(f"Dataset Shape: {df.shape}")
    # print("Class Balance:\n", df['default_flag'].value_counts())
    # X = df.drop(columns=['applicant_id', 'default_flag'])
    # y = df['default_flag']
    # return X, y
    pass


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Feature Engineering & Encoding
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. TRAP: Use X = X.copy() to prevent SettingWithCopyWarning
    2. Ratio: X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']
    3. Encoding: X_encoded = pd.get_dummies(X, drop_first=True)
    4. Return X_encoded!
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = X.copy()
    # X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']
    # X_encoded = pd.get_dummies(X, drop_first=True)
    # return X_encoded
    pass


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. TRAP: Always pass stratify=y on small datasets!
       X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    2. TRAP: fit_transform ON TRAIN, transform ON TEST ONLY!
       scaler = StandardScaler()
       X_tr_scaled = scaler.fit_transform(X_tr)
       X_te_scaled = scaler.transform(X_te)
    3. Return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler!
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # scaler = StandardScaler()
    # X_tr_scaled = scaler.fit_transform(X_tr)
    # X_te_scaled = scaler.transform(X_te)
    # return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler
    pass


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Model Training & Test Set Evaluation
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    2. y_pred = model.predict(X_test_scaled)
    3. print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    4. print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    5. print("Report:\n", classification_report(y_test, y_pred, zero_division=0))
    6. Return model!
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    # return model
    pass


def phase_5_model_persistence_and_inference(model, scaler, feature_columns):
    """
    [Phase 5] Persistence with Joblib & Sample Inference
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. joblib.dump(model, "loan_risk_model.joblib")
    2. joblib.dump(scaler, "loan_risk_scaler.joblib")
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # base_dir = Path(__file__).resolve().parent
    # joblib.dump(model, base_dir / "loan_risk_model.joblib")
    # joblib.dump(scaler, base_dir / "loan_risk_scaler.joblib")
    # print("Saved loan_risk_model.joblib and loan_risk_scaler.joblib")
    pass


def run_ml_pipeline():
    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE (Logistic Regression)")
    print("=" * 65)
    
    # X, y = phase_1_discovery_and_target_identification()
    # X_enc = phase_2_feature_engineering_and_encoding(X)
    # X_tr_s, X_te_s, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_enc, y)
    # model = phase_4_model_training_and_testing(X_tr_s, X_te_s, y_tr, y_te)
    # phase_5_model_persistence_and_inference(model, scaler, X_enc.columns.tolist())


if __name__ == "__main__":
    run_ml_pipeline()
