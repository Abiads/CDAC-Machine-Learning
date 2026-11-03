"""
================================================================================
PRACTICE TEMPLATE: Day 11 - ML Practice 3 (Customer Churn Random Forest)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: customer_churn_data.csv
TARGET:  'churn_flag' (0 = Active, 1 = Churned)
MODEL:   RandomForestClassifier(n_estimators=50, random_state=42)
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import joblib

csv_path = Path(__file__).resolve().parent / "customer_churn_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. X = df.drop(columns=['customer_id', 'churn_flag'])
    3. y = df['churn_flag']
    4. Return X, y!
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print("Class Balance:\n", df['churn_flag'].value_counts())
    # X = df.drop(columns=['customer_id', 'churn_flag'])
    # y = df['churn_flag']
    # return X, y
    pass


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Feature Engineering & Encoding
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X = X.copy()
    2. TRAP: Use .replace(0, 1) on tenure_months to avoid division by zero!
       X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'].replace(0, 1)
    3. X_encoded = pd.get_dummies(X, drop_first=True)
    4. Return X_encoded!
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = X.copy()
    # X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'].replace(0, 1)
    # X_encoded = pd.get_dummies(X, drop_first=True)
    # return X_encoded
    pass


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Splitting & Scaling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    2. scaler = StandardScaler()
    3. X_tr_scaled = scaler.fit_transform(X_tr)
    4. X_te_scaled = scaler.transform(X_te)
    5. Return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler!
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
    [Phase 4] Model Induction with Random Forest
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_scaled, y_train)
    2. y_pred = model.predict(X_test_scaled)
    3. Evaluate accuracy, confusion matrix, and classification report.
    4. Return model!
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    # return model
    pass


def run_churn_pipeline():
    print("=" * 65)
    print("STARTING SUPERVISED CLASSIFICATION LIFECYCLE (Random Forest)")
    print("=" * 65)
    
    # X, y = phase_1_discovery_and_target_identification()
    # X_enc = phase_2_feature_engineering_and_encoding(X)
    # X_tr_s, X_te_s, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_enc, y)
    # model = phase_4_model_training_and_testing(X_tr_s, X_te_s, y_tr, y_te)


if __name__ == "__main__":
    run_churn_pipeline()
