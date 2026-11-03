"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - ML Practice 3
DOMAIN: Telecom Subscriber Retention (Random Forest Classification)
================================================================================
PROBLEM STATEMENT (PS):
Predict customer churn (churn_flag: 0 = Active, 1 = Churned) using an ensemble
Random Forest model. Engineer avg_monthly_spend handling zero tenure accounts, 
one-hot encode categoricals, split 80/20 with stratification, scale features, 
and evaluate test accuracy and confusion matrix.

Fill in all the `___` blanks below to complete the implementation.
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "customer_churn_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Churn Target Audit
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    print("Class Balance:\n", df['churn_flag'].value_counts())
    
    # 1. Drop non-predictive ID and target
    X = df.drop(columns=[___, ___])
    
    # 2. Isolate binary target
    y = df[___]
    
    return ___, ___                     # 3. Return X, y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Ratio Engineering & Categorical Encoding
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    X = X.___()
    
    # 4. Compute avg monthly spend preventing division by zero on tenure_months == 0
    X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'].replace(0, ___)
    
    # 5. One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=___)
    
    return ___                          # 6. Return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Stratified Splitting & Scaling
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 7. Stratified train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=___)
    
    # 8. Fit on train, transform on test
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(___)
    X_te_scaled = scaler.transform(___)
    
    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Ensemble Modeling with Random Forest
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 9. Fit RandomForestClassifier with 50 estimators
    model = RandomForestClassifier(n_estimators=___, random_state=42).fit(___, ___)
    
    # 10. Predict on test set
    y_pred = model.predict(___)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    return ___                          # 11. Return model


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
