"""
================================================================================
EXAM PRACTICE: Day 11 - ML Practice 1
DOMAIN: Banking & Financial Credit Risk (Supervised Classification Pipeline)
================================================================================
PROBLEM STATEMENT (PS):
A commercial lending institution requires an automated credit evaluation pipeline 
to assess borrower default probability before approving unsecured loans. 
You are given historical applicant records comprising financial demographics, 
credit history, housing arrangements, and eventual loan repayment outcomes 
(default_flag: 0 = Paid in full, 1 = Defaulted).

FINANCIAL RISK QUESTIONS TO SOLVE:
1. Which columns represent non-predictive identifiers that must be dropped 
   prior to model training?
2. How can you construct a continuous debt-burden ratio comparing requested 
   loan amount against the applicant's annual income?
3. How do you transform categorical features like 'housing_status' into numeric 
   format without falling into the Dummy Variable Trap?
4. Why is simple random train/test splitting dangerous on small imbalanced datasets, 
   and what parameter guarantees representative class distributions in both sets?
5. How do you prevent data leakage when scaling features across train and test sets?
6. How do you persist the production-ready model and scaler using Joblib?

DATASET FILE: loan_risk_data.csv
AVAILABLE COLUMNS:
  - applicant_id (identifier)
  - age, annual_income_k, loan_amount_k, credit_score, housing_status
  - default_flag (binary target: 0 = Approved, 1 = Default)
================================================================================
"""

from pathlib import Path
# TODO: Import required libraries (Pandas, train_test_split, StandardScaler, LogisticRegression, metrics, joblib)

csv_path = Path(__file__).resolve().parent / "loan_risk_data.csv"


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery & Target Identification
    - Ingest loan_risk_data.csv.
    - Inspect dataset dimensions and target class distribution (default_flag).
    - Separate the target vector y from predictor features X (drop applicant_id and default_flag).
    - Return X, y.
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Feature Engineering & Categorical Encoding
    - Safely copy X to prevent SettingWithCopy warnings.
    - Engineer 'loan_to_income_ratio' = loan_amount_k / annual_income_k.
    - Apply One-Hot Encoding to categorical features, dropping the first dummy level.
    - Return the transformed feature matrix X_encoded.
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Stratified Splitting & Data Leakage Prevention
    - Perform an 80/20 train/test split with stratification on y and random_state=42.
    - Initialize StandardScaler.
    - THINK: On which set do you fit the scaler, and on which set do you only transform?
    - Return X_train_scaled, X_test_scaled, y_train, y_test, scaler.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Model Training & Multi-Metric Evaluation
    - Fit a Logistic Regression model on the scaled training set.
    - Generate predictions on the unseen test set.
    - Evaluate test accuracy, confusion matrix, and full classification_report.
    - Return the trained model.
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_5_model_persistence_and_inference(model, scaler, feature_columns):
    """
    [Phase 5] Pipeline Persistence & Production Inference
    - Save the trained model and scaler to disk using joblib.
    - Format a single sample applicant feature vector matching feature_columns order.
    - Scale the sample using the saved scaler and predict loan approval outcome.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


# ==============================================================================
# THE FACADE ORCHESTRATOR
# NOTE: In test environments, uncomment these calls chronologically as you solve.
# ==============================================================================
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
