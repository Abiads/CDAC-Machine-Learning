"""
================================================================================
EXAM PRACTICE (COMPLETE SOLUTION): Day 11 - Practice 4B
NEW PROBLEM STATEMENT (PS - Diagnostic Multi-Feature Variation):
--------------------------------------------------------------------------------
The clinical board requests an updated diagnostic classifier:
  "Investigate whether combining patient 'BMI' and 'Daily_Steps' yields accurate
   classification of 'High_Risk_Hypertension'. Audit class balance, compare 
   group means, train a LogisticRegression model, and evaluate accuracy and confusion matrix."
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

csv_path = Path(__file__).resolve().parent / "patient_classification_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Class Balance Audit
    - Ingest patient_classification_data.csv.
    - Check the distribution of 'High_Risk_Hypertension'.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    df = pd.read_csv(csv_path)
    target_col = "High_Risk_Hypertension"
    print(f"Class Balance ({target_col}):\n", df[target_col].value_counts())
    
    return df


def phase_2_supervised_segmentation(df):
    """
    [Phase 2] Supervised Risk Segmentation
    - Group by target class and compute mean of BMI and Daily_Steps.
    """
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    
    target_col = "High_Risk_Hypertension"
    feature_cols = ['BMI', 'Daily_Steps']
    means = df.groupby(target_col)[feature_cols].mean()
    print("Feature Means by Risk Category:\n", means)
    
    return df


def phase_3_classification_modeling(df):
    """
    [Phase 3] Classification Model Induction
    - Fit a LogisticRegression model.
    """
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    
    X = df[['BMI', 'Daily_Steps']]
    y = df['High_Risk_Hypertension']
    
    model = LogisticRegression().fit(X, y)
    print("Logistic Regression model trained successfully.")
    
    return model, X, y


def phase_4_model_evaluation(model, X, y):
    """
    [Phase 4] Clinical Classifier Evaluation
    - Predict on X, compute accuracy and confusion matrix.
    """
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Clinical Action: Flag sedentary patients with elevated BMI for proactive screening.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    df = phase_2_supervised_segmentation(df)
    model, X, y = phase_3_classification_modeling(df)
    phase_4_model_evaluation(model, X, y)


if __name__ == "__main__":
    run_analytics_lifecycle()
