"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - Practice 4B
NEW PROBLEM STATEMENT (PS - Diagnostic Multi-Feature Variation):
--------------------------------------------------------------------------------
The clinical board requests an updated diagnostic classifier:
  "Instead of only looking at activity, investigate whether combining patient 
   'BMI' and 'Daily_Steps' yields 100% accurate classification of 
   'High_Risk_Hypertension'. Audit class balance, compare group means, 
   train a LogisticRegression model, and evaluate accuracy and confusion matrix."

YOUR TASK:
Read the new Problem Statement (PS) above. Fill in all the `___` blanks below
to implement the end-to-end classification lifecycle.
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "patient_classification_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Class Balance Audit
    - Ingest patient_classification_data.csv.
    - Check the distribution of 'High_Risk_Hypertension'.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    target_col = ___                    # 1. Fill target column name
    print(f"Class Balance ({target_col}):\n", df[target_col].value_counts())
    
    return ___                          # 2. Return the DataFrame


def phase_2_supervised_segmentation(df):
    """
    [Phase 2] Supervised Risk Segmentation
    - Group by target class and compute mean of BMI and Daily_Steps.
    """
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 3. Group by target and select features to average
    target_col = ___
    feature_cols = [___, ___]           # 4. Fill the two feature names
    means = df.groupby(target_col)[feature_cols].___() # 5. Aggregate method (mean)
    print("Feature Means by Risk Category:\n", means)
    
    return ___                          # 6. Return the DataFrame


def phase_3_classification_modeling(df):
    """
    [Phase 3] Classification Model Induction
    - Fit a LogisticRegression model.
    """
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 7. Multi-column feature matrix (DOUBLE brackets)
    X = df[[___, ___]]                  
    
    # 8. Binary target vector (Single bracket)
    y = df[___]                         
    
    # 9. Instantiate and fit LogisticRegression
    model = LogisticRegression().fit(___, ___)
    print("Logistic Regression model trained successfully.")
    
    return ___, ___, ___                # 10. Return model, X, y


def phase_4_model_evaluation(model, X, y):
    """
    [Phase 4] Clinical Classifier Evaluation
    - Predict on X, compute accuracy and confusion matrix.
    """
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 11. Predict binary classes
    y_pred = model.predict(___)
    
    # 12. Compute accuracy percentage
    acc = accuracy_score(___, ___)
    
    # 13. Compute confusion matrix
    cm = confusion_matrix(___, ___)
    
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
    
    # df = phase_1_discovery_and_setup()
    # df = phase_2_supervised_segmentation(df)
    # model, X, y = phase_3_classification_modeling(df)
    # phase_4_model_evaluation(model, X, y)


if __name__ == "__main__":
    run_analytics_lifecycle()
