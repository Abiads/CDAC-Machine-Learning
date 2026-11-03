"""
================================================================================
PRACTICE TEMPLATE: Day 11 - Practice 4B (Healthcare Hypertension Classification)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_classification_template.py

DATASET: patient_classification_data.csv
COLUMNS: ['Patient_ID', 'Age', 'BMI', 'Daily_Steps', 'High_Risk_Hypertension']
INDEPENDENT FEATURES (X): ['BMI', 'Daily_Steps']
DEPENDENT TARGET (y):     'High_Risk_Hypertension' (0 = Low Risk, 1 = High Risk)
MODEL: LogisticRegression
METRICS: accuracy_score, confusion_matrix
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix

csv_path = Path(__file__).resolve().parent / "patient_classification_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (Class Balance)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. Check target balance:
       print("Class Balance:\n", df['High_Risk_Hypertension'].value_counts())
    3. Return df!
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print("Class Balance (High_Risk_Hypertension):")
    # print(df['High_Risk_Hypertension'].value_counts())
    # return df
    pass


def phase_2_supervised_segmentation(df):
    """
    [Phase 2] Supervised Segmentation & Group Means
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Group by target class and compute feature means:
       means = df.groupby('High_Risk_Hypertension')[['BMI', 'Daily_Steps']].mean()
       print("Feature Means by Risk Category:\n", means)
    2. Return df!
    """
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # print("Feature Means by Risk Category:")
    # print(df.groupby('High_Risk_Hypertension')[['BMI', 'Daily_Steps']].mean())
    # return df
    pass


def phase_3_classification_modeling(df):
    """
    [Phase 3] Classification Model Induction (Logistic Regression)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X = df[['BMI', 'Daily_Steps']]  <-- DOUBLE BRACKETS
    2. y = df['High_Risk_Hypertension']
    3. Fit model:
       model = LogisticRegression().fit(X, y)
    4. Return (model, X, y)!
    """
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = df[['BMI', 'Daily_Steps']]
    # y = df['High_Risk_Hypertension']
    # model = LogisticRegression().fit(X, y)
    # print("Logistic Regression model trained successfully.")
    # return model, X, y
    pass


def phase_4_model_evaluation(model, X, y):
    """
    [Phase 4] Model Evaluation & Clinical Action
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. y_pred = model.predict(X)
    2. acc = accuracy_score(y, y_pred)
    3. cm = confusion_matrix(y, y_pred)
    4. Print metrics and clinical recommendation.
    """
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # y_pred = model.predict(X)
    # print(f"Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")
    # print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    # print("\nClinical Action: Proactively flag patients with low daily steps and high BMI for hypertension screening.")
    pass


def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    # df = phase_2_supervised_segmentation(df)
    # model, X, y = phase_3_classification_modeling(df)
    # phase_4_model_evaluation(model, X, y)


if __name__ == "__main__":
    run_analytics_lifecycle()
