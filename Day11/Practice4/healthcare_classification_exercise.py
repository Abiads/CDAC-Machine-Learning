"""
Title: Healthcare Classification Lifecycle
Description: Simple, built-in function driven Python script for healthcare risk classification.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

csv_path = Path(__file__).resolve().parent / "patient_classification_data.csv"


def phase_1_discovery_and_setup():
    """Phase 1: Load data and check class distribution."""
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    print("Class Balance (High_Risk_Hypertension):")
    print(df['High_Risk_Hypertension'].value_counts())
    return df


def phase_2_supervised_segmentation(df):
    """Phase 2: Segment feature differences between risk groups."""
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    means = df.groupby('High_Risk_Hypertension')[['BMI', 'Daily_Steps']].mean()
    print("Feature Means by Risk Category:\n", means)
    return df


def phase_3_classification_modeling(df):
    """Phase 3: Train Logistic Regression model."""
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    X = df[['BMI', 'Daily_Steps']]
    y = df['High_Risk_Hypertension']
    model = LogisticRegression().fit(X, y)
    print("Logistic Regression model trained successfully.")
    return model, X, y


def phase_4_model_evaluation(model, X, y):
    """Phase 4: Evaluate accuracy, confusion matrix, and save confusion matrix heatmap."""
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)

    base_dir = Path(__file__).resolve().parent
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'High Risk (1)'],
                yticklabels=['Normal (0)', 'High Risk (1)'])
    plt.title("Confusion Matrix - Hypertension Risk Classification")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(base_dir / "hypertension_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved visualization: 'hypertension_confusion_matrix.png'")

    print("\nClinical Action: Proactively flag patients with low daily steps and high BMI for hypertension screening.")


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_supervised_segmentation(df)
    model, X, y = phase_3_classification_modeling(df)
    phase_4_model_evaluation(model, X, y)

    print("\n" + "=" * 65)
    print("CLASSIFICATION LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
