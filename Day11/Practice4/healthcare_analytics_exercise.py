"""
Title: Healthcare Patient Analytics Lifecycle 
Description: Simple, built-in function driven Python script for healthcare vitals analytics.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "patient_health_data.csv"


def phase_1_discovery_and_setup():
    """Phase 1: Load data and inspect structure."""
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    return df


def phase_2_descriptive_statistics(df):
    """Phase 2: Descriptive stats and IQR outlier check for Systolic_BP."""
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    col = df['Systolic_BP']
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    print(f"IQR: {iqr:.2f}")
    outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    print(f"Outliers Count: {len(outliers)}")
    return df


def phase_3_bivariate_association(df):
    """Phase 3: Bivariate Pearson correlation matrix & Visualizations."""
    print("\n[PHASE 3] Bivariate Association & Visualization")
    print("-" * 65)
    corr = df.corr(numeric_only=True)
    print(corr.round(4))
    print(f"\nBMI vs Systolic_BP: r = {corr.loc['BMI', 'Systolic_BP']:.4f}")
    print(f"Daily_Steps vs Systolic_BP: r = {corr.loc['Daily_Steps', 'Systolic_BP']:.4f}")

    base_dir = Path(__file__).resolve().parent

    # 1. Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", cbar=True)
    plt.title("Correlation Matrix - Healthcare Vitals")
    plt.tight_layout()
    plt.savefig(base_dir / "healthcare_correlation_heatmap.png", dpi=150)
    plt.close()

    # 2. Scatter plot with regression line
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x='BMI', y='Systolic_BP', color='crimson', line_kws={'color': 'navy'})
    plt.title("BMI vs Systolic Blood Pressure")
    plt.xlabel("Body Mass Index (BMI)")
    plt.ylabel("Systolic Blood Pressure (mmHg)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "bmi_vs_systolic_bp.png", dpi=150)
    plt.close()

    print("Saved visualizations: 'healthcare_correlation_heatmap.png', 'bmi_vs_systolic_bp.png'")
    return df


def phase_4_predictive_modeling(df):
    """Phase 4: Simple Linear Regression predicting Systolic_BP from BMI."""
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    X = df[['BMI']]
    y = df['Systolic_BP']
    model = LinearRegression().fit(X, y)
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """Phase 5: Evaluation and clinical recommendations."""
    print("\n[PHASE 5] Evaluation & Clinical Action (5 Marks)")
    print("-" * 65)
    y_pred = model.predict(X)
    print(f"R-squared Score: {model.score(X, y):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y, y_pred):.2f}")
    print("\nClinical Action: Higher BMI strongly predicts elevated blood pressure; prescribe lifestyle & weight management.")


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("HEALTHCARE LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
