"""
Title: Employee Productivity & Analytics Lifecycle 
Description: Simple, built-in function driven Python script for employee productivity analytics.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "employee_productivity.csv"


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
    """Phase 2: Descriptive stats and IQR outlier check for Weekly_Tasks_Completed."""
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    col = df['Weekly_Tasks_Completed']
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    print(col.describe())
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
    print(f"\nTraining_Hours vs Weekly_Tasks_Completed: r = {corr.loc['Training_Hours', 'Weekly_Tasks_Completed']:.4f}")

    base_dir = Path(__file__).resolve().parent
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", cbar=True)
    plt.title("Correlation Matrix - Employee Analytics")
    plt.tight_layout()
    plt.savefig(base_dir / "employee_correlation_heatmap.png", dpi=150)
    plt.close()

    # 2. Scatter Plot with Regression Line
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x='Training_Hours', y='Weekly_Tasks_Completed', color='teal', line_kws={'color': 'darkred'})
    plt.title("Training Hours vs Weekly Tasks Completed")
    plt.xlabel("Training Hours Completed")
    plt.ylabel("Weekly Tasks Completed")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "employee_training_vs_tasks.png", dpi=150)
    plt.close()

    print("Saved visualizations: 'employee_correlation_heatmap.png', 'employee_training_vs_tasks.png'")
    return df


def phase_4_predictive_modeling(df):
    """Phase 4: Simple Linear Regression predicting tasks from training hours."""
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    X = df[['Training_Hours']]
    y = df['Weekly_Tasks_Completed']
    model = LinearRegression().fit(X, y)
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """Phase 5: Evaluation and prescriptive HR action."""
    print("\n[PHASE 5] Evaluation & HR Recommendations")
    print("-" * 65)
    y_pred = model.predict(X)
    print(f"R-squared Score: {model.score(X, y):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y, y_pred):.2f}")
    print("\nHR Action: Training hours strongly drive task velocity; scale up employee training.")


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("EMPLOYEE LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
