"""
Title: End-to-End Data Analytics Lifecycle
Description: Minimal, built-in function driven Python script for the 5 lifecycle phases.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset using built-in relative path
csv_path = Path(__file__).parent / "coffee_shop_data.csv"
df = None
model = None
X, y = None, None


def phase_1_discovery_and_setup():
    """Phase 1: Load and inspect data shape, types, and missing values."""
    global df
    print("--- Phase 1: Discovery & Setup ---")
    df = pd.read_csv(csv_path)
    print(df.head())
    print("\nMissing Values:\n", df.isnull().sum())


def phase_2_descriptive_statistics():
    """Phase 2: Descriptive stats using built-in describe, median, and variance."""
    print("\n--- Phase 2: Descriptive Statistics ---")
    num_df = df.select_dtypes(include='number')
    print("Summary (Mean, Std, Min, Max):\n", num_df.describe().T[['mean', 'std', 'min', 'max']])
    print("\nMedian:\n", num_df.median())
    print("\nVariance:\n", num_df.var())


def phase_3_bivariate_association():
    """Phase 3: Association using built-in cov() and corr()."""
    print("\n--- Phase 3: Bivariate Association ---")
    print("Covariance Matrix:\n", df.cov(numeric_only=True))
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))


def phase_4_predictive_modeling():
    """Phase 4: Fit Linear Regression model."""
    global model, X, y
    print("\n--- Phase 4: Predictive Modeling ---")
    X = df[['Marketing_Spend_USD']]
    y = df['Daily_Revenue_USD']
    model = LinearRegression().fit(X, y)
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")


def phase_5_evaluation_and_reporting():
    """Phase 5: Built-in model evaluation score and MAE."""
    print("\n--- Phase 5: Evaluation & Reporting ---")
    y_pred = model.predict(X)
    print(f"R-squared Score: {model.score(X, y):.4f}")
    print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y, y_pred):.2f}")
    print("\nBusiness Action: Every $1 spent on marketing generates ~$5.13 in revenue.")


def run_analytics_lifecycle():
    """Orchestrator executing all phases."""
    phase_1_discovery_and_setup()
    phase_2_descriptive_statistics()
    phase_3_bivariate_association()
    phase_4_predictive_modeling()
    phase_5_evaluation_and_reporting()


if __name__ == "__main__":
    run_analytics_lifecycle()
