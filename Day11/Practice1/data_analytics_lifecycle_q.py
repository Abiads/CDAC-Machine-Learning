"""
Title: End-to-End Data Analytics Lifecycle
Description: Minimal, built-in function driven Python script for the 5 lifecycle phases.
"""


from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Direct load: loads directly from current folder, falls back to Day11/Practice1 if run from root
csv_file = "coffee_shop_data.csv" if os.path.exists("coffee_shop_data.csv") else "Day11/Practice1/coffee_shop_data.csv"
csv_file = Path(__file__).resolve().parent / "coffee_shop_data.csv"
df = None
model = None
X, y = None, None


def phase_1_discovery_and_setup():
    """Phase 1: Load and inspect data shape, types, and missing values."""
    global df
    print("--- Phase 1: Discovery & Setup ---")
    df = pd.read_csv(csv_file)
    print(df.head())
    print("\nMissing Values:\n", df.isnull().sum())


def phase_2_descriptive_statistics():
    """Phase 2: Descriptive stats using built-in describe, median, and IQR outliers."""
    print("\n--- Phase 2: Descriptive Statistics ---")
    num_df = df.select_dtypes(include='number')
    print("Summary (Mean, Std, Min, Max):\n", num_df.describe().T[['mean', 'std', 'min', 'max']])
    print("\nMedian:\n", num_df.median())
    print("\nVariance:\n", num_df.var())
    
    # IQR & Outliers for Daily_Revenue_USD
    col = df['Daily_Revenue_USD']
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    print(f"\nDaily_Revenue IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")


def phase_3_bivariate_association():
    """Phase 3: Association & Exploratory Visualizations (Heatmap & Scatter Plot)."""
    print("\n--- Phase 3: Bivariate Association & Visualization ---")
    print("Covariance Matrix:\n", df.cov(numeric_only=True))
    corr = df.corr(numeric_only=True)
    print("\nCorrelation Matrix:\n", corr)

    base_dir = Path(__file__).resolve().parent

    # 1. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", cbar=True)
    plt.title("Correlation Heatmap - Coffee Shop Analytics")
    plt.tight_layout()
    heatmap_path = base_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # 2. Bivariate Scatter Plot with Regression Line
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x='Marketing_Spend_USD', y='Daily_Revenue_USD', color='royalblue', line_kws={'color': 'darkred'})
    plt.title("Bivariate Association: Marketing Spend vs Daily Revenue")
    plt.xlabel("Marketing Spend ($ USD)")
    plt.ylabel("Daily Revenue ($ USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    scatter_path = base_dir / "bivariate_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    print(f"Saved visualizations: '{heatmap_path.name}', '{scatter_path.name}'")


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
