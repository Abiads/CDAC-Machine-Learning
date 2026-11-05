"""
Title: E-Commerce Customer Analytics Lifecycle 
Description: Simple, built-in function driven Python script for ecommerce spend analytics.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "ecommerce_customer_data.csv"


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
    """Phase 2: Descriptive stats and IQR outlier check for Annual_Spend_USD."""
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    col = df['Annual_Spend_USD']
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = col[(col < lower) | (col > upper)]
    print(f"IQR: {iqr:.2f}")
    print(f"Outliers Count: {len(outliers)}")
    return df


def phase_3_bivariate_association(df):
    """Phase 3: Bivariate Pearson correlation matrix & Visualizations."""
    print("\n[PHASE 3] Bivariate Association & Visualization")
    print("-" * 65)
    corr = df.corr(numeric_only=True)
    print(corr.round(4))
    print(f"\nBrowsing_Minutes vs Annual_Spend_USD: r = {corr.loc['Browsing_Minutes', 'Annual_Spend_USD']:.4f}")
    print(f"Items_In_Cart vs Annual_Spend_USD: r = {corr.loc['Items_In_Cart', 'Annual_Spend_USD']:.4f}")

    base_dir = Path(__file__).resolve().parent

    # 1. Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", cbar=True)
    plt.title("Correlation Matrix - E-Commerce Analytics")
    plt.tight_layout()
    plt.savefig(base_dir / "ecommerce_correlation_heatmap.png", dpi=150)
    plt.close()

    # 2. Scatter plot with regression line
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x='Browsing_Minutes', y='Annual_Spend_USD', color='darkorange', line_kws={'color': 'darkblue'})
    plt.title("Browsing Minutes vs Annual Spend ($ USD)")
    plt.xlabel("Browsing Minutes")
    plt.ylabel("Annual Spend ($ USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "browsing_minutes_vs_spend.png", dpi=150)
    plt.close()

    print("Saved visualizations: 'ecommerce_correlation_heatmap.png', 'browsing_minutes_vs_spend.png'")
    return df


def phase_4_predictive_modeling(df):
    """Phase 4: Simple Linear Regression predicting Annual_Spend from Browsing_Minutes."""
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    X = df[['Browsing_Minutes']]
    y = df['Annual_Spend_USD']
    model = LinearRegression().fit(X, y)
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """Phase 5: Evaluation and marketing recommendations."""
    print("\n[PHASE 5] Evaluation & Business Action")
    print("-" * 65)
    y_pred = model.predict(X)
    print(f"R-squared Score: {model.score(X, y):.4f}")
    print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y, y_pred):.2f}")
    print("\nMarketing Action: Browsing duration strongly drives customer checkout value; optimize on-site product engagement.")


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("E-COMMERCE LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
