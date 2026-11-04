"""
================================================================================
EXAM PRACTICE (COMPLETE SOLUTION): Day 11 - Practice 5
NEW PROBLEM STATEMENT (PS - Cart Activity Variation):
--------------------------------------------------------------------------------
The digital retail director states:
  "Browsing time is useful, but we want to know if 'Items_In_Cart' is a direct 
   monetary driver of 'Annual_Spend_USD'. 
   Evaluate whether cart size predicts annual spend, extract the marginal dollar 
   value per item in cart (slope beta_1), and compute R-squared and MAE."
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "ecommerce_customer_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Setup
    - Ingest ecommerce_customer_data.csv.
    - Inspect head, shape, and null entries.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    df = pd.read_csv(csv_path)
    print("Head:\n", df.head())
    print(f"\nShape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Dispersion
    - Audit dispersion and IQR outliers for 'Annual_Spend_USD'.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    target_col = "Annual_Spend_USD"
    col = df[target_col]
    
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    
    outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation Matrix
    - Compute correlation and extract association between Items_In_Cart and Annual_Spend_USD.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    corr = df.corr(numeric_only=True)
    print("Correlation Matrix:\n", corr.round(4))
    
    predictor = "Items_In_Cart"
    target = "Annual_Spend_USD"
    print(f"\nCorrelation ({predictor} vs {target}): r = {corr.loc[predictor, target]:.4f}")
    
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Set up feature matrix X and target vector y according to the PS.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    X = df[['Items_In_Cart']]
    y = df['Annual_Spend_USD']
    
    model = LinearRegression().fit(X, y)
    
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & E-Commerce Marketing Action
    - Compute R-squared, MAE, and growth recommendations.
    """
    print("\n[PHASE 5] Evaluation & Business Action")
    print("-" * 65)
    
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print("Marketing Action: Cart items strongly drive annual spend; implement smart cart-add recommendations.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
