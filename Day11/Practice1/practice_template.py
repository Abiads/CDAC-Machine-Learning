"""
================================================================================
PRACTICE TEMPLATE: Day 11 - Practice 1 (Local Coffee Shop Revenue Lifecycle)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: coffee_shop_data.csv
COLUMNS: ['Day', 'Marketing_Spend_USD', 'Average_Temperature_C', 'Foot_Traffic', 'Daily_Revenue_USD']
INDEPENDENT FEATURE (X): 'Marketing_Spend_USD'
DEPENDENT TARGET (y):    'Daily_Revenue_USD'
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

# PATH TRAP AVOIDANCE: Always resolve path relative to __file__, never bare string!
csv_path = Path(__file__).resolve().parent / "coffee_shop_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Read CSV using pandas: df = pd.read_csv(csv_path)
    2. Inspect structure: print(df.head()), print(df.shape), print(df.isnull().sum())
    3. CRITICAL: Always return df at the end!
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print("First 5 rows:\n", df.head())
    # print(f"Shape: {df.shape}")
    # print("Missing Values:\n", df.isnull().sum())
    # return df
    pass


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    Target Column to analyze: 'Daily_Revenue_USD'
    1. Compute describe(), median(), var():
       col = df['Daily_Revenue_USD']
       print(col.describe())
       print("Median:", col.median())
       print("Variance:", col.var())
    2. Compute IQR & Outlier check:
       q1 = col.quantile(0.25)
       q3 = col.quantile(0.75)
       iqr = q3 - q1
       # TRAP: Use bitwise '|' and wrap conditions in parentheses ()
       outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
       print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    3. CRITICAL: Always return df!
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # col = df['Daily_Revenue_USD']
    # print(col.describe())
    # print(f"Median: {col.median():.2f}")
    # print(f"Variance: {col.var():.2f}")
    # q1, q3 = col.quantile(0.25), col.quantile(0.75)
    # iqr = q3 - q1
    # outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    # print(f"IQR: {iqr:.2f}, Outlier count: {len(outliers)}")
    # return df
    pass


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Compute Pearson correlation:
       # TRAP: In Pandas 2.0+, always pass numeric_only=True to prevent crash on 'Day' column!
       corr = df.corr(numeric_only=True)
       print(corr.round(4))
       print(f"Correlation: {corr.loc['Marketing_Spend_USD', 'Daily_Revenue_USD']:.4f}")
    2. CRITICAL: Always return df!
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # corr = df.corr(numeric_only=True)
    # print("Correlation Matrix:\n", corr.round(4))
    # print(f"Marketing vs Revenue: r = {corr.loc['Marketing_Spend_USD', 'Daily_Revenue_USD']:.4f}")
    # return df
    pass


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Independent Feature X: df[['Marketing_Spend_USD']]  <-- TRAP: DOUBLE BRACKETS for 2D!
    2. Dependent Target y:    df['Daily_Revenue_USD']       <-- Single bracket for 1D Series
    3. Fit model:
       model = LinearRegression().fit(X, y)
       print(f"Slope (beta_1): {model.coef_[0]:.4f}")
       print(f"Intercept (beta_0): {model.intercept_:.4f}")
    4. CRITICAL: Return (model, X, y)!
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = df[['Marketing_Spend_USD']]  # 2D DataFrame (N, 1)
    # y = df['Daily_Revenue_USD']       # 1D Series (N,)
    # model = LinearRegression().fit(X, y)
    # print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    # print(f"Intercept (beta_0): {model.intercept_:.4f}")
    # return model, X, y
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Action (5 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Evaluate model:
       y_pred = model.predict(X)
       r2 = model.score(X, y) # Inbuilt R-squared
       mae = mean_absolute_error(y, y_pred)
       print(f"R-squared: {r2:.4f}, MAE: ${mae:.2f}")
    2. Prescriptive Business Action:
       print("Prescriptive Action: Each $1 marketing generates $4.88 revenue. Expand budget.")
    """
    print("\n[PHASE 5] Evaluation & Prescriptive Action")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # y_pred = model.predict(X)
    # r2 = model.score(X, y)
    # mae = mean_absolute_error(y, y_pred)
    # print(f"R-squared: {r2:.4f}")
    # print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    # print("Business Action: High ROI ($4.88 per $1 spend) justifies scaling ad spend.")
    pass


# ==============================================================================
# THE FACADE ORCHESTRATOR
# TRAP: In tests, these lines are commented out! Uncomment them one by one as you solve!
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING COFFEE SHOP ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    # df = phase_2_descriptive_statistics(df)
    # df = phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
