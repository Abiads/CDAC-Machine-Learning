"""
================================================================================
PRACTICE TEMPLATE: Day 11 - Practice 3 (Employee HR Productivity Lifecycle)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: employee_productivity.csv
COLUMNS: ['Employee_ID', 'Training_Hours', 'Years_Experience', 'Daily_Hours_Worked', 'Team_Size', 'Weekly_Tasks_Completed']
INDEPENDENT FEATURE (X): 'Training_Hours'
DEPENDENT TARGET (y):    'Weekly_Tasks_Completed'
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

# PATH TRAP AVOIDANCE
csv_path = Path(__file__).resolve().parent / "employee_productivity.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Read CSV: df = pd.read_csv(csv_path)
    2. Inspect structure: print(df.head()), print(df.shape), print(df.isnull().sum())
    3. CRITICAL: Always return df!
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # print(df.head())
    # print(f"Shape: {df.shape}")
    # print("Missing Values:\n", df.isnull().sum())
    # return df
    pass


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    Target Column to analyze: 'Weekly_Tasks_Completed'
    1. Compute describe(), IQR, and outliers:
       col = df['Weekly_Tasks_Completed']
       q1 = col.quantile(0.25)
       q3 = col.quantile(0.75)
       iqr = q3 - q1
       # TRAP: Wrap in parentheses with bitwise '|'
       outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    2. CRITICAL: Always return df!
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # col = df['Weekly_Tasks_Completed']
    # print(col.describe())
    # q1, q3 = col.quantile(0.25), col.quantile(0.75)
    # iqr = q3 - q1
    # print(f"IQR: {iqr:.2f}")
    # outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    # print(f"Outliers Count: {len(outliers)}")
    # return df
    pass


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation Matrix (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Compute correlation:
       # TRAP: Pass numeric_only=True because 'Employee_ID' is string!
       corr = df.corr(numeric_only=True)
       print(corr.round(4))
       print(f"Training_Hours vs Weekly_Tasks_Completed: r = {corr.loc['Training_Hours', 'Weekly_Tasks_Completed']:.4f}")
    2. CRITICAL: Always return df!
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # corr = df.corr(numeric_only=True)
    # print(corr.round(4))
    # print(f"\nTraining_Hours vs Weekly_Tasks_Completed: r = {corr.loc['Training_Hours', 'Weekly_Tasks_Completed']:.4f}")
    # return df
    pass


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. X = df[['Training_Hours']]  <-- TRAP: DOUBLE BRACKETS for 2D!
    2. y = df['Weekly_Tasks_Completed']
    3. Fit Linear Regression:
       model = LinearRegression().fit(X, y)
       print(f"Slope (beta_1): {model.coef_[0]:.4f}")
       print(f"Intercept (beta_0): {model.intercept_:.4f}")
    4. CRITICAL: Return (model, X, y)!
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # X = df[['Training_Hours']]
    # y = df['Weekly_Tasks_Completed']
    # model = LinearRegression().fit(X, y)
    # print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    # print(f"Intercept (beta_0): {model.intercept_:.4f}")
    # return model, X, y
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & HR Recommendations (5 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. y_pred = model.predict(X)
    2. r2 = model.score(X, y)
    3. mae = mean_absolute_error(y, y_pred)
    4. Print business action based on slope & R2.
    """
    print("\n[PHASE 5] Evaluation & HR Recommendations (5 Marks)")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # y_pred = model.predict(X)
    # r2 = model.score(X, y)
    # mae = mean_absolute_error(y, y_pred)
    # print(f"R-squared: {r2:.4f}, MAE: {mae:.2f}")
    # print("HR Action: Positive correlation justifies training, but experience also matters.")
    pass


# ==============================================================================
# THE FACADE ORCHESTRATOR
# TRAP: Uncomment lines chronologically as you complete each phase!
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    # df = phase_2_descriptive_statistics(df)
    # df = phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
