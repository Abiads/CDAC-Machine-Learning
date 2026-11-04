"""
================================================================================
EXAM PRACTICE (COMPLETE SOLUTION): Day 11 - Practice 3
NEW PROBLEM STATEMENT (PS - Feature Variation):
--------------------------------------------------------------------------------
The HR director states:
  "Training hours were informative, but we want to assess senior staff contribution.
   Evaluate whether 'Years_Experience' reliably predicts 'Weekly_Tasks_Completed'. 
   Determine the expected productivity gain per year of experience, evaluate 
   the model using R-squared and MAE, and recommend whether hiring experienced staff 
   provides higher task output."
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "employee_productivity.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Setup
    - Ingest employee_productivity.csv.
    - Inspect head, shape, and null values.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    df = pd.read_csv(csv_path)
    print("Head:\n", df.head())
    print(f"Shape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Dispersion
    - Analyze the primary target column (Weekly_Tasks_Completed).
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    target_col = "Weekly_Tasks_Completed"
    col = df[target_col]
    
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    
    # Tukey's 1.5 * IQR Rule
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    
    outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation Matrix
    - Compute correlation and extract association between Years_Experience and Weekly_Tasks_Completed.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    corr = df.corr(numeric_only=True)
    print("Correlation Matrix:\n", corr.round(4))
    
    predictor = "Years_Experience"
    target = "Weekly_Tasks_Completed"
    print(f"\nCorrelation ({predictor} vs {target}): r = {corr.loc[predictor, target]:.4f}")
    
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Set up feature matrix X and target vector y according to the PS.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    X = df[['Years_Experience']]
    y = df['Weekly_Tasks_Completed']
    
    model = LinearRegression().fit(X, y)
    
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & HR Recommendations
    - Compute R-squared, MAE, and strategic HR conclusion.
    """
    print("\n[PHASE 5] Evaluation & HR Recommendations (5 Marks)")
    print("-" * 65)
    
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"R-squared: {r2:.4f}, MAE: {mae:.2f}")
    print("HR Action: Experience positively predicts task velocity; factor into hiring compensation.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
