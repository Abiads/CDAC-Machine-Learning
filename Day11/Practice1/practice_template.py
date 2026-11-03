"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - Practice 1
NEW PROBLEM STATEMENT (PS - Operational Variation):
--------------------------------------------------------------------------------
The café regional manager reviewed our marketing spend analysis, but now poses
a new operational question:
  "Marketing is external, but what about in-store customer volume? I want to 
   know if store 'Foot_Traffic' directly predicts 'Daily_Revenue_USD'. 
   Also, audit the dispersion of daily revenue to ensure there are no extreme 
   revenue anomalies."

YOUR TASK:
Read the new Problem Statement (PS) above. Fill in all the `___` blanks below
to implement the end-to-end 5-phase analytics lifecycle for this new scenario.
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Placeholder for your answers
___ = None

# Step 0: Robust file path resolution
csv_path = Path(__file__).resolve().parent / "coffee_shop_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Setup
    - Load the CSV dataset into a DataFrame.
    - Inspect head, shape, and missing values.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path)
    print("First 5 rows:\n", df.head())
    print(f"Shape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    
    return df                           # What must you return to avoid NoneType crash?


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics on the Target Metric
    - Audit dispersion and IQR outliers for 'Daily_Revenue_USD'.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    target_col_name = ___               # 3. Fill the exact target column name from the PS
    col = df[target_col_name]
    
    print(col.describe())
    print(f"Median: {col.median():.2f}")
    print(f"Variance: {col.var():.2f}")
    
    # Tukey's 1.5 * IQR Outlier Rule
    q1 = col.quantile(___)              # 4. Fill 25th percentile value (e.g. 0.25)
    q3 = col.quantile(___)              # 5. Fill 75th percentile value (e.g. 0.75)
    iqr = ___ - ___                     # 6. Formula for IQR
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 7. Complete the boolean filter with bitwise operator and parentheses
    outliers = col[(col < ___) | (col > ___)]
    print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    
    return ___                          # 8. Return the DataFrame


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation
    - Compute correlation and extract association between Foot_Traffic and revenue.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 9. What parameter must be passed in Pandas 2.0+ to prevent non-numeric crashes?
    corr = df.corr(numeric_only=___)
    print("Correlation Matrix:\n", corr.round(4))
    
    predictor = ___                     # 10. Fill the predictor column requested in this new PS
    target = ___                        # 11. Fill the target column
    print(f"Correlation ({predictor} vs {target}): r = {corr.loc[predictor, target]:.4f}")
    
    return ___                          # 12. Return the DataFrame


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Set up feature matrix X and target vector y according to the PS.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 13. Which column is the predictor, and what bracket syntax ensures a 2D matrix?
    X = df[[___]]                       
    
    # 14. Which column is the dependent target?
    y = df[___]                         
    
    # 15. Instantiate and fit Linear Regression
    model = LinearRegression().fit(___, ___)
    
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    
    # 16. Return the three essential items needed for Phase 5
    return ___, ___, ___


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Action
    - Compute R-squared, MAE, and formulate a business takeaway.
    """
    print("\n[PHASE 5] Evaluation & Prescriptive Action")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 17. Generate predictions
    y_pred = model.predict(___)
    
    # 18. Compute inbuilt R-squared score
    r2 = model.score(___, ___)
    
    # 19. Compute Mean Absolute Error
    mae = mean_absolute_error(___, ___)
    
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print("Business Action: Foot traffic strongly predicts daily store revenue.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# NOTE: In test environments, uncomment these calls chronologically as you solve.
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING COFFEE SHOP ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # df = phase_1_discovery_and_setup()
    # df = phase_2_descriptive_statistics(df)
    # df = phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
