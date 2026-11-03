"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - Practice 4A
NEW PROBLEM STATEMENT (PS - Clinical Exercise Variation):
--------------------------------------------------------------------------------
The head cardiologist states:
  "We understand BMI increases blood pressure, but we need to demonstrate the 
   protective value of aerobic exercise. Investigate whether patient 'Daily_Steps' 
   can reliably predict 'Systolic_BP'. 
   NOTE: Because higher physical activity reduces hypertension, what sign 
   (positive or negative) do you expect for the correlation r and the slope beta_1?"

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

csv_path = Path(__file__).resolve().parent / "patient_health_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Clinical Setup
    - Ingest patient_health_data.csv.
    - Inspect head, shape, and null entries.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    print("Head:\n", df.head())
    print(f"Shape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    
    return ___                          # 1. Return the DataFrame


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Dispersion
    - Audit dispersion and IQR outliers for 'Systolic_BP'.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    target_col = ___                    # 2. Fill the dependent clinical target column
    col = df[target_col]
    
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    
    # Tukey's 1.5 * IQR Rule
    q1 = col.quantile(___)              # 3. 25th percentile
    q3 = col.quantile(___)              # 4. 75th percentile
    iqr = ___ - ___
    
    # 5. Outlier bounds filter
    outliers = col[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)]
    print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    
    return ___                          # 6. Return the DataFrame


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation Matrix
    - Compute correlation and extract association between Daily_Steps and Systolic_BP.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 7. Pass parameter to skip non-numeric Patient_ID
    corr = df.corr(numeric_only=___)
    print("Correlation Matrix:\n", corr.round(4))
    
    predictor = ___                     # 8. Fill the predictor feature from the new PS
    target = ___                        # 9. Fill the clinical target metric
    print(f"\nCorrelation ({predictor} vs {target}): r = {corr.loc[predictor, target]:.4f}")
    
    return ___                          # 10. Return the DataFrame


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Set up feature matrix X and target vector y according to the PS.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 11. Feature matrix (DOUBLE brackets for 2D DataFrame)
    X = df[[___]]                       
    
    # 12. Target vector (Single bracket for 1D Series)
    y = df[___]                         
    
    # 13. Fit model
    model = LinearRegression().fit(___, ___)
    
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    
    return ___, ___, ___                # 14. Return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Clinical Evaluation & Exercise Prescription
    - Compute R-squared, MAE, and clinical recommendations.
    """
    print("\n[PHASE 5] Evaluation & Clinical Action (5 Marks)")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 15. Predict on X
    y_pred = model.predict(___)
    
    # 16. Inbuilt R2 calculation
    r2 = model.score(___, ___)
    
    # 17. Mean Absolute Error
    mae = mean_absolute_error(___, ___)
    
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mmHg")
    print("Clinical Action: Increased physical activity directly lowers blood pressure.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # df = phase_1_discovery_and_setup()
    # df = phase_2_descriptive_statistics(df)
    # df = phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
