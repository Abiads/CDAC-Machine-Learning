"""
================================================================================
EXAM PRACTICE: Day 11 - Practice 1
DOMAIN: Retail & Hospitality Analytics (Coffee Shop Operations)
================================================================================
PROBLEM STATEMENT (PS):
A specialty coffee chain is evaluating whether increasing daily digital marketing
spend translates to higher store revenue. You are provided with daily operational 
data from the flagship store across 25 consecutive operating days.

MANAGEMENT QUESTIONS TO SOLVE:
1. Is the revenue distribution symmetrical, or are there significant outliers?
2. Which feature has the strongest linear association with daily revenue?
3. How much revenue increase is expected for every additional $1 spent on marketing?
4. How well does marketing spend alone explain revenue variance, and what is the 
   expected prediction error in dollars?

DATASET FILE: coffee_shop_data.csv
AVAILABLE COLUMNS:
  - Day (string ID)
  - Marketing_Spend_USD (continuous independent predictor)
  - Average_Temperature_C (ambient temperature)
  - Foot_Traffic (store visitor count)
  - Daily_Revenue_USD (continuous business response / target)

THINKING CHECKLIST FOR THE EXAM:
- Look at the column names in the dataset before selecting features.
- If the exam switches the feature to Foot_Traffic or Average_Temperature_C,
  which line in Phase 4 determines that choice?
- Why must you avoid bare filename paths when loading the CSV?
- What happens if you run .corr() on non-numeric columns in Pandas 2.0+?
- What array dimension does Scikit-Learn require for feature matrices?
================================================================================
"""

from pathlib import Path
# TODO: Import required libraries (Pandas, LinearRegression, metrics)

csv_path = Path(__file__).resolve().parent / "coffee_shop_data.csv"


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup
    - Load the dataset dynamically from csv_path.
    - Inspect the first 5 records, check total row/column dimensions.
    - Audit for missing or null entries across all fields.
    - Return the ingested DataFrame.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Outlier Audit
    - Analyze the primary target metric (Daily_Revenue_USD).
    - Compute parametric measures (count, mean, standard deviation, min, max).
    - Compute non-parametric dispersion (median, variance).
    - Implement Tukey's 1.5 * IQR rule to identify any revenue anomalies.
    - Return the DataFrame.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation
    - Compute the Pearson correlation matrix for all numeric columns.
    - Extract and report the correlation coefficient between marketing spend and revenue.
    - THINK: Check the correlation between Foot_Traffic and revenue as well.
    - Return the DataFrame.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Isolate the independent predictor matrix (X) and the dependent target vector (y).
    - THINK: What bracket syntax ensures X is a 2D matrix?
    - Fit an ordinary least squares Simple Linear Regression model.
    - Extract and print the regression slope (beta_1) and intercept (beta_0).
    - Return the trained model, feature matrix X, and target vector y.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Model Evaluation & Prescriptive Action
    - Generate model predictions on X.
    - Compute the coefficient of determination (R-squared) and Mean Absolute Error (MAE).
    - Translate the mathematical slope and error into an actionable business recommendation.
    """
    print("\n[PHASE 5] Evaluation & Prescriptive Action")
    print("-" * 65)
    # TODO: Write your implementation here
    pass


# ==============================================================================
# THE FACADE ORCHESTRATOR
# NOTE: In test environments, uncomment these calls chronologically as you solve.
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING COFFEE SHOP ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # Phase 1
    df = phase_1_discovery_and_setup()
    
    # Phase 2
    # df = phase_2_descriptive_statistics(df)
    
    # Phase 3
    # df = phase_3_bivariate_association(df)
    
    # Phase 4
    # model, X, y = phase_4_predictive_modeling(df)
    
    # Phase 5
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
