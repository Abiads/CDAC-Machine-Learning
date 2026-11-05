"""
================================================================================
EXAM PRACTICE (WITH TRAIN/TEST SPLIT & OVERFITTING METRICS): Day 11 - Practice 1
NEW PROBLEM STATEMENT (PS - Operational Variation):
--------------------------------------------------------------------------------
The café regional manager reviewed our marketing spend analysis, but now poses
a new operational question:
  "Marketing is external, but what about in-store customer volume? I want to 
   know if store 'Foot_Traffic' directly predicts 'Daily_Revenue_USD'. 
   Also, audit the dispersion of daily revenue to ensure there are no extreme 
   revenue anomalies."

OVERFITTING & GENERALIZATION REQUIREMENT:
  "Do not just evaluate on the same data you trained on! Perform an 80/20 
   train/test split. Compare Train R2 vs Test R2 and Train MAE vs Test MAE 
   to diagnose whether the model is Overfitting, Underfitting, or Well-Generalized."
================================================================================
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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
    
    df = pd.read_csv(csv_path)
    print("First 5 rows:\n", df.head())
    print(f"Shape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics on the Target Metric
    - Audit dispersion and IQR outliers for 'Daily_Revenue_USD'.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    target_col_name = "Daily_Revenue_USD"
    col = df[target_col_name]
    
    print(col.describe())
    print(f"Median: {col.median():.2f}")
    print(f"Variance: {col.var():.2f}")
    
    # Tukey's 1.5 * IQR Outlier Rule
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = col[(col < lower_bound) | (col > upper_bound)]
    print(f"IQR: {iqr:.2f}, Outliers Count: {len(outliers)}")
    
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Correlation & Visualizations
    - Compute correlation and extract association between Foot_Traffic and revenue.
    - Generate correlation heatmap and scatter plot with regression fit.
    """
    print("\n[PHASE 3] Bivariate Association & Visualization")
    print("-" * 65)
    
    corr = df.corr(numeric_only=True)
    print("Correlation Matrix:\n", corr.round(4))
    
    predictor = "Foot_Traffic"
    target = "Daily_Revenue_USD"
    print(f"\nCorrelation ({predictor} vs {target}): r = {corr.loc[predictor, target]:.4f}")
    
    base_dir = Path(__file__).resolve().parent
    
    # 1. Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(base_dir / "practice_correlation_heatmap.png", dpi=150)
    plt.close()
    
    # 2. Scatter with fit line
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x=predictor, y=target, color='teal', line_kws={'color': 'darkred'})
    plt.title(f"Bivariate Scatter: {predictor} vs {target}")
    plt.xlabel(f"{predictor} (Store Volume)")
    plt.ylabel(f"{target} ($)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "practice_foot_traffic_scatter.png", dpi=150)
    plt.close()
    
    print("Saved visualizations: 'practice_correlation_heatmap.png', 'practice_foot_traffic_scatter.png'")
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics, Train/Test Split & Regression Modeling
    - Isolate 2D feature matrix X and 1D target vector y.
    - Split into 80% Train and 20% Test sets to evaluate generalization.
    - Train Linear Regression on the Training set only.
    """
    print("\n[PHASE 4] Predictive Modeling with Train/Test Split")
    print("-" * 65)
    
    X = df[['Foot_Traffic']]              # 2D DataFrame (N, 1)
    y = df['Daily_Revenue_USD']           # 1D Series (N,)
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training Samples: {len(X_train)} rows, Testing Samples: {len(X_test)} rows")
    
    # Fit model on training set ONLY
    model = LinearRegression().fit(X_train, y_train)
    
    print(f"Slope (beta_1): {model.coef_[0]:.4f}")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    
    return model, X_train, X_test, y_train, y_test


def phase_5_evaluation_and_reporting(model, X_train, X_test, y_train, y_test):
    """
    [Phase 5] Evaluation & Overfitting vs Underfitting Diagnostics
    - Compute Train vs Test R-squared.
    - Compute Train vs Test MAE.
    - Calculate Generalization Gap to formally diagnose model fit.
    """
    print("\n[PHASE 5] Evaluation & Overfitting / Underfitting Diagnosis")
    print("-" * 65)
    
    # 1. Performance on Training Data (Seen)
    y_train_pred = model.predict(X_train)
    r2_train = model.score(X_train, y_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    # 2. Performance on Testing Data (Unseen)
    y_test_pred = model.predict(X_test)
    r2_test = model.score(X_test, y_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # 3. Key Overfitting / Underfitting Metrics:
    # Generalization Gap (Drop in R-squared)
    generalization_gap = r2_train - r2_test
    
    print(f"Training R2 : {r2_train:.4f}  |  Training MAE : ${mae_train:.2f}")
    print(f"Testing  R2 : {r2_test:.4f}  |  Testing  MAE : ${mae_test:.2f}")
    print(f"Generalization Gap (R2_train - R2_test): {generalization_gap:.4f}")
    print("-" * 65)
    
    # 4. Automated Diagnostic Rules
    print("DIAGNOSTIC VERDICT:")
    if r2_train < 0.50 and r2_test < 0.50:
        print("❌ STATUS: UNDERFITTING (High Bias)")
        print("   Explanation: Both train and test scores are low. The model is too simple.")
    elif generalization_gap > 0.15 or mae_test > 2.0 * mae_train:
        print("❌ STATUS: OVERFITTING (High Variance)")
        print(f"   Explanation: Train R2 ({r2_train:.2f}) is much higher than Test R2 ({r2_test:.2f}).")
        print("   The model memorized training noise and fails to generalize to unseen test data.")
    else:
        print("✅ STATUS: GOOD FIT (Well-Generalized)")
        print(f"   Explanation: Test R2 ({r2_test:.4f}) is close to Train R2 ({r2_train:.4f}).")
        print(f"   The generalization gap ({generalization_gap:.4f}) is under 0.05, confirming robust generalization.")
    
    print("\nBusiness Action: Foot traffic reliably predicts revenue without overfitting.")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING COFFEE SHOP ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X_train, X_test, y_train, y_test = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_analytics_lifecycle()
