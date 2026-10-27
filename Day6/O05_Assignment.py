# Assignment 2: Multi-Feature Tech Salary Prediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


# 1. Synthetic Dataset Generation
np.random.seed(102)
n_tech_samples = 100

years_experience = np.random.uniform(1, 12, size=n_tech_samples)
projects_completed = np.random.poisson(lam=5, size=n_tech_samples)
code_review_score = np.random.uniform(3.5, 5.0, size=n_tech_samples)

# Target: Salary = Base (40k) + 10k*(exp) + 3k*(projects) + 15k*(review_score) + Noise
annual_salary = (
    40000 
    + 10000 * years_experience 
    + 3000 * projects_completed 
    + 15000 * code_review_score 
    + np.random.normal(0, 4000, size=n_tech_samples)
)

df_tech = pd.DataFrame({
    "experience_years": years_experience.round(1),
    "projects_completed": projects_completed,
    "code_review_score": code_review_score.round(2),
    "annual_salary": annual_salary.round(2)
})

# ============================================================
# Step 1: Dataset Preview
# ============================================================
print("Step 1: Dataset Preview (First 5 Rows)")
print(df_tech.head())
print("=" * 60)

# ============================================================
# Step 2: Matrix OLS Closed-Form Solution (Normal Equation)
# beta = (X^T * X)^(-1) * X^T * y
# ============================================================
X_raw = df_tech[["experience_years", "projects_completed", "code_review_score"]].values
y_vec = df_tech["annual_salary"].values

# Add Bias Column (intercept column of 1s)
X_design = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

# Apply Normal Equation: (X^T * X)^(-1) * X^T * y
beta_matrix = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_vec
feature_names = ["Intercept (Beta 0)", "experience_years", "projects_completed", "code_review_score"]

print("Step 2: Matrix OLS Closed-Form Parameters (Normal Equation)")
for name, coef in zip(feature_names, beta_matrix):
    print(f"{name:<22}: ₹{coef:10.2f}")
print("=" * 60)

# ============================================================
# Step 3: Fitting with Scikit-Learn LinearRegression
# ============================================================
X_df = df_tech[["experience_years", "projects_completed", "code_review_score"]]
y_df = df_tech["annual_salary"]

model = LinearRegression()
model.fit(X_df, y_df)

print("Step 3: Scikit-Learn LinearRegression Parameters")
print(f"Intercept (Beta 0):     ₹{model.intercept_:.2f}")
coef_df = pd.DataFrame({
    "Feature": X_df.columns,
    "Coefficient (₹)": model.coef_
}).sort_values(by="Coefficient (₹)", ascending=False)
print(coef_df.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 4: Variance Inflation Factor (VIF) Analysis
# ============================================================
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = X_df.copy()
    X_vif["intercept"] = 1.0
    vifs = [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_df.shape[1])
    ]
except ImportError:
    vifs = []
    for col in X_df.columns:
        X_other = X_df.drop(columns=[col])
        r2_vif = LinearRegression().fit(X_other, X_df[col]).score(X_other, X_df[col])
        vifs.append(1.0 / (1.0 - r2_vif) if r2_vif < 1.0 else np.inf)

vif_data = pd.DataFrame({
    "Feature": X_df.columns,
    "VIF": np.round(vifs, 2)
})
print("Step 4: Variance Inflation Factor (VIF) Analysis")
print(vif_data.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 5: Model Evaluation Metrics
# ============================================================
y_pred_all = model.predict(X_df)
ss_total = np.sum((y_df - np.mean(y_df)) ** 2)
ss_res = np.sum((y_df - y_pred_all) ** 2)
r2_val = 1 - (ss_res / ss_total)
rmse_val = np.sqrt(np.mean((y_df - y_pred_all) ** 2))

print("Step 5: Model Evaluation Metrics")
print(f"R² Score: {r2_val:.4f}")
print(f"RMSE:     ₹{rmse_val:,.2f}")
print("=" * 60)

# ============================================================
# Step 6: Predictions on Sample Profiles
# ============================================================
sample_profiles = pd.DataFrame({
    "experience_years": [2.5, 5.0, 8.0],
    "projects_completed": [3, 6, 10],
    "code_review_score": [3.8, 4.2, 4.8]
})
sample_profiles["predicted_salary"] = model.predict(sample_profiles).round(2)

print("Step 6: Predictions on Sample Profiles")
print(sample_profiles.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 7: Visualizing Actual vs Predicted Salary
# ============================================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_df, y=y_pred_all, color="teal", alpha=0.8, label="Model Predictions")
plt.plot([y_df.min(), y_df.max()], [y_df.min(), y_df.max()], color="red", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
plt.xlabel("Actual Annual Salary (₹)")
plt.ylabel("Predicted Annual Salary (₹)")
plt.title("Actual vs Predicted Tech Salary (Multiple Linear Regression)")
plt.legend()
plt.tight_layout()
plt.show()