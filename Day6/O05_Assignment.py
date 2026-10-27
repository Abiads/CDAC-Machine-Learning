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
# Step 3: Model Building & Fitting (Scikit-Learn)
# ============================================================
X_tech = df_tech[["experience_years", "projects_completed", "code_review_score"]]
y_tech = df_tech["annual_salary"]

tech_model = LinearRegression()
tech_model.fit(X_tech, y_tech)

print("Step 3: Scikit-Learn Model Coefficients")
print(rf"Intercept (Base Salary): ₹{tech_model.intercept_:.2f}")
coef_summary = pd.DataFrame({
    "Feature": X_tech.columns,
    "Partial Coefficient (₹)": tech_model.coef_
}).sort_values(by="Partial Coefficient (₹)", ascending=False)
print(coef_summary.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 4: Variance Inflation Factor (VIF) Analysis
# ============================================================
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = X_tech.copy()
    X_vif["intercept"] = 1.0
    vifs = [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_tech.shape[1])
    ]
except ImportError:
    vifs = []
    for col in X_tech.columns:
        X_other = X_tech.drop(columns=[col])
        r2_vif = LinearRegression().fit(X_other, X_tech[col]).score(X_other, X_tech[col])
        vifs.append(1.0 / (1.0 - r2_vif) if r2_vif < 1.0 else np.inf)

vif_data = pd.DataFrame({
    "Feature": X_tech.columns,
    "VIF": np.round(vifs, 2)
})
print("Step 4: Variance Inflation Factor (VIF) Analysis")
print(vif_data.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 5: Inference / Predictions on New Engineers
# ============================================================
new_engineers = pd.DataFrame({
    "experience_years": [2.5, 8.0],
    "projects_completed": [3, 12],
    "code_review_score": [4.1, 4.9]
})
new_engineers["Salary_Predicted"] = tech_model.predict(new_engineers).round(2)

print("Step 5: Predictions for New Candidates")
print(new_engineers.to_string(index=False))
print("=" * 60)

# ============================================================
# Step 6: Actual vs Predicted Comparison & Evaluation Metrics
# ============================================================
y_tech_pred = tech_model.predict(X_tech)
comparison_df = pd.DataFrame({
    "Actual": y_tech,
    "Predicted": y_tech_pred.round(2),
    "Difference": (y_tech - y_tech_pred).round(2)
})
print("Step 6: Actual vs Predicted Salaries (First 5 Rows)")
print(comparison_df.head().to_string(index=False))

ss_total = np.sum((y_tech - np.mean(y_tech)) ** 2)
ss_res = np.sum((y_tech - y_tech_pred) ** 2)
r2_val = 1 - (ss_res / ss_total)
rmse_val = np.sqrt(np.mean((y_tech - y_tech_pred) ** 2))

print(f"\nR² Score: {r2_val:.4f}")
print(f"RMSE:     ₹{rmse_val:,.2f}")
print("=" * 60)

# ============================================================
# Step 7: Visualizing Actual vs Predicted Salary
# ============================================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_tech, y=y_tech_pred, color="teal", alpha=0.8, label="Model Predictions")
plt.plot([y_tech.min(), y_tech.max()], [y_tech.min(), y_tech.max()], color="red", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
plt.xlabel("Actual Annual Salary (₹)")
plt.ylabel("Predicted Annual Salary (₹)")
plt.title("Actual vs Predicted Tech Salary (Multiple Linear Regression)")
plt.legend()
plt.tight_layout()
plt.show()
