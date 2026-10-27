# Assignment : Predict Sales Revenue from amount spent on digital ads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. Dataset Generation
np.random.seed(101)
n_campaigns = 40

ad_spend_k = np.random.uniform(5, 50, size=n_campaigns)  # Ad spend in ₹10,000s
revenue_k = 12 + 3.4 * ad_spend_k + np.random.normal(0, 8, size=n_campaigns)  # Revenue in ₹10,000s

df_marketing = pd.DataFrame({
    "ad_spend_inr_k": ad_spend_k.round(2),
    "revenue_inr_k": revenue_k.round(2)
})
print("Step 1: Dataset Preview (First 5 Campaigns)")
print(df_marketing.head())
print("=" * 60)

# 2. Manual OLS Calculation (Closed-Form Analytical Math)
x = df_marketing["ad_spend_inr_k"].values
y = df_marketing["revenue_inr_k"].values
x_mean = np.mean(x)
y_mean = np.mean(y)

beta_1_manual = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta_0_manual = y_mean - beta_1_manual * x_mean

print("Step 2: Manual OLS Closed-Form Parameters")
print(f"Manual Slope (Beta 1):     {beta_1_manual:.2f}")
print(f"Manual Intercept (Beta 0): {beta_0_manual:.2f}")
print("=" * 60)

# 3. Fitting with Scikit-Learn LinearRegression
X = df_marketing[["ad_spend_inr_k"]]
y_target = df_marketing["revenue_inr_k"]

model = LinearRegression()
model.fit(X, y_target)

print("Step 3: Scikit-Learn LinearRegression Parameters")
print(f"Scikit-Learn Slope (Beta 1):     {model.coef_[0]:.2f}")
print(f"Scikit-Learn Intercept (Beta 0): {model.intercept_:.2f}")
print("=" * 60)

# 4. Model Evaluation Metrics
y_pred_all = model.predict(X)
ss_total = np.sum((y - y_mean) ** 2)
ss_residual = np.sum((y - y_pred_all) ** 2)
r2_score_val = 1 - (ss_residual / ss_total)

print("Step 4: Model Evaluation Metrics")
print(f"R² Score: {r2_score_val:.4f}")
print("=" * 60)

# 5. Prediction on New Ad Spend Values
new_campaigns = pd.DataFrame({"ad_spend_inr_k": [10.0, 20.0, 35.0, 50.0]})
predicted_revenue = model.predict(new_campaigns)

print("Step 5: Predictions for New Marketing Campaigns")
for spend, rev in zip(new_campaigns["ad_spend_inr_k"], predicted_revenue):
    print(f"Predicted revenue for ₹{spend * 10000:,.0f} ad spend ({spend}k): ₹{rev * 10000:,.2f} ({rev:.2f}k)")
print("=" * 60)

# 6. Plotting Data and Regression Line
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))
sns.scatterplot(x="ad_spend_inr_k", y="revenue_inr_k", data=df_marketing, color="blue", label="Campaign Data")
plt.plot(df_marketing["ad_spend_inr_k"], y_pred_all, color="red", label=f"Fit Line: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
plt.xlabel("Ad Spend (in ₹10,000s)")
plt.ylabel("Sales Revenue (in ₹10,000s)")
plt.title("Sales Revenue vs Digital Ad Spend")
plt.legend()
plt.tight_layout()
plt.show()