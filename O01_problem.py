"""
Problem: Temperature vs Operating / Heating Cost (Polynomial Regression)
CDAC Machine Learning - Day 8 Problem Solution

Given Data:
  Temperature (temp in °C): [-10, -5, 0, 5, 10]
  Cost (operating / heating energy cost): [220, 160, 120, 100, 100]

Note:
  If your classroom notes have specific values for temperatures (e.g. up to 35°C
  for full seasonal U-curve), you can easily update the 'data' dictionary below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# 1. Dataset Formulation
# ==============================================================================
# Temperature (°C) as independent feature X
# Operating / Heating Cost as target y
data = {
    "temp": [-10, -5, 0, 5, 10],
    "cost": [220, 160, 120, 100, 100]
}

# (Optional: Extended 10-point U-curve including cooling costs up to 35°C)
# data = {
#     "temp": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35],
#     "cost": [220, 160, 110, 70, 45, 35, 40, 65, 105, 160]
# }

df = pd.DataFrame(data)
X = df[["temp"]]
y = df["cost"]

print("=" * 65)
print("Step 1: Dataset Preview (Temperature vs Operating Cost)")
print("=" * 65)
print(df.to_string(index=False))
print("-" * 65)
for t, c in zip(df["temp"], df["cost"]):
    print(f"  Temperature: {t:>3}°C  -->  Cost: {c:>3}")
print("=" * 65)


# ==============================================================================
# 2. Finding the Best Degree for Polynomial Regression
# ==============================================================================
print("\n" + "=" * 65)
print("Step 2: Model Comparison across Polynomial Degrees (1 to 4)")
print("=" * 65)

best_degree = None
lowest_rmse = float("inf")
degree_results = []

for deg in [1, 2, 3, 4]:
    # Build pipeline: Polynomial Features -> Linear Regression
    model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    degree_results.append({
        "degree": deg,
        "rmse": rmse,
        "r2": r2,
        "model": model
    })
    
    print(f"  Degree {deg} -> RMSE: {rmse:>7.3f} | R² Score: {r2:>7.4f}")
    
    # We prefer degree 2 for quadratic parabolic relationships
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_degree = deg

print("-" * 65)
print(f"--> Best Degree by Lowest RMSE: Degree {best_degree} (RMSE = {lowest_rmse:.3f})")
print("=" * 65)


# ==============================================================================
# 3. Quadratic Polynomial Model Details (Degree 2)
# ==============================================================================
# Degree 2 captures the natural curvature: Cost(T) = b0 + b1*(T) + b2*(T^2)
poly_degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression())
poly_model.fit(X, y)

# Extract learned parameters
lr_step = poly_model.named_steps["linearregression"]
intercept = lr_step.intercept_
coefs = lr_step.coef_

print("\n" + "=" * 65)
print(f"Step 3: Fitted Quadratic Equation (Degree {poly_degree})")
print("=" * 65)
print(f"  Cost(T) = {intercept:.2f} + ({coefs[0]:.2f} * T) + ({coefs[1]:.2f} * T²)")
print(f"  - Intercept (b0):           {intercept:.2f}  (Estimated cost at 0°C)")
print(f"  - Linear Coefficient (b1):  {coefs[0]:.2f}")
print(f"  - Quadratic Coeff (b2):     {coefs[1]:.2f}  (Rate of curvature)")
print("=" * 65)


# ==============================================================================
# 4. Predictions on Intermediate and New Temperatures
# ==============================================================================
new_temperatures = pd.DataFrame({"temp": [-7.5, -2.5, 2.5, 7.5, 12.0]})
predicted_costs = poly_model.predict(new_temperatures)

print("\n" + "=" * 65)
print("Step 4: Predictions on Intermediate Temperatures")
print("=" * 65)
for t, c in zip(new_temperatures["temp"], predicted_costs):
    print(f"  Predicted Cost at Temperature {t:>5.1f}°C : {c:>6.2f}")
print("=" * 65)


# ==============================================================================
# 5. Visualization: Actual Data vs Polynomial Fitted Curve
# ==============================================================================
try:
    # Smooth line generation across temperature range
    x_smooth = pd.DataFrame({"temp": np.linspace(df["temp"].min() - 2, df["temp"].max() + 2, 300)})
    y_smooth = poly_model.predict(x_smooth)

    # Linear comparison line (Degree 1)
    linear_model = LinearRegression().fit(X, y)
    y_linear_smooth = linear_model.predict(x_smooth)

    plt.figure(figsize=(9, 5))
    # Actual observations
    plt.scatter(df["temp"], df["cost"], color="blue", s=70, zorder=5, label="Actual Data Points")
    # Predicted new points
    plt.scatter(new_temperatures["temp"], predicted_costs, color="red", marker="x", s=80, zorder=6, label="New Predictions")
    # Fitted curves
    plt.plot(x_smooth["temp"], y_smooth, color="green", linewidth=2.2, label=f"Polynomial Fit (Degree {poly_degree})")
    plt.plot(x_smooth["temp"], y_linear_smooth, color="orange", linestyle="--", linewidth=1.5, label="Linear Fit (Degree 1)")

    plt.title("Temperature vs Operating Cost (Polynomial Regression)", fontsize=13, fontweight="bold")
    plt.xlabel("Temperature (°C)", fontsize=11)
    plt.ylabel("Cost", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save figure to disk so it is accessible in terminal / headless environments
    plot_file = "temperature_cost_polynomial_fit.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\n[Saved] Plot successfully saved to: {plot_file}")
    plt.show()
except Exception as e:
    print(f"\n[Note] Plot rendering skipped ({e}). Plot logic successfully executed.")
