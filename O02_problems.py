import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 1. Dataset Generation (Given in Lecture)
# ==============================================================================
n_samples = 200
np.random.seed(2026)

# Generate 200 random fertilizer amounts between 20 kg/ha and 250 kg/ha
fertilizer = np.random.uniform(20, 250, size=n_samples)

# True mathematical relation: inverted parabola (arch/bell curve)
# Yield = 2.0 + 0.08*Fertilizer - 0.00028*(Fertilizer^2) + Noise
true_yield = 2.0 + 0.08 * fertilizer - 0.00028 * (fertilizer**2)
observed_yield = true_yield + np.random.normal(0, 0.8, size=n_samples)

# Create DataFrame
df = pd.DataFrame(
    {
        "fertilizer_kg_ha": fertilizer.round(1),
        "crop_yield_tons": observed_yield.round(2),
    }
)

# New farm plots given for prediction
new_farm_plots = pd.DataFrame(
    {
        "fertilizer_kg_ha": [50.0, 140.0, 220.0]
    }
)

print("--- First 5 Rows of Dataset ---")
print(df.head())
print("=" * 60)


# ==============================================================================
# 2. Split Features (X) and Target (y)
# ==============================================================================
# X must be 2D DataFrame: df[["column_name"]]
# y is 1D Series: df["column_name"]
X = df[["fertilizer_kg_ha"]]
y = df["crop_yield_tons"]


# ==============================================================================
# 3. Check Best Degree (Degrees 1, 2, 3, 4)
# ==============================================================================
print("--- Finding the Best Degree ---")
for deg in [1, 2, 3, 4]:
    model_deg = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
    model_deg.fit(X, y)
    y_pred = model_deg.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Degree {deg} RMSE: {rmse:.2f}")

print("--> Degree 2 is best (parabola with 1 bend)!")
print("=" * 60)


# ==============================================================================
# 4. Train Degree 2 Polynomial Regression Model
# ==============================================================================
# Pipeline:
#   Step 1: PolynomialFeatures turns [x] into [x, x^2]
#   Step 2: LinearRegression fits y = b0 + b1*x + b2*(x^2)
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model.fit(X, y)

# Print learned parameters: intercept (b0) and coefficients (b1, b2)
lr = model.named_steps["linearregression"]
print(f"Intercept (b0)          : {lr.intercept_:.2f}")
print(f"Linear Coeff (b1)       : {lr.coef_[0]:.5f}")
print(f"Quadratic Coeff (b2)    : {lr.coef_[1]:.7f}")
print(f"Equation: Yield = {lr.intercept_:.2f} + ({lr.coef_[0]:.4f} * Fert) + ({lr.coef_[1]:.6f} * Fert^2)")
print("=" * 60)


# ==============================================================================
# 5. Predict for New Farm Plots
# ==============================================================================
predicted_yield = model.predict(new_farm_plots)

print("--- Predictions for New Farm Plots ---")
for fert, pred in zip(new_farm_plots["fertilizer_kg_ha"], predicted_yield):
    print(f"Fertilizer: {fert:>5.1f} kg/ha  -->  Predicted Crop Yield: {pred:.2f} tons")
print("=" * 60)


# ==============================================================================
# 6. Visualize Results (Actual vs Fitted Bell Curve)
# ==============================================================================
# 300 points for a smooth curved line (same as lecture O02 and O04)
x_smooth = pd.DataFrame({"fertilizer_kg_ha": np.linspace(df["fertilizer_kg_ha"].min(), df["fertilizer_kg_ha"].max(), 300)})
y_smooth = model.predict(x_smooth)

# Scatter plot: actual data (blue) and new predictions (red)
plt.scatter(df["fertilizer_kg_ha"], df["crop_yield_tons"], color="blue", alpha=0.5, label="Actual Data")
plt.scatter(new_farm_plots["fertilizer_kg_ha"], predicted_yield, color="red", marker="X", s=100, label="Predicted")

# Fitted bell curve (green)
plt.plot(x_smooth["fertilizer_kg_ha"], y_smooth, color="green", linewidth=2, label="Fitted Curve (Bell)")

plt.xlabel("Fertilizer (kg/ha)")
plt.ylabel("Crop Yield (tons)")
plt.title("Fertilizer vs Crop Yield (Polynomial Regression)")
plt.legend()
plt.tight_layout()
plt.savefig("fertilizer_crop_yield_polynomial_fit.png", dpi=150)
plt.show()
