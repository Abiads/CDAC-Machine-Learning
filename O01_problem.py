"""
================================================================================
Problem: Temperature vs Operating / Heating / Cooling Cost (Polynomial Regression)
CDAC Machine Learning - Day 8
================================================================================

Objective:
  Model the non-linear relationship between ambient Temperature (°C) and total
  facility Operating Cost (heating + cooling electricity) using Polynomial Regression.

Dataset Overview:
  - When Temperature is freezing (-10°C to 0°C), heating costs are extremely high.
  - When Temperature is mild (15°C to 20°C), neither heating nor cooling is needed,
    reaching the minimum operating cost (~35 - 40).
  - When Temperature is hot (25°C to 40°C), air conditioning costs drive total cost up.
  This forms a characteristic parabolic "U-curve" (Degree 2 relationship).
"""

# ==============================================================================
# IMPORTS: Required Libraries and Tools
# ==============================================================================
import numpy as np               # Used for numerical arrays, square root, and linspace
import pandas as pd              # Used for 2D DataFrames (feature matrix X) and 1D Series (target y)
import matplotlib.pyplot as plt  # Used for visualizing actual data, fitted curve, and predictions

# scikit-learn modules:
from sklearn.linear_model import LinearRegression        # Standard Ordinary Least Squares (OLS) solver
from sklearn.preprocessing import PolynomialFeatures     # Feature engineer: expands [x] into [x, x^2, ..., x^d]
from sklearn.pipeline import make_pipeline               # Sequentially chains transformer (poly) + estimator (linear reg)
from sklearn.metrics import mean_squared_error, r2_score # Metrics: Root Mean Squared Error (RMSE) and R^2 score


# ==============================================================================
# STEP 1: Dataset Formulation
# ==============================================================================
# Given Data: 11 temperature observations and their corresponding facility energy costs.
data = {
    # Independent variable X: Ambient temperature in degrees Celsius (°C)
    "temp": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
    
    # Dependent variable y: Total energy/operating cost (₹ or $)
    "cost": [220, 160, 110, 70, 45, 35, 40, 65, 65, 105, 160]
}

# Create a pandas DataFrame from dictionary
df = pd.DataFrame(data)

# Extract Feature Matrix X and Target Vector y:
# IMPORTANT:
#   - X must be a 2D matrix (shape: [n_samples, n_features] = [11, 1]).
#     We use df[["temp"]] with double brackets to keep it as a 2D DataFrame.
#   - y is a 1D vector (shape: [n_samples] = [11]).
#     We use df["cost"] with single brackets to get a 1D Series.
X = df[["temp"]]
y = df["cost"]

print("=" * 70)
print("Step 1: Dataset Preview (Ambient Temperature vs Operating Cost)")
print("=" * 70)
print(df.to_string(index=False))
print("-" * 70)
for t, c in zip(df["temp"], df["cost"]):
    print(f"  Temperature: {t:>3}°C  -->  Observed Cost: {c:>3}")
print("=" * 70)


# ==============================================================================
# STEP 2: Model Comparison Across Polynomial Degrees (Finding Best Degree)
# ==============================================================================
# Why compare degrees?
#   - Degree 1 (Linear): y = b0 + b1*x
#     Straight line with NO bend. Cannot capture the U-turn curvature (Underfitting).
#   - Degree 2 (Quadratic): y = b0 + b1*x + b2*x^2
#     Has 1 bend (Parabola / U-curve). Ideal for heating-cooling balance.
#   - Degree 3 (Cubic): y = b0 + b1*x + b2*x^2 + b3*x^3
#     Has 2 bends (S-curve).
#   - Degree 4 (Quartic): y = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
#     Has 3 bends (W-curve). May overfit noise if dataset is small.

print("\n" + "=" * 70)
print("Step 2: Model Comparison across Polynomial Degrees (1 to 4)")
print("=" * 70)

best_degree = None
lowest_rmse = float("inf")
degree_results = []

for deg in [1, 2, 3, 4]:
    # make_pipeline automates two sequential operations:
    #   1. PolynomialFeatures: converts input [x] into [x^1, x^2, ..., x^deg]
    #      include_bias=False: Do not add a column of 1s here because
    #      LinearRegression(fit_intercept=True) automatically adds the intercept b0.
    #   2. LinearRegression: fits OLS regression on the engineered polynomial features.
    model = make_pipeline(
        PolynomialFeatures(degree=deg, include_bias=False),
        LinearRegression()
    )
    
    # Train the pipeline model on X and y
    model.fit(X, y)
    
    # Predict cost for the training data
    y_pred = model.predict(X)
    
    # Calculate performance metrics:
    # - RMSE (Root Mean Squared Error): Average magnitude of prediction error in same units (Cost)
    # - R^2 Score (Coefficient of Determination): Proportion of variance explained (0.0 to 1.0)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    degree_results.append({"degree": deg, "rmse": rmse, "r2": r2, "model": model})
    
    diagnosis = "Underfitting (straight line)" if deg == 1 else "Optimal parabolic fit" if deg == 2 else "Potential overfitting"
    print(f"  Degree {deg} -> RMSE: {rmse:>7.3f} | R² Score: {r2:>7.4f}  [{diagnosis}]")
    
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_degree = deg

print("-" * 70)
print(f"--> Lowest Training RMSE: Degree {best_degree} (RMSE = {lowest_rmse:.3f})")
print("--> Recommended for Physical Interpretability: Degree 2 (Parabolic U-Curve)")
print("=" * 70)


# ==============================================================================
# STEP 3: Quadratic Polynomial Model Inspection (Degree 2)
# ==============================================================================
# In engineering and physical systems, Degree 2 provides the optimal balance of
# capturing physical curvature without overfitting:
#   Cost(T) = b0 + b1*(T) + b2*(T^2)
poly_degree = 2
poly_model = make_pipeline(
    PolynomialFeatures(degree=poly_degree, include_bias=False),
    LinearRegression()
)
poly_model.fit(X, y)

# Inspect the learned regression parameters:
# Access the 'linearregression' step inside the pipeline
lr_step = poly_model.named_steps["linearregression"]
intercept = lr_step.intercept_       # b0 (intercept)
coefs = lr_step.coef_                # [b1, b2] (coefficients for T and T^2)

print("\n" + "=" * 70)
print(f"Step 3: Fitted Quadratic Equation (Degree {poly_degree})")
print("=" * 70)
print(f"  Mathematical Formula:")
print(f"    Cost(T) = {intercept:.2f} + ({coefs[0]:.2f} * T) + ({coefs[1]:.2f} * T²)")
print(f"\n  Parameter Breakdown:")
print(f"    - Intercept (b0):           {intercept:.2f}  (Estimated cost when ambient temp is 0°C)")
print(f"    - Linear Coeff (b1):        {coefs[0]:.2f}  (Downward slope due to reduced heating need)")
print(f"    - Quadratic Coeff (b2):     {coefs[1]:.2f}  (Curvature turning cost upward at high temps)")
print("=" * 70)


# ==============================================================================
# STEP 4: Predictions on Unseen / Intermediate Temperatures
# ==============================================================================
# Predict operating cost for new temperatures not present in the training set:
# Example: -7.5°C (sub-zero), -2.5°C (cold), 2.5°C (chilly), 7.5°C (cool), 12.0°C (mild)
new_temperatures = pd.DataFrame({"temp": [-7.5, -2.5, 2.5, 7.5, 12.0]})

# The pipeline automatically transforms new_temperatures into [T, T^2] before predicting!
predicted_costs = poly_model.predict(new_temperatures)

print("\n" + "=" * 70)
print("Step 4: Predictions on New / Intermediate Temperatures")
print("=" * 70)
for t, c in zip(new_temperatures["temp"], predicted_costs):
    print(f"  Predicted Cost at Temperature {t:>5.1f}°C : {c:>6.2f}")
print("=" * 70)


# ==============================================================================
# STEP 5: Visualization (Actual Points vs Polynomial Curve vs Linear Line)
# ==============================================================================
try:
    # Generate 300 evenly spaced temperature values from (min - 2) to (max + 2).
    # Why? If we only plot predictions at the 11 original points, the line would look
    # jagged with straight segments. 300 points create a smooth, continuous curve.
    x_smooth = pd.DataFrame({"temp": np.linspace(df["temp"].min() - 2, df["temp"].max() + 2, 300)})
    y_smooth = poly_model.predict(x_smooth)

    # Fit a simple linear model (Degree 1) on the same data for visual comparison
    linear_model = LinearRegression().fit(X, y)
    y_linear_smooth = linear_model.predict(x_smooth)

    # Plot setup
    plt.figure(figsize=(10, 5.5))

    # 1. Plot actual training observations as blue dots
    plt.scatter(df["temp"], df["cost"], color="royalblue", s=80, zorder=5, label="Actual Observed Data (11 points)")

    # 2. Plot new predictions as red 'x' markers
    plt.scatter(new_temperatures["temp"], predicted_costs, color="crimson", marker="x", s=90, linewidth=2, zorder=6, label="New Predictions")

    # 3. Plot Degree 2 fitted polynomial curve as a smooth green line
    plt.plot(x_smooth["temp"], y_smooth, color="forestgreen", linewidth=2.5, label=f"Polynomial Fit (Degree {poly_degree} - Parabola)")

    # 4. Plot Degree 1 linear regression as an orange dashed line to highlight underfitting
    plt.plot(x_smooth["temp"], y_linear_smooth, color="darkorange", linestyle="--", linewidth=1.8, label="Linear Baseline (Degree 1 - Underfitting)")

    # Chart formatting
    plt.title("Temperature vs Operating Cost (Polynomial Regression Analysis)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Ambient Temperature (°C)", fontsize=11)
    plt.ylabel("Operating Energy Cost", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", loc="upper center")
    plt.tight_layout()

    # Save figure to file so it is viewable anytime (e.g. on GitHub or headless servers)
    plot_file = "temperature_cost_polynomial_fit.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\n[Saved] Regression plot saved successfully to: {plot_file}")

    # Display plot window (interactive mode)
    plt.show()

except Exception as e:
    print(f"\n[Note] Plot rendering skipped in non-GUI environment ({e}). Plot file saved.")
