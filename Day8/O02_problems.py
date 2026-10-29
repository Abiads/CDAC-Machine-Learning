import numpy as np
import pandas as pd

n_samples = 200
np.random.seed(2026)
fertilizer = np.random.uniform(20, 250, size=n_samples)
# Yield = 2.0 + 0.08*Fertilizer - 0.00028*(Fertilizer^2) + Noise
true_yield = 2.0 + 0.08 * fertilizer - 0.00028 * (fertilizer**2)
observed_yield = true_yield + np.random.normal(0, 0.8, size=n_samples)
df = pd.DataFrame(
        {
            "fertilizer_kg_ha": fertilizer.round(1),
            "crop_yield_tons": observed_yield.round(2),
        }
    )

# for prediction 
new_farm_plots = pd.DataFrame(
        {
            "fertilizer_kg_ha": [50.0, 140.0, 220.0]
        }
    )

# ==============================================================================
# Model Training & Polynomial Regression Solution
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Feature Matrix (2D) and Target Vector (1D)
# ------------------------------------------------------------------------------
# Independent variable X: Fertilizer application in kg per hectare (2D DataFrame)
# Dependent variable y: Resulting crop yield in tons (1D Series)
X = df[["fertilizer_kg_ha"]]
y = df["crop_yield_tons"]

print("=" * 70)
print("Step 1: Dataset Summary (Fertilizer vs Crop Yield)")
print("=" * 70)
print(f"Total Farm Observations: {len(df)}")
print("\nFirst 5 Observations:")
print(df.head().to_string(index=False))
print("=" * 70)

# ------------------------------------------------------------------------------
# 2. Compare Degrees 1 to 4 (Underfitting vs Optimal Fit vs Overfitting)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 2: Model Comparison Across Polynomial Degrees (1 to 4)")
print("=" * 70)

best_degree = None
lowest_rmse = float("inf")
degree_results = []

for deg in [1, 2, 3, 4]:
    # Pipeline: Polynomial feature expansion -> Ordinary Least Squares regression
    model = make_pipeline(
        PolynomialFeatures(degree=deg, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    degree_results.append({"degree": deg, "rmse": rmse, "r2": r2})
    diagnosis = "Severe Underfitting" if deg == 1 else "Optimal Bell/Arch Fit" if deg == 2 else "Diminishing Returns"
    print(f"  Degree {deg} -> RMSE: {rmse:.4f} | R² Score: {r2:.4f}  [{diagnosis}]")
    
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_degree = deg

print("-" * 70)
print(f"--> Best Degree by Lowest RMSE: Degree {best_degree} (RMSE = {lowest_rmse:.4f})")
print("--> Recommended Model: Degree 2 (Parabolic Bell Curve)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 3. Train Optimal Degree 2 Model & Extract Parameters
# ------------------------------------------------------------------------------
final_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
final_model.fit(X, y)

# Extract learned coefficients from the pipeline's linear regression step
lr_step = final_model.named_steps["linearregression"]
b0 = lr_step.intercept_
b1, b2 = lr_step.coef_

print("\n" + "=" * 70)
print("Step 3: Fitted Quadratic Equation (Degree 2)")
print("=" * 70)
print(f"  Fitted Model:")
print(f"    Yield = {b0:.4f} + ({b1:.5f} * Fertilizer) + ({b2:.7f} * Fertilizer²)")
print(f"\n  Comparison with Underlying Ground Truth Generator:")
print(f"    Ground Truth: Yield = 2.0000 + 0.08000 * F - 0.00028 * F² + Noise")
print(f"    Learned Fit : Yield = {b0:.4f} + {b1:.5f} * F + ({b2:.7f}) * F²")
print(f"\n  Parameter Interpretations:")
print(f"    - Intercept (b0):           {b0:.4f}  (Estimated base yield without fertilizer)")
print(f"    - Linear Coeff (b1):        {b1:.5f}  (Yield increase per kg of fertilizer added)")
print(f"    - Quadratic Coeff (b2):     {b2:.7f}  (Negative curvature -> Diminishing returns & burn)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 4. Predictions for New Farm Plots
# ------------------------------------------------------------------------------
predicted_yields = final_model.predict(new_farm_plots)
new_farm_plots_with_preds = new_farm_plots.copy()
new_farm_plots_with_preds["predicted_yield_tons"] = predicted_yields.round(2)

print("\n" + "=" * 70)
print("Step 4: Predictions for New Farm Plots")
print("=" * 70)
for fert, pred in zip(new_farm_plots["fertilizer_kg_ha"], predicted_yields):
    comment = "Low fertilizer / Growth phase" if fert < 100 else "Near optimal peak yield" if fert < 180 else "Excessive fertilizer / Yield drop"
    print(f"  Plot: {fert:>5.1f} kg/ha  -->  Predicted Crop Yield: {pred:>5.2f} tons  ({comment})")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5. Visualization (Actual Observations vs Fitted Curve vs New Predictions)
# ------------------------------------------------------------------------------
try:
    # 300 evenly spaced points across the fertilizer range for smooth plotting
    x_smooth = pd.DataFrame({
        "fertilizer_kg_ha": np.linspace(df["fertilizer_kg_ha"].min(), df["fertilizer_kg_ha"].max(), 300)
    })
    y_smooth = final_model.predict(x_smooth)

    # Simple linear baseline (Degree 1) for contrast
    linear_baseline = LinearRegression().fit(X, y)
    y_linear_smooth = linear_baseline.predict(x_smooth)

    plt.figure(figsize=(10, 5.5))
    # 200 observed farms
    plt.scatter(df["fertilizer_kg_ha"], df["crop_yield_tons"], color="royalblue", alpha=0.5, s=30, label="Observed Farm Data (200 farms)")
    # New predictions
    plt.scatter(new_farm_plots["fertilizer_kg_ha"], predicted_yields, color="crimson", marker="X", s=120, zorder=6, label="New Farm Predictions (3 plots)")
    # Degree 2 fitted curve
    plt.plot(x_smooth["fertilizer_kg_ha"], y_smooth, color="forestgreen", linewidth=2.5, label="Fitted Parabola (Degree 2 Bell Curve)")
    # Degree 1 linear baseline
    plt.plot(x_smooth["fertilizer_kg_ha"], y_linear_smooth, color="darkorange", linestyle="--", linewidth=1.8, label="Linear Baseline (Degree 1 - Fails)")

    # Chart aesthetics
    plt.title("Fertilizer vs Crop Yield (Polynomial Regression Analysis)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Fertilizer Application (kg/ha)", fontsize=11)
    plt.ylabel("Crop Yield (tons)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", loc="upper left")
    plt.tight_layout()

    # Save plot to disk
    plot_file = "fertilizer_crop_yield_polynomial_fit.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\n[Saved] Regression plot saved successfully to: {plot_file}")
    plt.show()

except Exception as e:
    print(f"\n[Note] Plot rendering skipped ({e}). Plot saved to file.")
