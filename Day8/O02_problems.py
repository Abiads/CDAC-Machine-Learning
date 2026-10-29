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