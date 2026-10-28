from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split



def generate_synthetic_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    years_experience = np.random.uniform(1, 15, size=n_samples)
    certifications = np.random.poisson(lam=2, size=n_samples)
    performance_rating = np.random.uniform(3.0, 5.0, size=n_samples)
    hours_per_week = np.random.normal(40, 4, size=n_samples)
      # Underlying formula with noise:
    # Base 45k + 9.5k*(Exp) + 3.5k*(Certs) + 12k*(Perf) + 200*(Hours) + Gaussian Noise
    salary = (
        45000
        + 9500 * years_experience
        + 3500 * certifications
        + 12000 * performance_rating
        + 200 * hours_per_week
        + np.random.normal(0, 6000, size=n_samples)
    )
    df = pd.DataFrame({
        "years_experience": years_experience,
        "certifications": certifications,
        "performance_rating": performance_rating,
        "hours_per_week": hours_per_week,
        "salary": salary
    })
    return df


def main():
    print("=" * 60)
    print("Salary Prediction")
    df = generate_synthetic_data()
    print(df.head())
    print("=" * 60)

if __name__ == "__main__":
    main()