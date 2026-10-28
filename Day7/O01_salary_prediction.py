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
    """Generate synthetic software engineer salary dataset."""
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
        "years_experience": years_experience.round(1),
        "certifications": certifications,
        "performance_rating": performance_rating.round(2),
        "hours_per_week": hours_per_week.round(1),
        "salary": salary.round(2)
    })
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "salary", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate features (X) and target (y), then perform train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Instantiate and train LinearRegression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Compute regression performance metrics on the test set."""
    y_pred = model.predict(X_test)
    n = len(y_test)
    p = X_test.shape[1]

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    }


def get_feature_importance(model: LinearRegression, feature_names: list) -> pd.DataFrame:
    """Format model coefficients into a ranked DataFrame."""
    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient (INR)": model.coef_
    }).sort_values(by="Coefficient (INR)", ascending=False)


def main():
    print("=" * 60)
    print("Day 7: Multi-Feature Tech Salary Prediction Pipeline")
    print("=" * 60)

    # Step 1: Generate Dataset
    print("\nStep 1: Dataset Generation & Preview (First 5 Rows)")
    df = generate_synthetic_data(n_samples=250)
    print(df.head())
    print("-" * 60)

    # Step 2: Split Data
    print("\nStep 2: Train-Test Split (80% Train, 20% Test)")
    X_train, X_test, y_train, y_test = split_features_target(df, test_size=0.2)
    print(f"Training Features Shape : {X_train.shape}")
    print(f"Testing Features Shape  : {X_test.shape}")
    print("-" * 60)

    # Step 3: Train Model
    print("\nStep 3: Training Linear Regression Model")
    model = train_linear_regression(X_train, y_train)
    print("Model training completed successfully!")
    print(f"Learned Base Intercept (Beta 0): INR {model.intercept_:,.2f}")
    print("-" * 60)

    # Step 4: Model Coefficients
    print("\nStep 4: Learned Feature Coefficients")
    coef_df = get_feature_importance(model, X_train.columns.tolist())
    print(coef_df.to_string(index=False))
    print("-" * 60)

    # Step 5: Evaluate on Test Set
    print("\nStep 5: Model Evaluation on Unseen Test Data")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Mean Absolute Error (MAE)       : INR {metrics['MAE']:,.2f}")
    print(f"Mean Squared Error (MSE)        : INR {metrics['MSE']:,.2f}")
    print(f"Root Mean Squared Error (RMSE)  : INR {metrics['RMSE']:,.2f}")
    print(f"R-squared Score (R2)            : {metrics['R2']:.4f}")
    print(f"Adjusted R-squared (Adj R2)     : {metrics['Adjusted_R2']:.4f}")
    print("-" * 60)

    # Step 6: Sample Predictions
    print("\nStep 6: Sample Predictions for New Software Engineers")
    sample_candidates = pd.DataFrame({
        "years_experience": [2.0, 5.0, 10.0],
        "certifications": [1, 3, 4],
        "performance_rating": [3.5, 4.2, 4.9],
        "hours_per_week": [40.0, 42.0, 45.0]
    })
    sample_candidates["Predicted_Salary_INR"] = model.predict(sample_candidates).round(2)
    print(sample_candidates.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
