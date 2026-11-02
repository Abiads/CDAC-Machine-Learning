"""
Title: E-Commerce Customer Analytics Lifecycle 
Description: Simple, built-in function driven Python script for e-commerce customer analytics.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

csv_path = Path(__file__).resolve().parent / "ecommerce_customer_data.csv"


def phase_1_discovery_and_setup():
    df= pd.read_csv(csv_path)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    return df

def phase_2_descriptive_statistics(df):
    col=df['Browsing_Minutes']
    print(col.describe())
    print(f"Variance: {col.var():.2f}")
    q1= col.quantile(0.25)
    q3= col.quantile(0.75)
    iqr =q3-q1
    lower =q1-1.5*iqr
    upper=q3+1.5*iqr
    outlier=df[(col<lower)|(col>upper)]
    print(f"IQR: {iqr: .2f}")
    print(f"Outliers Count: {len(outlier)}")
    return df
def phase_3_bivariate_association(df):
    
    return df
 
 
 
 

def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)

    print("\n" + "=" * 65)
    print("E-COMMERCE LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
