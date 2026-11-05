"""
================================================================================
EXAM CRASH COUNTERMEASURES: 8 COMMON ERRORS & HOW TO FIX THEM
================================================================================
This script demonstrates the 8 most frequent errors that break machine learning
code during practical exams and tests.

Each section demonstrates:
  1. BUGGY CODE: Triggers the real Python error (caught with try/except so you see it).
  2. FIX CODE:   The exact 1-line or 2-line crash-proof countermeasure.

Run directly:
  python Day11/All_In_One_Practice/test.py
================================================================================
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression


# ==============================================================================
# ERROR 1: File Loading & Terminal CWD Mismatch
# ==============================================================================
def demo_error_1():
    print("\n" + "=" * 70)
    print("ERROR 1: File Path & Current Working Directory (CWD) Trap")
    print("=" * 70)

    print("[BUGGY CODE] pd.read_csv('coffee_shop_data.csv') from terminal root:")
    try:
        # Relative path without base directory lookup
        pd.read_csv("non_existent_relative_file.csv")
    except FileNotFoundError as e:
        print(f"--> CRASH: FileNotFoundError: {e}")

    print("\n[CRASH-PROOF FIX] Using Path(__file__).resolve().parent:")
    base_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    # Find dataset dynamically checking parent / subfolders
    dataset_path = None
    for candidate in [base_dir, base_dir.parent / "Practice1", Path.cwd() / "Day11" / "Practice1"]:
        if (candidate / "coffee_shop_data.csv").is_file():
            dataset_path = candidate / "coffee_shop_data.csv"
            break
            
    if dataset_path:
        df = pd.read_csv(dataset_path)
        print(f"--> SUCCESS: Loaded '{dataset_path.name}' ({df.shape[0]} rows, {df.shape[1]} cols)")


# ==============================================================================
# ERROR 2: Expected 2D Array vs 1D Array (ValueError)
# ==============================================================================
def demo_error_2():
    print("\n" + "=" * 70)
    print("ERROR 2: Expected 2D Array vs 1D Array in Scikit-Learn")
    print("=" * 70)

    # Synthetic sample data
    df = pd.DataFrame({
        'experience_years': [1, 2, 3, 4, 5],
        'monthly_salary': [3000, 4200, 5100, 6300, 7500]
    })

    print("[BUGGY CODE] Passing 1D Series X = df['experience_years'] to LinearRegression:")
    try:
        X_1d = df['experience_years']  # Single brackets -> 1D Series
        y = df['monthly_salary']
        model = LinearRegression().fit(X_1d, y)
    except ValueError as e:
        print(f"--> CRASH: ValueError: {e}")

    print("\n[CRASH-PROOF FIX] Double brackets df[['experience_years']] creates a 2D Matrix:")
    X_2d = df[['experience_years']]  # DOUBLE BRACKETS -> 2D DataFrame
    y = df['monthly_salary']        # SINGLE BRACKET -> 1D Series Target
    model = LinearRegression().fit(X_2d, y)
    print(f"--> SUCCESS: LinearRegression fit successfully! Slope = {model.coef_[0]:.2f}")


# ==============================================================================
# ERROR 3: Data Leakage in Feature Scaling
# ==============================================================================
def demo_error_3():
    print("\n" + "=" * 70)
    print("ERROR 3: Data Leakage in Feature Scaling (Exam Marks Penalty)")
    print("=" * 70)

    X_train = np.array([[10], [20], [30], [40]])
    X_test = np.array([[50], [60]])

    print("[BUGGY CODE] Calling fit_transform on X_test:")
    scaler_bad = StandardScaler()
    X_train_bad = scaler_bad.fit_transform(X_train)
    X_test_bad = scaler_bad.fit_transform(X_test)  # WRONG: Leaks test set mean/std
    print("--> PENALTY: Test set distribution was re-fit, destroying true evaluation!")

    print("\n[CRASH-PROOF FIX] fit_transform on Train, ONLY transform on Test:")
    scaler_good = StandardScaler()
    X_train_good = scaler_good.fit_transform(X_train)  # Learn mu, sigma from Train
    X_test_good = scaler_good.transform(X_test)       # Apply Train's mu, sigma to Test
    print(f"--> SUCCESS: Train mean = {scaler_good.mean_[0]:.1f} | Test transformed using train mean.")


# ==============================================================================
# ERROR 4: Division by Zero Creating `inf` / `NaN` Inputs
# ==============================================================================
def demo_error_4():
    print("\n" + "=" * 70)
    print("ERROR 4: Division by Zero & `inf` Values in Feature Engineering")
    print("=" * 70)

    df = pd.DataFrame({
        'customer_id': [101, 102, 103],
        'total_charges': [300.0, 150.0, 0.0],
        'tenure_months': [10, 0, 0]  # Note zero tenure for new signups!
    })

    print("[BUGGY CODE] Direct division by zero: df['total_charges'] / df['tenure_months']:")
    X_bad = df.copy()
    X_bad['avg_monthly_spend'] = X_bad['total_charges'] / X_bad['tenure_months']
    print(f"Columns contain inf/NaN:\n{X_bad[['total_charges', 'tenure_months', 'avg_monthly_spend']]}")
    try:
        LinearRegression().fit(X_bad[['avg_monthly_spend']], [1, 2, 3])
    except ValueError as e:
        print(f"--> CRASH: ValueError: {e}")

    print("\n[CRASH-PROOF FIX] Replace 0 with 1 before division:")
    X_good = df.copy()
    X_good['avg_monthly_spend'] = X_good['total_charges'] / X_good['tenure_months'].replace(0, 1)
    print(f"--> SUCCESS: Sanitized avg_monthly_spend:\n{X_good[['total_charges', 'tenure_months', 'avg_monthly_spend']]}")


# ==============================================================================
# ERROR 5: Dirty Datetime String Crashes
# ==============================================================================
def demo_error_5():
    print("\n" + "=" * 70)
    print("ERROR 5: Dirty Datetime Strings Crashing pd.to_datetime")
    print("=" * 70)

    raw_dates = pd.Series(["January 15 1990", "March 22 1985", "Unknown Date", "1995-07-04"])

    print("[BUGGY CODE] pd.to_datetime(raw_dates) without error handling:")
    try:
        pd.to_datetime(raw_dates)
    except Exception as e:
        print(f"--> CRASH: {type(e).__name__}: {e}")

    print("\n[CRASH-PROOF FIX] Using errors='coerce' turns bad dates safely into NaT:")
    clean_dates = pd.to_datetime(raw_dates, errors='coerce')
    print(f"--> SUCCESS: Clean dates:\n{clean_dates}")
    print(f"Missing count: {clean_dates.isnull().sum()} records handled gracefully.")


# ==============================================================================
# ERROR 6: String Lookup Mismatch & Hall of Fame Asterisks (*)
# ==============================================================================
def demo_error_6():
    print("\n" + "=" * 70)
    print("ERROR 6: Exact String Equality vs Substring Search (NBA Hall of Fame *)")
    print("=" * 70)

    # In player_data.csv, Hall of Fame superstars have an asterisk: "Michael Jordan*"
    df_players = pd.DataFrame({
        'player': ["Alaa Abdelnaby", "Kareem Abdul-Jabbar*", "Michael Jordan*", "Kobe Bryant*"],
        'birth_date': ["1968-06-24", "1947-04-16", "1963-02-17", "1978-08-23"]
    })

    search_target = "Michael Jordan"

    print(f"[BUGGY CODE] Exact equality df['player'] == '{search_target}':")
    bad_match = df_players[df_players['player'] == search_target]
    print(f"--> FAILED: Found {len(bad_match)} records (missed because of asterisk '*')")

    print(f"\n[CRASH-PROOF FIX] df['player'].str.contains(search_target, case=False, na=False):")
    good_match = df_players[df_players['player'].str.contains(search_target, case=False, na=False)]
    print(f"--> SUCCESS: Found {len(good_match)} record: {good_match.iloc[0]['player']} (Born: {good_match.iloc[0]['birth_date']})")


# ==============================================================================
# ERROR 7: Categorical Dummy Variable Trap (Multicollinearity)
# ==============================================================================
def demo_error_7():
    print("\n" + "=" * 70)
    print("ERROR 7: Categorical Dummy Variable Trap (pd.get_dummies)")
    print("=" * 70)

    df_house = pd.DataFrame({
        'sqft': [1500, 2000, 2500],
        'location': ['Rural', 'Suburban', 'Urban']
    })

    print("[BUGGY CODE] pd.get_dummies(df, columns=['location']) without drop_first:")
    encoded_bad = pd.get_dummies(df_house, columns=['location'])
    print(f"--> Columns generated (3 dummies for 3 levels -> redundant):\n{list(encoded_bad.columns)}")

    print("\n[CRASH-PROOF FIX] pd.get_dummies(df, columns=['location'], drop_first=True):")
    encoded_good = pd.get_dummies(df_house, columns=['location'], drop_first=True)
    print(f"--> SUCCESS: Columns generated (2 dummies for 3 levels, prevents collinearity):\n{list(encoded_good.columns)}")


# ==============================================================================
# ERROR 8: Sample Inference on Scaled Models
# ==============================================================================
def demo_error_8():
    print("\n" + "=" * 70)
    print("ERROR 8: Testing Sample Inference with Raw Unscaled Numbers")
    print("=" * 70)

    # Train a model on scaled data
    X_raw = np.array([[20, 20000], [40, 60000], [60, 100000]])
    y = np.array([0, 1, 1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    clf = LogisticRegression().fit(X_scaled, y)

    sample_applicant = [[30, 45000]]  # Age 30, Income $45,000

    print("[BUGGY CODE] Feeding raw values directly into clf.predict(sample_applicant):")
    pred_bad = clf.predict(sample_applicant)
    print(f"--> WRONG PREDICTION: Model received huge raw values (45000) instead of z-scores -> Pred: {pred_bad[0]}")

    print("\n[CRASH-PROOF FIX] Always pass sample through scaler.transform() first:")
    sample_scaled = scaler.transform(sample_applicant)
    pred_good = clf.predict(sample_scaled)
    print(f"--> SUCCESS: Scaled sample = {sample_scaled.round(2)} | Correct Pred: {pred_good[0]}")


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
if __name__ == "__main__":
    demo_error_1()
    demo_error_2()
    demo_error_3()
    demo_error_4()
    demo_error_5()
    demo_error_6()
    demo_error_7()
    demo_error_8()
    print("\n" + "=" * 70)
    print("ALL 8 EXAM ERROR DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
