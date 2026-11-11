"""
================================================================================
CDAC PGCP-AI: PRACTICAL MACHINE LEARNING - DAY 11 CONSOLIDATED MASTER PIPELINE
================================================================================
File: day11_all_ml_models_master.py
Author: abs6187 <23f2000876@ds.study.iitm.ac.in>
Course: CDAC PGCP-AI (Post Graduate Certificate Programme in Artificial Intelligence)
Topic: Comprehensive Machine Learning Suite - Classification, Regression, Clustering,
       Defensive Exam Countermeasures, and Syllabus Telco Churn Benchmark Pipeline.

CONTENTS AT A GLANCE:
  - SECTION 1: Bank Loan Default Risk Classification (Logistic Regression)
  - SECTION 2: Customer Churn Retention Classification (Random Forest Classifier)
  - SECTION 3: Real Estate Valuation & Price Prediction (Linear Regression)
  - SECTION 4: Customer Behavioral Segmentation (K-Means Clustering)
  - SECTION 5: Cricket Player Archetype Segmentation (K-Means Clustering)
  - SECTION 6: Healthcare Hypertension Risk Classification (Logistic Regression)
  - SECTION 7: Exam Crash Countermeasures & Defensive Coding Lab (8 Fatal Traps & Fixes)
  - SECTION 8 (LAST): Syllabus Benchmark Pipeline - Telco Customer Churn (LogReg & RF)

USAGE:
  Run all pipelines:
    python day11_all_ml_models_master.py
  Run a specific section:
    python day11_all_ml_models_master.py --section 1   (Bank Loan Risk)
    python day11_all_ml_models_master.py --section 2   (Customer Churn RF)
    python day11_all_ml_models_master.py --section 3   (Real Estate Regression)
    python day11_all_ml_models_master.py --section 4   (Customer Segmentation KMeans)
    python day11_all_ml_models_master.py --section 5   (Cricket Player Clustering)
    python day11_all_ml_models_master.py --section 6   (Healthcare Risk Classification)
    python day11_all_ml_models_master.py --section 7   (8 Exam Crash Countermeasures)
    python day11_all_ml_models_master.py --section 8   (Syllabus Telco Customer Churn)
================================================================================
"""

import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Headless matplotlib configuration to prevent GUI window blocking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)
import joblib

# Filter non-critical runtime warnings
warnings.filterwarnings('ignore')

# Base directory anchor (resolves repository root regardless of current terminal working directory)
ROOT_DIR = Path(__file__).resolve().parent


# ==============================================================================
# SECTION 1: BANK LOAN DEFAULT RISK CLASSIFICATION (LOGISTIC REGRESSION)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. Logistic Regression models the probability that an applicant will default (y=1):
       P(y=1|X) = sigma(z) = 1 / (1 + exp(-z))
       where z = beta_0 + beta_1 * x_1 + ... + beta_p * x_p (Log-Odds / Logit).
  2. Loss Function: Binary Cross-Entropy (Log-Loss):
       J(theta) = - (1/m) * sum [ y_i * log(p_hat_i) + (1 - y_i) * log(1 - p_hat_i) ]
  3. Feature Engineering:
       loan_to_income_ratio = loan_amount_k / annual_income_k
       Captures debt leverage: higher debt-to-income strongly correlates with default risk.
  4. Best Practices Applied:
       - Categorical variables one-hot encoded with `drop_first=True` to eliminate multicollinearity.
       - Stratified train-test split (`stratify=y`) preserves minority default class proportions.
       - `StandardScaler` is fit EXCLUSIVELY on training data to prevent evaluation data leakage.
--------------------------------------------------------------------------------
"""
def run_section_1_bank_loan():
    print("\n" + "=" * 80)
    print("SECTION 1: BANK LOAN DEFAULT RISK CLASSIFICATION (LOGISTIC REGRESSION)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "ML_Practice" / "Bank_Loan_Risk" / "loan_risk_data.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'applicant_id': range(1001, 1001 + n),
            'age': np.random.randint(21, 65, size=n),
            'annual_income_k': np.random.uniform(25, 150, size=n).round(1),
            'credit_score': np.random.randint(480, 850, size=n),
            'open_credit_lines': np.random.randint(1, 12, size=n),
            'loan_amount_k': np.random.uniform(5, 60, size=n).round(1),
            'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], size=n, p=[0.4, 0.2, 0.4])
        })
        dti = df['loan_amount_k'] / df['annual_income_k']
        risk_score = 1.5 * dti - 0.005 * df['credit_score'] + 0.3 * (df['home_ownership'] == 'RENT')
        df['default_flag'] = (risk_score > np.percentile(risk_score, 75)).astype(int)
    else:
        df = pd.read_csv(csv_path)

    print(f"[Phase 1] Ingested Dataset: {df.shape[0]} applicants, {df.shape[1]} columns")
    print("Class Balance:\n", df['default_flag'].value_counts(normalize=True).round(3))

    # Phase 2: Feature Engineering & One-Hot Encoding
    X = df.drop(columns=['applicant_id', 'default_flag'])
    y = df['default_flag']
    X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"[Phase 2] Features Engineered. Transformed Matrix Columns ({len(X_encoded.columns)}):", list(X_encoded.columns))

    # Phase 3: Stratified Splitting & Scaling (Anti-Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Phase 4: Model Training & Evaluation
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[Phase 4] Model Evaluation:")
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    print("\n  Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Phase 5: Persistence & Inference Verification
    artifacts_dir = ROOT_DIR / "Day11" / "ML_Practice" / "Bank_Loan_Risk"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "loan_risk_model.joblib")
    joblib.dump(scaler, artifacts_dir / "loan_risk_scaler.joblib")
    print("[Phase 5] Saved artifacts to:", artifacts_dir)

    # Sample Inference Demonstration
    sample_data = X_encoded.iloc[[0]].copy()
    sample_scaled = scaler.transform(sample_data)
    sample_prob = model.predict_proba(sample_scaled)[0, 1]
    sample_pred = model.predict(sample_scaled)[0]
    print(f"  Sample Applicant Default Probability: {sample_prob:.4f} -> Prediction: {sample_pred} ({'Default' if sample_pred else 'Approved'})")


# ==============================================================================
# SECTION 2: CUSTOMER CHURN RETENTION CLASSIFICATION (RANDOM FOREST)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. Random Forest Classifier is an ensemble learning method combining multiple Decision Trees:
       - Bootstrap Aggregation (Bagging): Each tree is trained on an independent bootstrap sample
         drawn with replacement from the training dataset.
       - Random Subspace Feature Selection: At each node split, only a random subset of features
         (typically sqrt(p)) is evaluated, decorrelating individual decision trees.
       - Final prediction is determined by majority vote among all B base estimators:
           y_hat = mode { h_1(x), h_2(x), ..., h_B(x) }
  2. Impurity Criteria:
       - Gini Impurity: I_G = 1 - sum(p_k^2)
       - Information Entropy: H = - sum(p_k * log2(p_k))
  3. Defensive Feature Engineering:
       avg_monthly_spend = total_charges / tenure_months.replace(0, 1)
       Safeguards against division-by-zero crashes when new customers have 0 months tenure.
--------------------------------------------------------------------------------
"""
def run_section_2_customer_churn():
    print("\n" + "=" * 80)
    print("SECTION 2: CUSTOMER CHURN RETENTION CLASSIFICATION (RANDOM FOREST)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "ML_Practice" / "Customer_Churn" / "customer_churn_data.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 600
        df = pd.DataFrame({
            'customer_id': range(5001, 5001 + n),
            'tenure_months': np.random.randint(0, 72, size=n),
            'monthly_charges': np.random.uniform(20, 120, size=n).round(2),
            'contract_type': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n, p=[0.5, 0.3, 0.2]),
            'tech_support': np.random.choice(['No', 'Yes'], size=n, p=[0.6, 0.4]),
            'payment_method': np.random.choice(['Electronic', 'Mailed Check', 'Bank Transfer'], size=n)
        })
        df['total_charges'] = (df['tenure_months'] * df['monthly_charges']).round(2)
        churn_prob = 0.5 - 0.005 * df['tenure_months'] + 0.002 * df['monthly_charges'] - 0.2 * (df['contract_type'] == 'Two-Year')
        churn_prob = np.clip(churn_prob, 0.05, 0.95)
        df['churn_flag'] = (np.random.uniform(0, 1, size=n) < churn_prob).astype(int)
    else:
        df = pd.read_csv(csv_path)

    print(f"[Phase 1] Ingested Dataset: {df.shape[0]} subscribers, {df.shape[1]} columns")
    print("Class Balance:\n", df['churn_flag'].value_counts(normalize=True).round(3))

    # Phase 2: Defensive Feature Engineering & Categorical Encoding
    X = df.drop(columns=['customer_id', 'churn_flag'])
    y = df['churn_flag']
    X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'].replace(0, 1)
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"[Phase 2] Features Engineered. Transformed Dimension: {X_encoded.shape}")

    # Phase 3: Train-Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Phase 4: Model Training & Feature Importance
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[Phase 4] Random Forest Test Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    # Top Feature Importances
    fi_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 5 Churn Predictors by Gini Importance:\n", fi_df.head(5).to_string(index=False))

    # Phase 5: Persistence
    artifacts_dir = ROOT_DIR / "Day11" / "ML_Practice" / "Customer_Churn"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, artifacts_dir / "churn_rf_model.joblib")
    joblib.dump(scaler, artifacts_dir / "churn_scaler.joblib")
    print("[Phase 5] Saved artifacts to:", artifacts_dir)


# ==============================================================================
# SECTION 3: REAL ESTATE VALUATION & PRICE PREDICTION (LINEAR REGRESSION)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. Multiple Linear Regression models a continuous target variable y (Home Price):
       y_hat = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_p * x_p
  2. Ordinary Least Squares (OLS) closed-form solution:
       beta = (X^T * X)^(-1) * X^T * y
     Minimizes Residual Sum of Squares: RSS = sum( (y_i - y_hat_i)^2 )
  3. Continuous Regression Evaluation Metrics:
       - Mean Absolute Error (MAE): (1/n) * sum |y_i - y_hat_i| (Robust to outliers)
       - Mean Squared Error (MSE): (1/n) * sum (y_i - y_hat_i)^2 (Penalizes large errors)
       - Root Mean Squared Error (RMSE): sqrt(MSE) (Interpreted in target currency units)
       - Coefficient of Determination (R^2): 1 - (SS_res / SS_tot)
         Represents the proportion of variance in y explained by the linear model.
  4. Domain Feature Engineering:
       sqft_per_room = square_feet / num_bedrooms
--------------------------------------------------------------------------------
"""
def run_section_3_real_estate():
    print("\n" + "=" * 80)
    print("SECTION 3: REAL ESTATE VALUATION & PRICE PREDICTION (LINEAR REGRESSION)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "ML_Practice" / "Real_Estate" / "housing_ml_data.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 500
        sqft = np.random.uniform(800, 4500, size=n).round(0)
        beds = np.random.randint(1, 6, size=n)
        age = np.random.randint(0, 40, size=n)
        loc = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n, p=[0.4, 0.4, 0.2])
        garage = np.random.choice(['Yes', 'No'], size=n, p=[0.7, 0.3])
        price = 50000 + 180 * sqft + 15000 * beds - 1200 * age + 40000 * (loc == 'Urban') + 20000 * (garage == 'Yes') + np.random.normal(0, 25000, size=n)
        df = pd.DataFrame({
            'house_id': range(8001, 8001 + n),
            'square_feet': sqft,
            'num_bedrooms': beds,
            'property_age': age,
            'location_type': loc,
            'has_garage': garage,
            'price_usd': price.round(2)
        })
    else:
        df = pd.read_csv(csv_path)

    print(f"[Phase 1] Ingested Housing Records: {df.shape[0]} rows, {df.shape[1]} columns")

    # Phase 2: Feature Engineering
    X = df.drop(columns=['house_id', 'price_usd'])
    y = df['price_usd']
    X['sqft_per_room'] = X['square_feet'] / X['num_bedrooms']
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Phase 3: Train-Test Split (Standard 80/20) & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Phase 4: Model Training & Residual Evaluation
    reg_model = LinearRegression()
    reg_model.fit(X_train_scaled, y_train)
    y_pred = reg_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"[Phase 4] Regression Performance Metrics:")
    print(f"  R-squared (R2):                {r2:.4f}")
    print(f"  Mean Absolute Error (MAE):     ${mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    # Phase 5: Artifact Persistence & Single Property Valuation
    artifacts_dir = ROOT_DIR / "Day11" / "ML_Practice" / "Real_Estate"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_model, artifacts_dir / "housing_model.joblib")
    joblib.dump(scaler, artifacts_dir / "housing_scaler.joblib")
    print("[Phase 5] Saved regression model artifacts to:", artifacts_dir)

    sample_residence = X_encoded.iloc[[0]].copy()
    sample_val = reg_model.predict(scaler.transform(sample_residence))[0]
    print(f"  Sample Residence Actual: ${y.iloc[0]:,.2f} | Predicted Valuation: ${sample_val:,.2f}")


# ==============================================================================
# SECTION 4: CUSTOMER BEHAVIORAL SEGMENTATION (K-MEANS CLUSTERING)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. K-Means is a centroid-based, unsupervised partitioning algorithm that clusters
     unlabeled data points into K non-overlapping clusters:
       Minimizes WCSS (Within-Cluster Sum of Squares / Inertia):
         WCSS = sum_{k=1}^K sum_{x in C_k} || x - mu_k ||^2
  2. The Critical Role of Standardization (StandardScaler):
       K-Means uses Euclidean distance: d(p, q) = sqrt( sum( (p_i - q_i)^2 ) ).
       If feature scales differ (e.g., Annual Income in $10,000s vs Age in 10s),
       unscaled features with larger ranges will artificially dominate cluster assignments.
  3. Silhouette Score Evaluation:
       s(i) = (b(i) - a(i)) / max(a(i), b(i))
         a(i): Mean intra-cluster distance between sample i and all other points in same cluster.
         b(i): Mean nearest-cluster distance between sample i and points in the closest other cluster.
       Range: [-1, +1]. Higher values indicate tight, well-separated cluster induction.
--------------------------------------------------------------------------------
"""
def run_section_4_customer_segmentation():
    print("\n" + "=" * 80)
    print("SECTION 4: CUSTOMER BEHAVIORAL SEGMENTATION (K-MEANS CLUSTERING)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "ML_Practice" / "Customer_Segmentation" / "customer_segmentation_data.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 450
        df = pd.DataFrame({
            'customer_id': range(3001, 3001 + n),
            'age': np.random.randint(18, 70, size=n),
            'annual_income_k': np.random.uniform(15, 130, size=n).round(1),
            'spending_score_1_100': np.random.randint(1, 100, size=n),
            'purchase_frequency_monthly': np.random.randint(1, 15, size=n)
        })
    else:
        df = pd.read_csv(csv_path)

    print(f"[Phase 1] Ingested Unlabeled Records: {df.shape[0]} customers")
    X = df.drop(columns=['customer_id'])

    # Phase 2: Feature Standardization for Euclidean Distance Invariance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Phase 3: K-Means Clustering (K=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clustered = df.copy()
    df_clustered['cluster_id'] = cluster_labels

    # Phase 4: Cluster Profiling & Silhouette Metric
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"[Phase 4] K-Means Clustering Evaluation:")
    print(f"  Model Inertia (WCSS):     {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score:         {sil_score:.4f} (Cohesive separation)")

    profiles = df_clustered.groupby('cluster_id')[['age', 'annual_income_k', 'spending_score_1_100', 'purchase_frequency_monthly']].mean()
    print("\nSegment Centroid Profiles (Cluster Means):\n", profiles.round(2))

    # Phase 5: Persistence
    artifacts_dir = ROOT_DIR / "Day11" / "ML_Practice" / "Customer_Segmentation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, artifacts_dir / "customer_segmentation_model.joblib")
    joblib.dump(scaler, artifacts_dir / "customer_segmentation_scaler.joblib")
    print("[Phase 5] Saved clustering model artifacts to:", artifacts_dir)


# ==============================================================================
# SECTION 5: CRICKET PLAYER ARCHETYPE SEGMENTATION (K-MEANS CLUSTERING)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. Multidimensional Unsupervised Archetype Discovery:
       Sports performance features often form natural clusters:
         - Pure Batsmen: High runs, high batting average, low/zero wickets.
         - Pure Bowlers: Low runs, high wickets, low bowling economy.
         - All-Rounders: Balanced runs, moderate batting average, substantial wickets.
  2. K-Means Partitions player vectors into K=3 distinct strategic roles.
  3. Feature selection: ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy'].
--------------------------------------------------------------------------------
"""
def run_section_5_cricket_clustering():
    print("\n" + "=" * 80)
    print("SECTION 5: CRICKET PLAYER ARCHETYPE SEGMENTATION (K-MEANS CLUSTERING)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "ML_Practice" / "Cricket_Player_Statistics" / "cricket_player_stats.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'player_name': [f"Player_{i}" for i in range(1, n + 1)],
            'runs_scored': np.random.randint(100, 12000, size=n),
            'batting_average': np.random.uniform(10.0, 55.0, size=n).round(2),
            'wickets_taken': np.random.randint(0, 450, size=n),
            'bowling_economy': np.random.uniform(3.5, 9.5, size=n).round(2)
        })
    else:
        df = pd.read_csv(csv_path)

    features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    X = df[features]
    print(f"[Phase 1] Ingested Player Stats: {df.shape[0]} athletes across {len(features)} attributes")

    # Phase 2: Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Phase 3: KMeans Induction
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_players = df.copy()
    df_players['cluster_role_id'] = labels

    # Phase 4: Silhouette Evaluation & Archetype Mapping
    sil = silhouette_score(X_scaled, labels)
    print(f"[Phase 4] Silhouette Score: {sil:.4f}")
    role_profiles = df_players.groupby('cluster_role_id')[features].mean()
    print("\nArchetype Centroids (Batsmen / Bowler / All-Rounder):\n", role_profiles.round(2))

    # Phase 5: Persistence
    artifacts_dir = ROOT_DIR / "Day11" / "ML_Practice" / "Cricket_Player_Statistics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, artifacts_dir / "cricket_kmeans_model.joblib")
    joblib.dump(scaler, artifacts_dir / "cricket_scaler.joblib")
    print("[Phase 5] Saved cricket role clustering artifacts to:", artifacts_dir)


# ==============================================================================
# SECTION 6: HEALTHCARE HYPERTENSION RISK CLASSIFICATION (LOGISTIC REGRESSION)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & MATHEMATICAL BACKGROUND:
  1. Clinical Risk Prediction:
       Predicting high-risk hypertension diagnosis based on physiological and lifestyle markers:
       Target: High_Risk_Hypertension in {0, 1}
       Predictors: Body Mass Index (BMI), Daily Physical Activity (Daily_Steps)
  2. Supervised Group Segmentation:
       Comparing group mean vectors E[X | y=0] vs E[X | y=1] reveals clinical separations:
       Hypertension patients exhibit higher mean BMI and significantly reduced daily step count.
  3. Clinical Utility: Provides actionable cutoff probability scores for preventative screening.
--------------------------------------------------------------------------------
"""
def run_section_6_healthcare_classification():
    print("\n" + "=" * 80)
    print("SECTION 6: HEALTHCARE HYPERTENSION RISK CLASSIFICATION (LOGISTIC REGRESSION)")
    print("=" * 80)

    csv_path = ROOT_DIR / "Day11" / "Practice4" / "patient_classification_data.csv"
    if not csv_path.exists():
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic fallback...")
        np.random.seed(42)
        n = 400
        bmi = np.random.uniform(18.5, 42.0, size=n).round(1)
        steps = np.random.randint(1500, 15000, size=n)
        risk_logit = -5.0 + 0.25 * bmi - 0.0004 * steps
        prob = 1 / (1 + np.exp(-risk_logit))
        hypertension = (prob > np.random.uniform(0, 1, size=n)).astype(int)
        df = pd.DataFrame({
            'Patient_ID': range(7001, 7001 + n),
            'BMI': bmi,
            'Daily_Steps': steps,
            'High_Risk_Hypertension': hypertension
        })
    else:
        df = pd.read_csv(csv_path)

    print(f"[Phase 1] Loaded Clinical Patient Records: {df.shape[0]} patients")
    print("Class Distribution:\n", df['High_Risk_Hypertension'].value_counts())

    # Phase 2: Supervised Segmentation
    means = df.groupby('High_Risk_Hypertension')[['BMI', 'Daily_Steps']].mean()
    print("\n[Phase 2] Mean Biomarkers by Hypertension Status:\n", means.round(2))

    # Phase 3: Train-Test Split & Classification
    X = df[['BMI', 'Daily_Steps']]
    y = df['High_Risk_Hypertension']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[Phase 4] Clinical Model Test Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print("Clinical Recommendation: Patients with BMI > 28 and Steps < 5,000 flagged for immediate preventative care.")


# ==============================================================================
# SECTION 7: EXAM CRASH COUNTERMEASURES & DEFENSIVE CODING LAB (8 FATAL TRAPS)
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & EXAM PITFALLS:
  This section demonstrates the 8 most frequent coding mistakes that cause Python
  scripts to crash or lose marks during CDAC practical ML exams, along with their
  bulletproof single-line countermeasures:
    - Trap 1: File not found due to Terminal Working Directory (CWD) mismatch.
    - Trap 2: ValueError: Expected 2D array, got 1D array instead.
    - Trap 3: Data leakage by calling fit_transform() on the test split.
    - Trap 4: Division by zero generating infinite (inf) or NaN values.
    - Trap 5: Dimensionality mismatch due to unseen categories in test one-hot encoding.
    - Trap 6: Imbalanced dataset split without class stratification.
    - Trap 7: Silent execution failure from unhandled NaNs before scikit-learn estimators.
    - Trap 8: Passing raw string columns into scikit-learn estimators.
--------------------------------------------------------------------------------
"""
def run_section_7_exam_countermeasures():
    print("\n" + "=" * 80)
    print("SECTION 7: EXAM CRASH COUNTERMEASURES & DEFENSIVE CODING LAB (8 FATAL TRAPS)")
    print("=" * 80)

    # TRAP 1: File Path CWD Trap
    print("\n[TRAP 1] Path Resolution: pd.read_csv('file.csv') fails when running from root")
    safe_path = Path(__file__).resolve().parent / "Day11" / "Practice1" / "coffee_shop_data.csv"
    print(f"  Crash-Proof Fix: safe_path = Path(__file__).resolve().parent / 'relative_path'")
    print(f"  Resolved Path Exists: {safe_path.exists()}")

    # TRAP 2: 1D vs 2D Array Shape in Scikit-Learn
    print("\n[TRAP 2] Expected 2D Array vs 1D Array in Model Fitting")
    sample_df = pd.DataFrame({'experience_years': [1, 2, 3, 4], 'salary': [30, 42, 51, 65]})
    try:
        # BUG: single bracket returns 1D pandas Series
        LinearRegression().fit(sample_df['experience_years'], sample_df['salary'])
    except ValueError as e:
        print(f"  [BUGGY] X = df['col'] -> CRASH: {type(e).__name__}: {str(e)[:45]}...")
    # FIX: double brackets return 2D DataFrame
    lr_fix = LinearRegression().fit(sample_df[['experience_years']], sample_df['salary'])
    print(f"  [FIX]   X = df[['col']] (Double Brackets) -> SUCCESS! Fitted Slope: {lr_fix.coef_[0]:.2f}")

    # TRAP 3: Data Leakage in Feature Scaling
    print("\n[TRAP 3] Data Leakage in Feature Scaling")
    X_train_raw = np.array([[10.0], [20.0], [30.0], [40.0]])
    X_test_raw = np.array([[50.0], [60.0]])
    scaler_leak = StandardScaler()
    scaler_leak.fit_transform(X_train_raw)
    # BUG: fit_transform re-calculates mu/sigma on test set, corrupting generalization evaluation
    print("  [BUGGY] scaler.fit_transform(X_test) leaks test distribution statistics!")
    # FIX: Only transform test set using training set distribution parameters
    scaler_safe = StandardScaler()
    scaler_safe.fit_transform(X_train_raw)
    X_test_safe = scaler_safe.transform(X_test_raw)
    print(f"  [FIX]   scaler.transform(X_test) -> SUCCESS! Test scaled with train mean: {scaler_safe.mean_[0]:.1f}")

    # TRAP 4: Division by Zero & Infinite (inf) Values
    print("\n[TRAP 4] Division by Zero in Feature Ratios")
    tenure = pd.Series([12, 0, 5, 24])
    charges = pd.Series([1200, 150, 400, 2400])
    # BUG: 0-tenure produces inf
    ratio_bug = charges / tenure
    print(f"  [BUGGY] Has 'inf': {np.isinf(ratio_bug).any()}")
    # FIX: Replace 0 with 1 or clip to positive minimum
    ratio_fix = charges / tenure.replace(0, 1)
    print(f"  [FIX]   charges / tenure.replace(0, 1) -> SUCCESS! Has 'inf': {np.isinf(ratio_fix).any()}")

    # TRAP 5: Categorical Encoding Alignment Across Splits
    print("\n[TRAP 5] Categorical Encoding Alignment")
    print("  [FIX]   Apply pd.get_dummies(df, drop_first=True) on entire dataframe prior to train_test_split")
    print("          or use sklearn OneHotEncoder(handle_unknown='ignore') inside a Pipeline.")

    # TRAP 6: Imbalanced Class Stratification
    print("\n[TRAP 6] Preserving Minority Classes with Stratification")
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)
    X_dummy = pd.DataFrame(np.random.randn(100, 2))
    X_tr, X_te, y_tr, y_te = train_test_split(X_dummy, y_imbalanced, test_size=0.2, stratify=y_imbalanced, random_state=42)
    print(f"  [FIX]   stratify=y ensures minority class ratio: Train={y_tr.mean():.1%}, Test={y_te.mean():.1%}")

    # TRAP 7: Unhandled Missing Values
    print("\n[TRAP 7] Missing Value Imputation Before Estimators")
    s_na = pd.Series([10.0, np.nan, 30.0, 40.0])
    s_clean = s_na.fillna(s_na.median())
    print(f"  [FIX]   df.fillna(df.median()) eliminated NaN values: {s_clean.isna().sum()} nulls remaining")

    # TRAP 8: Unencoded String Passing to Model
    print("\n[TRAP 8] String Columns Passing to Estimator")
    print("  [FIX]   Always verify X.select_dtypes(include='object').empty is True prior to model.fit()")


# ==============================================================================
# SECTION 8 (LAST): SYLLABUS BENCHMARK MODEL - TELCO CUSTOMER CHURN PIPELINE
# ==============================================================================
"""
--------------------------------------------------------------------------------
THEORY & SYLLABUS BACKGROUND (ACTS PGCP-AI BENCHMARK):
  1. Syllabus Reference:
       CDAC ACTS PGCP-AI Practical Machine Learning Teaching Guidelines (120 Hours).
       Session 8-9 (Logistic Regression & Classification), Session 15-16 (Ensemble Trees),
       and Session 23 (ROC/AUC, Confusion Matrix, Precision, Recall, F1).
  2. Dataset:
       IBM Telco Customer Churn (7,043 customer accounts, 21 telemetry columns).
       Source location in repository:
         Day1/O2_Telco_Data_Ingestion/data/raw/Telco-Customer-Churn.csv
  3. Industrial End-to-End Workflow:
       - Step 1: Ingestion & Column Type Sanitization.
                 CRITICAL TRAP: `TotalCharges` is stored as an object column containing
                 whitespace strings (' ') for brand new customers (tenure = 0).
                 Fix: pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(df['TotalCharges'].median())
       - Step 2: Binary & Multi-class One-Hot Encoding:
                 Drop unique identifiers (`customerID`).
                 Convert target `Churn` to integer: {'Yes': 1, 'No': 0}.
                 One-hot encode contract, payment method, internet service (`drop_first=True`).
       - Step 3: Stratified 80/20 Train-Test Splitting (`stratify=y`).
       - Step 4: Standard scaling continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`)
                 fitted strictly on training set.
       - Step 5: Model A - Logistic Regression & Odds-Ratio Analysis:
                 Odds Ratio = exp(beta).
                 Features with Odds Ratio > 1 increase churn risk.
                 Features with Odds Ratio < 1 decrease churn risk (retention drivers).
       - Step 6: Model B - Random Forest Classifier:
                 Gini feature importances to identify top non-linear churn drivers.
       - Step 7: Dual Comparative Evaluation:
                 Accuracy, Precision, Recall, F1, ROC-AUC Score, and Confusion Matrix.
       - Step 8: Persistence & Sample Single Customer Inference Pipeline.
--------------------------------------------------------------------------------
"""
def run_section_8_telco_churn_benchmark():
    print("\n" + "=" * 80)
    print("SECTION 8 (LAST): SYLLABUS BENCHMARK MODEL - TELCO CUSTOMER CHURN PIPELINE")
    print("=" * 80)

    # Step 1: Locate Dataset with Fallback
    telco_path = ROOT_DIR / "Day1" / "O2_Telco_Data_Ingestion" / "data" / "raw" / "Telco-Customer-Churn.csv"
    if not telco_path.exists():
        # Secondary fallback location
        telco_path = ROOT_DIR / "Day3" / "O3_Data_Preprocessing_Cleaning_Eda" / "data" / "processed" / "clean_telco_telemetry.csv"

    if telco_path.exists():
        print(f"[Step 1] Loading Telco Customer Churn from: {telco_path}")
        df = pd.read_csv(telco_path)
    else:
        print("[!] Raw Telco dataset not found. Generating official 1,000-sample synthetic benchmark...")
        np.random.seed(2026)
        n = 1000
        tenure = np.random.randint(1, 72, size=n)
        monthly = np.random.uniform(20.0, 120.0, size=n).round(2)
        total = (tenure * monthly + np.random.normal(0, 50, size=n)).round(2)
        total = np.clip(total, 20.0, None)
        contract = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n, p=[0.5, 0.3, 0.2])
        tech_support = np.random.choice(["No", "Yes"], size=n, p=[0.6, 0.4])
        internet = np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.4, 0.2])
        logit = 1.2 - 0.05 * tenure + 0.03 * monthly - 1.5 * (contract == "Two-Year") - 0.8 * (tech_support == "Yes")
        prob_churn = 1 / (1 + np.exp(-logit))
        churn = np.where(prob_churn > np.random.uniform(0, 1, size=n), "Yes", "No")

        df = pd.DataFrame({
            "customerID": [f"ID_{i:04d}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "TechSupport": tech_support,
            "InternetService": internet,
            "Churn": churn
        })

    print(f"  Raw Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")

    # Step 2: Data Cleaning & Coercion of TotalCharges whitespace trap
    df = df.copy()
    if 'TotalCharges' in df.columns:
        # Coerce strings with whitespace ' ' to NaN and impute median
        total_num = pd.to_numeric(df['TotalCharges'], errors='coerce')
        nan_count = total_num.isna().sum()
        median_val = total_num.median()
        df['TotalCharges'] = total_num.fillna(median_val)
        if nan_count > 0:
            print(f"  [Cleaned] Repaired {nan_count} whitespace strings in TotalCharges using median (${median_val:.2f})")

    # Remove non-predictive ID column
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Standardize target Churn column to 0/1 integer
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        target_col = 'Churn'
    elif 'churn' in df.columns:
        if df['churn'].dtype == 'object':
            df['churn'] = df['churn'].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        target_col = 'churn'
    else:
        raise KeyError("Could not identify target 'Churn' column in Telco dataset.")

    print(f"  Target Distribution ({target_col}):\n", df[target_col].value_counts(normalize=True).map("{:.2%}".format))

    # Step 3: Categorical One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    print(f"[Step 3] One-Hot Encoded Feature Matrix Dimensions: {X.shape[0]} samples x {X.shape[1]} features")

    # Step 4: Stratified Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Step 4] Split Completed: {len(X_train)} Train samples | {len(X_test)} Test samples")

    # Step 5: Feature Scaling for Continuous Numerical Features
    # Identify continuous numerical columns present in dataset
    continuous_candidates = ['tenure', 'tenure_months', 'MonthlyCharges', 'monthly_charges', 'TotalCharges', 'total_charges']
    num_cols = [c for c in continuous_candidates if c in X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if num_cols:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        print(f"[Step 5] Standard Scaled Numerical Columns: {num_cols}")

    # ==========================================================================
    # MODEL A: LOGISTIC REGRESSION WITH ODDS RATIO EXTRACTION
    # ==========================================================================
    print("\n--- MODEL A: LOGISTIC REGRESSION ---")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    y_pred_lr = logreg.predict(X_test_scaled)
    y_proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    auc_lr = roc_auc_score(y_test, y_proba_lr)

    print(f"  Logistic Regression Test Accuracy:  {acc_lr * 100:.2f}%")
    print(f"  Logistic Regression Precision:      {prec_lr:.4f}")
    print(f"  Logistic Regression Recall:         {rec_lr:.4f}")
    print(f"  Logistic Regression F1-Score:       {f1_lr:.4f}")
    print(f"  Logistic Regression ROC-AUC Score:  {auc_lr:.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    # Odds Ratios Calculation: exp(beta)
    odds_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Beta_Coefficient': logreg.coef_[0],
        'Odds_Ratio': np.exp(logreg.coef_[0])
    }).sort_values(by='Odds_Ratio', ascending=False)
    print("\n  Top 3 Churn Risk Multipliers (Odds Ratio > 1):\n", odds_df.head(3).to_string(index=False))
    print("\n  Top 3 Retention Protectors (Odds Ratio < 1):\n", odds_df.tail(3).to_string(index=False))

    # ==========================================================================
    # MODEL B: RANDOM FOREST CLASSIFIER WITH GINI IMPORTANCES
    # ==========================================================================
    print("\n--- MODEL B: RANDOM FOREST CLASSIFIER ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    print(f"  Random Forest Test Accuracy:        {acc_rf * 100:.2f}%")
    print(f"  Random Forest Precision:            {prec_rf:.4f}")
    print(f"  Random Forest Recall:               {rec_rf:.4f}")
    print(f"  Random Forest F1-Score:             {f1_rf:.4f}")
    print(f"  Random Forest ROC-AUC Score:        {auc_rf:.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # Feature Importances
    fi_telco = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\n  Top 5 Features by Random Forest Gini Importance:\n", fi_telco.head(5).to_string(index=False))

    # ==========================================================================
    # STEP 8: MODEL COMPARISON & PERSISTENCE
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TELCO BENCHMARK SUMMARY COMPARISON:")
    print("-" * 60)
    summary_table = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": f"{acc_lr:.4f}", "Precision": f"{prec_lr:.4f}", "Recall": f"{rec_lr:.4f}", "F1": f"{f1_lr:.4f}", "ROC-AUC": f"{auc_lr:.4f}"},
        {"Model": "Random Forest", "Accuracy": f"{acc_rf:.4f}", "Precision": f"{prec_rf:.4f}", "Recall": f"{rec_rf:.4f}", "F1": f"{f1_rf:.4f}", "ROC-AUC": f"{auc_rf:.4f}"}
    ])
    print(summary_table.to_string(index=False))

    # Save artifacts
    save_dir = ROOT_DIR / "Day8"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, save_dir / "telco_logistic_regression.joblib")
    joblib.dump(rf, save_dir / "telco_random_forest.joblib")
    if num_cols:
        joblib.dump(scaler, save_dir / "telco_scaler.joblib")
    print(f"\n[Saved] Telco model artifacts persisted to: {save_dir}")

    # Single customer inference test
    sample_sub = X_test_scaled.iloc[[0]].copy()
    sub_pred_lr = logreg.predict(sample_sub)[0]
    sub_prob_lr = logreg.predict_proba(sample_sub)[0, 1]
    print(f"  Inference Test on Subscriber 0: Churn Probability = {sub_prob_lr:.2%} -> Class {sub_pred_lr} ({'Churn Risk' if sub_pred_lr else 'Retained'})")


# ==============================================================================
# MAIN ORCHESTRATOR & CLI RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Day 11 Consolidated ML Master Pipeline & Syllabus Telco Benchmark")
    parser.add_argument(
        "--section",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Execute only a specific section (1 to 8). If omitted, all sections run sequentially."
    )
    args = parser.parse_args()

    section_dispatch = {
        1: ("Section 1: Bank Loan Risk", run_section_1_bank_loan),
        2: ("Section 2: Customer Churn (Random Forest)", run_section_2_customer_churn),
        3: ("Section 3: Real Estate Valuation", run_section_3_real_estate),
        4: ("Section 4: Customer Segmentation (K-Means)", run_section_4_customer_segmentation),
        5: ("Section 5: Cricket Player Statistics (K-Means)", run_section_5_cricket_clustering),
        6: ("Section 6: Healthcare Hypertension Risk", run_section_6_healthcare_classification),
        7: ("Section 7: 8 Exam Crash Countermeasures", run_section_7_exam_countermeasures),
        8: ("Section 8: Telco Churn Syllabus Benchmark", run_section_8_telco_churn_benchmark)
    }

    print("\n" + "#" * 80)
    print("CDAC PGCP-AI PRACTICAL MACHINE LEARNING - DAY 11 MASTER CONSOLIDATED SUITE")
    print("#" * 80)

    if args.section:
        title, func = section_dispatch[args.section]
        print(f"\nExecuting Selected Section: {title}")
        func()
    else:
        print("\nExecuting Complete End-to-End Suite (Sections 1 through 8)...")
        for sec_num, (title, func) in section_dispatch.items():
            func()

    print("\n" + "=" * 80)
    print("ALL REQUESTED MACHINE LEARNING PIPELINES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
