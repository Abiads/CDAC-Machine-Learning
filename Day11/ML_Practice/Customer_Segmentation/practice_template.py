"""
================================================================================
PRACTICE TEMPLATE: Day 11 - ML Practice 4 (Customer Segmentation K-Means)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: customer_segmentation_data.csv
FEATURES: ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
MODEL:   KMeans(n_clusters=3, random_state=42)
METRIC:  silhouette_score
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import joblib

csv_path = Path(__file__).resolve().parent / "customer_segmentation_data.csv"


def phase_1_discovery_and_feature_selection():
    """
    [Phase 1] Discovery & Feature Selection
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    3. X = df[features]
    4. Return df, X!
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    # X = df[features]
    # return df, X
    pass


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. scaler = StandardScaler()
    2. X_scaled = scaler.fit_transform(X)
    3. Return X_scaled, scaler!
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # return X_scaled, scaler
    pass


def phase_3_kmeans_clustering(df, X_scaled, n_clusters=3):
    """
    [Phase 3] K-Means Clustering & Segment Assignment
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    2. df['segment_id'] = kmeans.labels_
    3. Return kmeans, df!
    """
    print("\n[PHASE 3] K-Means Clustering & Segment Assignment")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    # df['segment_id'] = kmeans.labels_
    # return kmeans, df
    pass


def phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df):
    """
    [Phase 4] Silhouette Score & Persona Profiling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. score = silhouette_score(X_scaled, kmeans.labels_)
    2. features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    3. centroids = df.groupby('segment_id')[features].mean()
    """
    print("\n[PHASE 4] Cluster Evaluation & Persona Profiling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # score = silhouette_score(X_scaled, kmeans.labels_)
    # print(f"Silhouette Score: {score:.4f}")
    # features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    # centroids = df.groupby('segment_id')[features].mean()
    # print("Customer Segment Centroids:\n", centroids.round(2))
    pass


def run_segmentation_pipeline():
    print("=" * 65)
    print("STARTING CUSTOMER SEGMENTATION LIFECYCLE (K-Means)")
    print("=" * 65)
    
    # df, X = phase_1_discovery_and_feature_selection()
    # X_scaled, scaler = phase_2_feature_scaling(X)
    # kmeans, df = phase_3_kmeans_clustering(df, X_scaled)
    # phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df)


if __name__ == "__main__":
    run_segmentation_pipeline()
