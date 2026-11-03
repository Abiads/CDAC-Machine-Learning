"""
================================================================================
PRACTICE TEMPLATE: Day 11 - ML Practice 2 (Cricket Player Role Clustering)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: cricket_player_stats.csv
FEATURES: ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
MODEL: KMeans(n_clusters=3, random_state=42)
METRIC: silhouette_score
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import joblib

csv_path = Path(__file__).resolve().parent / "cricket_player_stats.csv"


def phase_1_discovery_and_feature_selection():
    """
    [Phase 1] Discovery & Feature Selection
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. df = pd.read_csv(csv_path)
    2. features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    3. X = df[features]
    4. Return df, X!
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    # X = df[features]
    # return df, X
    pass


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    TRAP: StandardScaler is mandatory! Without it, runs_scored swamped wickets_taken.
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
    [Phase 3] K-Means Clustering
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    2. df['cluster_id'] = kmeans.labels_
    3. Return kmeans, df!
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    # df['cluster_id'] = kmeans.labels_
    # return kmeans, df
    pass


def phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df):
    """
    [Phase 4] Evaluation with Silhouette Score & Centroid Profiling
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. score = silhouette_score(X_scaled, kmeans.labels_)
    2. Centroid means: df.groupby('cluster_id')[features].mean()
    """
    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # score = silhouette_score(X_scaled, kmeans.labels_)
    # print(f"Silhouette Score: {score:.4f}")
    # features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    # centroids = df.groupby('cluster_id')[features].mean()
    # print("Cluster Centroids (Profiles):\n", centroids.round(2))
    pass


def run_clustering_pipeline():
    print("=" * 65)
    print("STARTING CRICKET PLAYER ROLE CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)
    
    # df, X = phase_1_discovery_and_feature_selection()
    # X_scaled, scaler = phase_2_feature_scaling(X)
    # kmeans, df = phase_3_kmeans_clustering(df, X_scaled)
    # phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df)


if __name__ == "__main__":
    run_clustering_pipeline()
