"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - ML Practice 4
DOMAIN: Retail Persona Segmentation (K-Means Clustering)
================================================================================
PROBLEM STATEMENT (PS):
Segment 25 retail shoppers into 3 distinct marketing personas based on age, income, 
spending score, and purchase frequency.
Apply StandardScaler, fit KMeans(n_clusters=3), validate with silhouette_score, 
and profile centroid means.

Fill in all the `___` blanks below to complete the implementation.
================================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "customer_segmentation_data.csv"


def phase_1_discovery_and_feature_selection():
    """
    [Phase 1] Discovery & Segmentation Feature Selection
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    
    # 1. Select the 4 continuous shopping features
    features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    X = df[___]
    
    return df, ___                      # 2. Return df, X


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    scaler = StandardScaler()
    # 3. Fit and transform scaler
    X_scaled = scaler.fit_transform(___)
    
    return X_scaled, ___                # 4. Return X_scaled, scaler


def phase_3_kmeans_clustering(df, X_scaled, n_clusters=3):
    """
    [Phase 3] K-Means Clustering & Segment Assignment
    """
    print("\n[PHASE 3] K-Means Clustering & Segment Assignment")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 5. Fit KMeans
    kmeans = KMeans(n_clusters=___, random_state=42).fit(___)
    
    # 6. Assign cluster labels
    df['segment_id'] = kmeans.___
    
    return kmeans, ___                  # 7. Return kmeans, df


def phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df):
    """
    [Phase 4] Silhouette Score & Persona Profiling
    """
    print("\n[PHASE 4] Cluster Evaluation & Persona Profiling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 8. Compute silhouette score
    score = silhouette_score(___, kmeans.labels_)
    print(f"Silhouette Score: {score:.4f}")
    
    features = ['age', 'annual_income_k', 'spending_score', 'purchase_frequency']
    # 9. Compute centroid averages
    centroids = df.groupby('segment_id')[features].___()
    print("Customer Segment Centroids:\n", centroids.round(2))


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
