"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - ML Practice 2
DOMAIN: Sports Analytics & Unsupervised Role Clustering (K-Means)
================================================================================
PROBLEM STATEMENT (PS):
Segment 20 international cricket players into 3 playing roles (Batsmen, Bowlers, 
All-Rounders) using unsupervised K-Means clustering.
Scale features using StandardScaler, fit KMeans(n_clusters=3), compute silhouette_score, 
and profile cluster centroids.

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

csv_path = Path(__file__).resolve().parent / "cricket_player_stats.csv"


def phase_1_discovery_and_feature_selection():
    """
    [Phase 1] Discovery & Feature Selection
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    
    # 1. Select the 4 performance features
    features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    X = df[___]
    
    return df, ___                      # 2. Return df, X


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 3. Fit and transform StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(___)
    
    return X_scaled, ___                # 4. Return X_scaled, scaler


def phase_3_kmeans_clustering(df, X_scaled, n_clusters=3):
    """
    [Phase 3] K-Means Clustering & Role Assignment
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 5. Fit KMeans with 3 clusters
    kmeans = KMeans(n_clusters=___, random_state=42).fit(___)
    
    # 6. Assign labels back to dataframe
    df['cluster_id'] = kmeans.___
    
    return kmeans, ___                  # 7. Return kmeans, df


def phase_4_cluster_evaluation_and_profiling(X_scaled, kmeans, df):
    """
    [Phase 4] Silhouette Score & Role Profiling
    """
    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 8. Compute silhouette score
    score = silhouette_score(___, kmeans.labels_)
    print(f"Silhouette Score: {score:.4f}")
    
    features = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    # 9. Compute centroid mean per cluster
    centroids = df.groupby('cluster_id')[features].___()
    print("Cluster Centroid Profiles:\n", centroids.round(2))


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
