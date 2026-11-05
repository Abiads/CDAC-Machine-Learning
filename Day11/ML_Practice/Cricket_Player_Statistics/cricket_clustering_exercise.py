"""
Title: Unsupervised Cricket Player Role Segmentation 
Description: Simple, built-in function driven Python script for KMeans cricket clustering.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

csv_path = Path(__file__).resolve().parent / "cricket_player_stats.csv"


def phase_1_discovery_and_setup():
    """Phase 1: Ingest player stats and select numeric performance features."""
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    feature_cols = ['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']
    X = df[feature_cols]
    print(f"Dataset Shape: {df.shape}")
    print(df[['player_name'] + feature_cols].head(3))
    return X, df


def phase_2_feature_scaling(X):
    """Phase 2: Scale features with StandardScaler to equalize distance weightings."""
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled, df_original):
    """Phase 3: Train KMeans with 3 clusters (Batsmen, Bowlers, All-Rounders)."""
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    model = KMeans(n_clusters=3, random_state=42, n_init='auto')
    labels = model.fit_predict(X_scaled)
    df_clustered = df_original.copy()
    df_clustered['cluster_id'] = labels
    return model, labels, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered):
    """Phase 4: Evaluate silhouette score and profile cluster centroids with Visualization."""
    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (Cluster Quality): {score:.4f}")
    profile = df_clustered.groupby('cluster_id')[['runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy']].mean()
    print("\nCluster Centroid Profiles (Roles):\n", profile.round(2))

    base_dir = Path(__file__).resolve().parent
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_clustered['runs_scored'], df_clustered['wickets_taken'], 
                          c=df_clustered['cluster_id'], cmap='viridis', s=120, edgecolor='black', alpha=0.85)
    plt.title("Cricket Player Archetypes (K-Means Clustering)")
    plt.xlabel("Career Runs Scored")
    plt.ylabel("Career Wickets Taken")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Assigned Role Cluster (0, 1, 2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "cricket_player_clusters.png", dpi=150)
    plt.close()
    print("Saved visualization: 'cricket_player_clusters.png'")


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """Phase 5: Serialize KMeans pipeline artifacts and test sample player prediction."""
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "cricket_kmeans_model.joblib")
    joblib.dump(scaler, base_dir / "cricket_scaler.joblib")
    print("Artifacts saved: cricket_kmeans_model.joblib, cricket_scaler.joblib")

    # Sample player inference: 10,000 runs, 50.0 avg, 2 wickets, 7.5 economy
    sample_scaled = scaler.transform([[10000, 50.0, 2, 7.5]])
    cluster_pred = model.predict(sample_scaled)
    print(f"Sample Player Classified into Cluster ID: {cluster_pred[0]}")


def run_cricket_clustering_pipeline():
    """Facade function coordinating the cricket player segmentation pipeline."""
    print("=" * 65)
    print("STARTING CRICKET PLAYER ROLE CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)

    X, df_original = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, df_original)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))

    print("\n" + "=" * 65)
    print("CRICKET CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_cricket_clustering_pipeline()
