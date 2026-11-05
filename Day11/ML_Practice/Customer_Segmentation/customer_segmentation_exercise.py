"""
Title: Unsupervised Customer Segmentation Pipeline with K-Means
Description: Simple, built-in function driven Python script for customer segmentation.
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

csv_path = Path(__file__).resolve().parent / "customer_segmentation_data.csv"


def phase_1_discovery_and_setup():
    """Phase 1: Ingest behavioral data and select numeric clustering features."""
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['customer_id'])
    print(f"Dataset Shape: {df.shape}")
    print(X.head(3))
    return X, df


def phase_2_feature_scaling(X):
    """Phase 2: Scale features using StandardScaler for distance metrics."""
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled, df_original):
    """Phase 3: Train KMeans with 3 clusters and assign segment labels."""
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    model = KMeans(n_clusters=3, random_state=42, n_init='auto')
    labels = model.fit_predict(X_scaled)
    df_clustered = df_original.copy()
    df_clustered['cluster_id'] = labels
    return model, labels, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered):
    """Phase 4: Calculate silhouette score, inspect segment centroids, and plot customer clusters."""
    print("\n[PHASE 4] Cluster Evaluation & Profiling")
    print("-" * 65)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (Cluster Cohesion): {score:.4f}")
    profile = df_clustered.groupby('cluster_id')[['age', 'annual_income_k', 'spending_score_1_100', 'purchase_frequency_monthly']].mean()
    print("\nCustomer Segment Profiles:\n", profile.round(2))

    base_dir = Path(__file__).resolve().parent
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_clustered['annual_income_k'], df_clustered['spending_score_1_100'],
                          c=df_clustered['cluster_id'], cmap='plasma', s=120, edgecolor='black', alpha=0.85)
    plt.title("Customer Behavioral Segments (K-Means)")
    plt.xlabel("Annual Income ($k)")
    plt.ylabel("Spending Score (1-100)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Segment Cluster ID (0, 1, 2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / "customer_segments_clusters.png", dpi=150)
    plt.close()
    print("Saved visualization: 'customer_segments_clusters.png'")


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """Phase 5: Serialize KMeans artifacts and test segment prediction on new customer."""
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    base_dir = Path(__file__).resolve().parent
    joblib.dump(model, base_dir / "customer_segmentation_model.joblib")
    joblib.dump(scaler, base_dir / "customer_segmentation_scaler.joblib")
    print("Artifacts saved: customer_segmentation_model.joblib, customer_segmentation_scaler.joblib")

    # Sample customer: Age 25, $20k income, 90 spending score, 8 monthly visits
    sample_scaled = scaler.transform([[25, 20, 90, 8]])
    cluster_pred = model.predict(sample_scaled)
    print(f"Sample Customer Assigned to Segment ID: {cluster_pred[0]}")


def run_clustering_pipeline():
    """Facade function coordinating the entire unsupervised segmentation pipeline."""
    print("=" * 65)
    print("STARTING UNSUPERVISED CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)

    X, df_original = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, df_original)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))

    print("\n" + "=" * 65)
    print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_clustering_pipeline()
