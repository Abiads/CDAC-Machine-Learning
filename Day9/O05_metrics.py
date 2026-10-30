'''
Clustering metrics for evaluating clustering models
1. Inertia metric for evaluating clustering models (also known as elbow method )
    a) works on KMeans and other centroid-based clustering models
    b) lower inertia indicates better clustering
    c) is it intra or inter cluster distance based (intra-cluster distance)
    d) sensitive to scale of features
    pitfalls
        - sensitive to outliers
        - may not work well with non-globular clusters
2. Silhouette score for evaluating clustering models
    a) works on any clustering model
    b) ranges from -1 to 1, higher is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    d) sensitive to the number of clusters
    pitfalls
        - may not work well with clusters of varying density or size
3. Davies-Bouldin index for evaluating clustering models
    a) works on any clustering model
    b) lower is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    pitfalls
        - may not work well with clusters of varying density or size

4. Calinski-Harabasz index for evaluating clustering models
    a) works on any clustering model
    b) higher is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    pitfalls
        - may not work well with clusters of varying density or size
5. Adjusted Rand index for evaluating clustering models
    a) works on any clustering model
    b) ranges from -1 to 1, higher is better
    c) considers both true and predicted cluster assignments
    pitfalls
        - may not work well with clusters of varying size
6. Mutual Information for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both true and predicted cluster assignments
    pitfalls
        - may not work well with clusters of varying size
7. V-measure for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both homogeneity and completeness
    pitfalls
        - may not work well with clusters of varying size
8. Homogeneity score for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) measures if each cluster contains only members of a single class
    pitfalls
        - may not work well with clusters of varying size
9. Completeness score for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) measures if all members of a given class are assigned to the same cluster
    pitfalls
        - may not work well with clusters of varying size
10. Fowlkes-Mallows index for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both precision and recall of cluster assignments
    pitfalls
        - may not work well with clusters of varying size
11. Adjusted Mutual Information for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both true and predicted cluster assignments, adjusted for chance
    pitfalls
        - may not work well with clusters of varying size


'''

'''
Davies-Bouldin Index (DBI)
    a) Popular internal evaluation metric used to validate clustering algorithms, including hierarchical clustering.
    b) It measures how well-separated and compact the resulting clusters are.
    c) Helps determine the optimal cut-off point on a dendrogram. [
    meaning it tells you the ideal number of clusters K by finding the balance where clusters are tightly grouped internally and widely separated from one another.
    ]

'''

'''
How DBI Works:
    1. Intra-cluster distance (si)
        - Measures the average distance between each point in a cluster and the cluster centroid.
    2. Inter-cluster distance (d(ci, cj))
        - Measures the distance between the centroids of two different clusters ci and cj.
    3. Similarity ratio (Rij)
        - Measures the ratio of the sum of intra-cluster distances to the inter-cluster distance between clusters ci and cj.
        - The Davies-Bouldin Index is then computed as the average of the maximum similarity ratios for each cluster.
        Rij = (si + sj) / d(ci, cj)
    4. Cluster Similarity (Ri)
        - For a given cluster i, we find the maximum similarity it shares with any other cluster j.
            Ri = max(Rij) for all j ≠ i
    5. Overall Davies-Bouldin Index (DBI)
        - The average of maximum similarity ratios (Ri) for all K clusters.

        Note: 
        - A lower DBI indicates better clustering, as it implies that clusters are compact and well-separated.
'''

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from pathlib import Path
import pandas as pd

'''
Generate synthetic dataset using make_blobs
make_blobs : Generates isotropic Gaussian blobs for clustering.
             :generates mock 2D data points that naturaly form 4 distinct clusters.
AgglomerativeClustering : A hierarchical clustering method that merges clusters iteratively based on a linkage criterion.
                        :  We loop through it testing k = 2 up to k = 6.
davies_bouldin_score : Computes the Davies-Bouldin Index for evaluating clustering quality.
                    : Measures cluster overlap and dispersion
                    : Lower values indicate better clustering.
                    : which k yields the minimum DBI value.
'''
# step 1 Create a simple Dataset with 4 natural groups (clusters)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
'''
Parameters for make_blobs:
    - n_samples: Total number of data points to generate.
    - centers: Number of distinct clusters to generate.
    - cluster_std: Standard deviation of the clusters.
    - random_state: Seed for reproducibility.
'''
# print(X[:5])  # Print the first 5 data points to verify the dataset
print("=" * 60)

# 2. Test different numbers of clusters (k from 2 to 6)
k_values = range(2, 7)
best_k = None
best_score = float('inf') # Start with infinity so any score is lower
best_labels = None
metrics_results = []

print("--- Clustering Metrics Evaluation (k = 2 to 6) ---")
for k in k_values:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    
    # 1. Davies-Bouldin Index (Lower is better)
    dbi = davies_bouldin_score(X, labels)
    
    # 2. Silhouette Score (Higher is better, range -1 to +1)
    sil = silhouette_score(X, labels)
    
    # 3. Calinski-Harabasz Index (Higher is better)
    chi = calinski_harabasz_score(X, labels)
    
    metrics_results.append({
        "k": k,
        "Davies-Bouldin (Lower=Best)": round(dbi, 4),
        "Silhouette (Higher=Best)": round(sil, 4),
        "Calinski-Harabasz (Higher=Best)": round(chi, 2)
    })
    print(f"k={k} | DBI: {dbi:.4f} (min) | Silhouette: {sil:.4f} (max) | CHI: {chi:.2f}")
    
    if dbi < best_score:
        best_score = dbi
        best_k = k
        best_labels = labels

print("=" * 60)
df_metrics = pd.DataFrame(metrics_results)
print("Summary Metrics Table:")
print(df_metrics.to_string(index=False))
print("=" * 60)
print(f"--> Optimal k: {best_k} with Lowest DBI: {best_score:.4f}")
print("=" * 60)

'''
3. Visualize the clusters for the optimal k and plot diagnostic curves
'''
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel 1: Clustered Data Scatter Plot
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='tab10', s=45, edgecolor='black', linewidth=0.5)
    ax1.set_title(f"Agglomerative Clustering (Optimal k = {best_k})\nBest DBI: {best_score:.4f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature 1 (X)", fontsize=11)
    ax1.set_ylabel("Feature 2 (Y)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Metrics Evaluation Curve across K
    ax2.plot(list(k_values), df_metrics["Davies-Bouldin (Lower=Best)"], marker='o', color='crimson', linewidth=2, label="DBI (Lower is Better)")
    ax2.plot(list(k_values), df_metrics["Silhouette (Higher=Best)"], marker='s', color='forestgreen', linewidth=2, label="Silhouette (Higher is Better)")
    ax2.axvline(best_k, color='blue', linestyle=':', label=f"Optimal k={best_k}")
    ax2.set_title("Clustering Validation Metrics vs Number of Clusters (k)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_xticks(list(k_values))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plot_file = Path(__file__).parent / "clustering_metrics_evaluation.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Evaluation plot saved successfully to: {plot_file}")
    plt.show()
except Exception as e:
    print(f"[Note] Plot display skipped ({e}). Plot saved to file.")


