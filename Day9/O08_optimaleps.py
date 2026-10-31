import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_wine
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# KneeLocator : A utility to find the "knee" or "elbow" point in a curve, often used to determine the optimal value of eps for DBSCAN.
# The "knee" point represents a point of maximum curvature, which is often considered the optimal trade-off between the number of clusters and the distance threshold (eps).

wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
print("--- Wine Dataset (First 5 Rows) ---")
print(df.head())

X_raw = df[['alcohol', 'malic_acid']]

# golden rule: always standardize your data before applying DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

min_samples = 5
neigh = NearestNeighbors(n_neighbors=min_samples)
neigh.fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled)

k_distances = np.sort(distances[:, -1])  # sort the k-distances in ascending order (farthest neighbor distance)
x = range(len(k_distances))

knee_locator = KneeLocator(x, k_distances, curve='convex', direction='increasing')
optimal_eps = knee_locator.knee_y

print(f"\n[+] Auto-detected optimal eps: {optimal_eps:.4f} (at index x = {knee_locator.knee})")

dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)

clusters = dbscan.fit_predict(X_scaled)
df['cluster'] = clusters
print("\n--- Clustering Result Sample ---")
print(df[['alcohol', 'malic_acid', 'cluster']].head())
print(df[['alcohol', 'malic_acid', 'cluster']].tail())

n_noise = int(np.sum(clusters == -1))
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"\n[+] Clusters formed: {n_clusters} | Noise points detected: {n_noise}")

# 5. Visualize the Results (2-Panel Figure: Knee Locator + Cluster Scatter)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: K-Distance Knee curve
ax1.plot(x, k_distances, color='#1f77b4', linewidth=2.2, label=f'{min_samples}-NN Distance')
ax1.axhline(y=optimal_eps, color='#d62728', linestyle='--', linewidth=1.5, label=f'Optimal $\\epsilon = {optimal_eps:.2f}$')
if knee_locator.knee is not None:
    ax1.plot(knee_locator.knee, optimal_eps, 'ro', markersize=8, label=f'Knee Point (x={knee_locator.knee})')
ax1.set_title('K-Distance Graph & Automated Knee Detection', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sorted Observations', fontsize=10)
ax1.set_ylabel(f'{min_samples}-th Nearest Neighbor Distance', fontsize=10)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Wine Clusters
scatter = ax2.scatter(
    X_raw['alcohol'], X_raw['malic_acid'], c=clusters, cmap='tab10', edgecolor='k', s=60
)
ax2.set_title(
    f'DBSCAN on Real Wine Data\n(Auto-detected $\\epsilon = {optimal_eps:.2f}$, min_samples = {min_samples})',
    fontsize=12, fontweight='bold'
)
ax2.set_xlabel('Alcohol Content', fontsize=10)
ax2.set_ylabel('Malic Acid', fontsize=10)
cbar = fig.colorbar(scatter, ax=ax2)
cbar.set_label('Cluster ID (-1 = Noise)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

output_plot = Path(__file__).parent / 'dbscan_wine_optimal_eps.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n[+] Plot saved successfully to: {output_plot}")

try:
    plt.show()
except Exception:
    pass
