"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
=====================================================================
Customer Segmentation on Annual Income vs. Spending Score.

Key Concepts:
- Density connectivity: Forms clusters based on local sample density.
- Arbitrary shapes: Unlike K-Means, does not assume spherical clusters.
- Noise robustness: Explicitly detects and isolates outliers (Cluster label -1).
- Golden Rule: Always standardize features before DBSCAN using StandardScaler.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. Real-world dataset: Customer Annual Income (k$) vs. Spending Score (1-100)
# =====================================================================
data = {
    'Annual_Income': [
        15,        16,        17,        18,        19,        20,
        25,        26,        28,        30,        50,        52,
        54,        55,        58,        60,        80,        82,
        85,        88,        90,        95,        110,       115,
        120,       125,
    ],
    'Spending_Score': [
        39,        81,        6,         77,        40,        76,
        72,        55,        60,        42,        50,        55,        
        48,        52,        54,        49,        15,        10,        
        12,        14,        18,        16,        80,        85,        
        79,        83,
    ],
}

df = pd.DataFrame(data)
print("--- Initial Dataset Sample ---")
print(df.head())

# =====================================================================
# 2. Standardization
# =====================================================================
"""
Golden rule before applying DBSCAN: always standardize your data.
DBSCAN is sensitive to the scale of the data, so standardization is crucial.
DBSCAN relies on distance to calculate eps.
If the data is not standardized, features with larger scales will dominate
the distance calculation, leading to suboptimal clustering results.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Annual_Income', 'Spending_Score']])

# =====================================================================
# 3. Apply DBSCAN
# =====================================================================
"""
DBSCAN parameters:
- eps: The maximum distance between two samples for them to be considered
       as in the same neighborhood. eps = epsilon = radius of neighborhood.
- min_samples: The number of samples in a neighborhood for a point to be
               considered as a core point.
"""
eps = 0.5
min_samples = 3

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)
df['Cluster'] = clusters

# Identify Point Types: Core, Border, and Noise
core_mask = np.zeros_like(clusters, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

point_types = []
for i in range(len(df)):
    if clusters[i] == -1:
        point_types.append("Noise")
    elif core_mask[i]:
        point_types.append("Core")
    else:
        point_types.append("Border")

df['Point_Type'] = point_types

# Business Persona Mapping
persona_names = {
    0: "Sensible Spenders (Low Inc, Mod Spend)",
    1: "Careless Spenders (Low Inc, High Spend)",
    2: "Balanced Middle Class (Mid Inc, Mid Spend)",
    3: "Frugal Savers (High Inc, Low Spend)",
    4: "VIP High Rollers (High Inc, High Spend)",
    -1: "Anomalous Outlier / Inactive",
}
df['Persona'] = df['Cluster'].map(persona_names)

print("\n--- DBSCAN Clustering Results with Point Types ---")
print(df[['Annual_Income', 'Spending_Score', 'Cluster', 'Point_Type', 'Persona']])

# Metrics
valid_mask = clusters != -1
n_clusters_formed = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = int(np.sum(clusters == -1))
n_core = int(np.sum(core_mask))
n_border = int(np.sum((clusters != -1) & (~core_mask)))
sil_val = silhouette_score(X_scaled[valid_mask], clusters[valid_mask]) if n_clusters_formed > 1 else 0.0

print(f"\n--- Diagnostic Summary ---")
print(f"Clusters Formed: {n_clusters_formed}")
print(f"Core Points:     {n_core} ({n_core / len(df) * 100:.1f}%)")
print(f"Border Points:   {n_border} ({n_border / len(df) * 100:.1f}%)")
print(f"Noise Points:    {n_noise} ({n_noise / len(df) * 100:.1f}%)")
print(f"Silhouette Score (excl. noise): {sil_val:.4f}")

# =====================================================================
# 4. Multi-Panel Comprehensive Visualization Dashboard
# =====================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "DBSCAN Customer Segmentation Dashboard\n"
    f"Parameters: $\\epsilon={eps}$, min_samples={min_samples} | "
    f"Clusters: {n_clusters_formed} | Noise Points: {n_noise} | Silhouette: {sil_val:.3f}",
    fontsize=16,
    fontweight='bold',
    y=0.98,
)

cluster_colors = {
    0: '#2ca02c',  # Green
    1: '#ff7f0e',  # Orange
    2: '#1f77b4',  # Blue
    3: '#9467bd',  # Purple
    4: '#e377c2',  # Pink
    -1: '#d62728', # Red for noise
}

# ---------------------------------------------------------------------
# Subplot 1: Scaled Feature Space (Z-Scores) & Epsilon Neighborhoods
# ---------------------------------------------------------------------
ax1 = plt.subplot(2, 2, 1)

# Draw eps neighborhood circles around core points
for idx in dbscan.core_sample_indices_:
    circ = patches.Circle(
        (X_scaled[idx, 0], X_scaled[idx, 1]),
        radius=eps,
        fill=True,
        alpha=0.08,
        facecolor=cluster_colors[clusters[idx]],
        edgecolor=cluster_colors[clusters[idx]],
        linestyle='--',
        linewidth=1.2,
    )
    ax1.add_patch(circ)

# Plot Core Points
for c in range(n_clusters_formed):
    mask = (clusters == c) & core_mask
    ax1.scatter(
        X_scaled[mask, 0],
        X_scaled[mask, 1],
        c=cluster_colors[c],
        s=100,
        edgecolor='black',
        linewidth=1.2,
        label=f"Cluster {c} (Core)",
        zorder=4,
    )

# Plot Border Points
for c in range(n_clusters_formed):
    mask = (clusters == c) & (~core_mask)
    if np.any(mask):
        ax1.scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            c=cluster_colors[c],
            s=120,
            marker='D',
            edgecolor='black',
            linewidth=1.5,
            label=f"Cluster {c} (Border)",
            zorder=5,
        )

# Plot Noise Points
if n_noise > 0:
    mask_noise = clusters == -1
    ax1.scatter(
        X_scaled[mask_noise, 0],
        X_scaled[mask_noise, 1],
        c=cluster_colors[-1],
        s=160,
        marker='X',
        linewidth=2.5,
        label="Noise / Outlier (-1)",
        zorder=6,
    )

ax1.set_title("1. Scaled Space (Z-Scores) with $\\epsilon$-Neighborhoods", fontsize=13, fontweight='bold')
ax1.set_xlabel("Scaled Annual Income (Z-Score)", fontsize=11)
ax1.set_ylabel("Scaled Spending Score (Z-Score)", fontsize=11)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.set_aspect('equal', 'datalim')
ax1.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------
# Subplot 2: Real-World Customer Persona Segmentation
# ---------------------------------------------------------------------
ax2 = plt.subplot(2, 2, 2)

for c in range(n_clusters_formed):
    mask = clusters == c
    subset = df[mask]
    ax2.scatter(
        subset['Annual_Income'],
        subset['Spending_Score'],
        c=cluster_colors[c],
        s=90,
        edgecolor='black',
        linewidth=1.0,
        label=f"Cluster {c}: {persona_names[c].split(' (')[0]}",
        zorder=4,
    )
    # Persona center label
    cx = subset['Annual_Income'].mean()
    cy = subset['Spending_Score'].mean()
    ax2.text(
        cx,
        cy + 4,
        f"Cluster {c}\n({len(subset)} customers)",
        fontsize=9,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle="round,pad=0.25", fc='white', ec=cluster_colors[c], lw=1.5, alpha=0.9),
        zorder=5,
    )

# Highlight Outliers
if n_noise > 0:
    noise_df = df[clusters == -1]
    ax2.scatter(
        noise_df['Annual_Income'],
        noise_df['Spending_Score'],
        c='#d62728',
        s=180,
        marker='X',
        linewidth=2.5,
        label="Outlier (Label -1)",
        zorder=6,
    )
    for _, row in noise_df.iterrows():
        ax2.annotate(
            f"Anomalous Outlier\n(Income: ${row['Annual_Income']}k, Spend: {row['Spending_Score']})",
            xy=(row['Annual_Income'], row['Spending_Score']),
            xytext=(row['Annual_Income'] + 8, row['Spending_Score'] + 14),
            arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=8),
            fontsize=9,
            fontweight='bold',
            color='#d62728',
            bbox=dict(boxstyle="round,pad=0.3", fc='#ffebee', ec='#d62728', lw=1.2),
            zorder=7,
        )

ax2.set_title("2. Real-World Customer Persona Segmentation", fontsize=13, fontweight='bold')
ax2.set_xlabel("Annual Income ($k)", fontsize=11)
ax2.set_ylabel("Spending Score (1-100)", fontsize=11)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------
# Subplot 3: 3-NN Distance Graph (Knee/Elbow Selection for Epsilon)
# ---------------------------------------------------------------------
ax3 = plt.subplot(2, 2, 3)

nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, min_samples - 1])

ax3.plot(range(len(k_distances)), k_distances, color='#1f77b4', marker='o', linewidth=2.0, markersize=5, label=f"{min_samples}-NN Distance Curve")
ax3.axhline(y=eps, color='#d62728', linestyle='--', linewidth=1.8, label=f"Selected $\\epsilon = {eps}$")

# Annotate knee
elbow_idx = len(k_distances) - n_noise - 1
ax3.scatter([elbow_idx], [k_distances[elbow_idx]], color='#2ca02c', s=140, zorder=5)
ax3.annotate(
    f"Optimal Knee ($\\epsilon \\approx {eps}$)\nCore/Border Boundary",
    xy=(elbow_idx, k_distances[elbow_idx]),
    xytext=(elbow_idx - 10, k_distances[elbow_idx] + 0.35),
    arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=7),
    fontsize=9,
    fontweight='bold',
    color='#2ca02c',
    bbox=dict(boxstyle="round,pad=0.25", fc='#e8f5e9', ec='#2ca02c', lw=1.2),
)

# Annotate Noise spike
noise_pt_idx = len(k_distances) - 1
ax3.scatter([noise_pt_idx], [k_distances[noise_pt_idx]], color='#d62728', s=140, zorder=5)
ax3.annotate(
    f"Noise Sample (dist = {k_distances[-1]:.2f})\nExceeds $\\epsilon={eps}$",
    xy=(noise_pt_idx, k_distances[noise_pt_idx]),
    xytext=(noise_pt_idx - 8, k_distances[noise_pt_idx] - 0.25),
    arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
    fontsize=9,
    fontweight='bold',
    color='#d62728',
    bbox=dict(boxstyle="round,pad=0.25", fc='#ffebee', ec='#d62728', lw=1.2),
)

ax3.set_title("3. 3-NN Distance Plot (Knee / Elbow Selection for $\\epsilon$)", fontsize=13, fontweight='bold')
ax3.set_xlabel("Samples Sorted by Distance to 3rd Nearest Neighbor", fontsize=11)
ax3.set_ylabel("Distance in Scaled Space", fontsize=11)
ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax3.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------
# Subplot 4: Cluster Composition & Point Type Breakdown
# ---------------------------------------------------------------------
ax4 = plt.subplot(2, 2, 4)

summary_counts = df['Cluster'].value_counts().sort_index()
cluster_labels_sorted = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in summary_counts.index]
bar_colors = [cluster_colors[c] for c in summary_counts.index]

bars = ax4.bar(cluster_labels_sorted, summary_counts.values, color=bar_colors, edgecolor='black', linewidth=1.2, width=0.55)

# Add values above bars
for bar in bars:
    height = bar.get_height()
    ax4.annotate(
        f"{height} pts\n({height / len(df) * 100:.1f}%)",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

# Summary statistics text box inside Subplot 4
stats_text = (
    "DBSCAN Diagnostics Summary:\n"
    "• Total Observations: 26\n"
    f"• Core Points: {n_core} ({n_core / len(df) * 100:.1f}%)\n"
    f"• Border Points: {n_border} ({n_border / len(df) * 100:.1f}%)\n"
    f"• Noise / Outliers: {n_noise} ({n_noise / len(df) * 100:.1f}%)\n"
    f"• Silhouette Score: {sil_val:.4f}\n"
    f"• Epsilon ($\\epsilon$): {eps}\n"
    f"• Min Samples: {min_samples}"
)
ax4.text(
    0.97,
    0.95,
    stats_text,
    transform=ax4.transAxes,
    fontsize=9.5,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#6c757d', linewidth=1.2),
)

ax4.set_ylim(0, max(summary_counts.values) + 2.5)
ax4.set_title("4. Cluster Distribution & Diagnostic Breakdown", fontsize=13, fontweight='bold')
ax4.set_xlabel("Identified Clusters", fontsize=11)
ax4.set_ylabel("Number of Customers", fontsize=11)
ax4.grid(axis='y', linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------
# Save and Show Figure
# ---------------------------------------------------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

output_plot = Path(__file__).parent / 'dbscan_customer_segmentation.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n[+] High-resolution dashboard saved successfully to: {output_plot}")

try:
    plt.show()
except Exception:
    pass
