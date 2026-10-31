# Conceptual Summary: Hierarchical Clustering & DBSCAN

Comprehensive conceptual summary, algorithm mechanics, evaluation metrics, and cheat sheet for **Day 9 (Unsupervised Learning Part 2)**.

---

## 1. Agglomerative Hierarchical Clustering

A bottom-up, deterministic tree-building algorithm that groups observations iteratively based on pairwise linkage distance.

### Core Mechanics
1. **Initialization**: Every sample observation starts in its own singleton cluster: $C_i = \{x_i\}$.
2. **Linkage Step**: At each step, find the closest pair of clusters $(A, B)$ based on the linkage criterion:
   $$C_{\text{new}} := A \cup B$$
3. **Hierarchy Construction**: Repeat $N - 1$ times until all samples merge into a single root node (**Dendrogram**).
4. **Cluster Extraction**: Cut the tree horizontally at distance threshold $t$ (or extract $k$ discrete clusters).

### Linkage Methods Comparison

| Linkage | Formula / Definition | Properties & Use Cases |
| :--- | :--- | :--- |
| **Ward** | $d(A, B) = \sqrt{\frac{2|A||B|}{|A| + |B|}} \|\mu_A - \mu_B\|$ | Minimizes within-cluster variance ($\Delta \text{ESS}$). Produces balanced, compact, spherical clusters. Requires Euclidean metric. |
| **Complete** | $d(A, B) = \max_{u \in A, v \in B} \|u - v\|$ | Measures maximum distance between members. Prevents chaining, produces compact clusters of equal diameter. |
| **Average** | $d(A, B) = \frac{1}{|A||B|} \sum_{u \in A} \sum_{v \in B} \|u - v\|$ | Measures average pairwise distance. Robust compromise between single and complete linkage. |
| **Single** | $d(A, B) = \min_{u \in A, v \in B} \|u - v\|$ | Minimum pairwise distance. Detects arbitrary non-linear shapes, but suffers from *chaining* (forming stringy bridges). |

### Cophenetic Correlation Coefficient ($r$)
Measures how faithfully the dendrogram merge heights ($c_{ij}$) preserve the original pairwise Euclidean distances ($d_{ij}$):
$$r = \frac{\sum_{i < j} (d_{ij} - \bar{d})(c_{ij} - \bar{c})}{\sqrt{\sum_{i < j} (d_{ij} - \bar{d})^2 \sum_{i < j} (c_{ij} - \bar{c})^2}}$$
- $r \ge 0.75$ indicates high structural fidelity.

---

## 2. Clustering Quality & Validation Metrics

When ground-truth labels are absent, internal validation metrics quantify the balance between intra-cluster cohesion and inter-cluster separation:

### A. Davies-Bouldin Index (DBI) — *Lower is Better*
$$DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left( \frac{s_i + s_j}{d(\mu_i, \mu_j)} \right)$$
- Evaluates worst-case cluster overlap. Minimum possible value is 0.

### B. Silhouette Score ($s$) — *Higher is Better*
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- Where $a(i)$ is mean intra-cluster distance, $b(i)$ is mean nearest-neighbor cluster distance.
- Range: $[-1, +1]$. Scores $> 0.5$ indicate strong separation.

### C. Calinski-Harabasz Index (CHI) — *Higher is Better*
$$CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - k}{k - 1}$$
- Ratio of between-cluster variance ($B_k$) to within-cluster variance ($W_k$). Peaks at optimal $k$.

---

## 3. DBSCAN (Density-Based Spatial Clustering)

Non-parametric clustering algorithm that groups dense contiguous regions and isolates sparse noise.

### Core Definitions
- **$\varepsilon$ (Epsilon)**: Maximum neighborhood radius around each point.
- **$\text{min\_samples}$ (MinPts)**: Minimum points required within radius $\varepsilon$ to form a dense region.
- **Core Point**: Point with $\ge \text{min\_samples}$ within $\varepsilon$.
- **Border Point**: Point with $< \text{min\_samples}$ within $\varepsilon$, but reachable from a Core Point.
- **Noise Point**: Neither Core nor Border. Assigned cluster label **`-1`**.

> [!IMPORTANT]
> **Golden Rule**: Always standardize features using `StandardScaler` prior to DBSCAN. Distance metrics directly determine $\varepsilon$-neighborhood density.

---

## 4. Automated $\varepsilon$ Selection via KneeLocator

Instead of guessing $\varepsilon$:
1. Fit `NearestNeighbors(n_neighbors=min_samples)` on scaled data.
2. Compute the distance to each point's $k$-th nearest neighbor and sort ascending.
3. Locate the point of maximum curvature ("knee") using `kneed.KneeLocator(x, k_distances, curve='convex', direction='increasing')`.
4. `knee_locator.knee_y` provides the mathematically optimal $\varepsilon^*$.

---

## 5. Master Cheat Sheet & Scikit-Learn Templates

### Automated DBSCAN Template
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# 1. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Compute k-NN Distances
min_samples = 5
neigh = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

# 3. Auto-Detect Knee (Optimal Epsilon)
knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
optimal_eps = knee.knee_y

# 4. Fit DBSCAN
dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)
n_noise = (labels == -1).sum()
```

### Agglomerative Clustering Template
```python
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.cluster import AgglomerativeClustering

# 1. Compute Condensed Distances & Ward Linkage
dists = pdist(X_scaled, metric='euclidean')
Z = linkage(X_scaled, method='ward')

# 2. Check Cophenetic Correlation
r, _ = cophenet(Z, dists)

# 3. Fit Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cluster_labels = model.fit_predict(X_scaled)
```
