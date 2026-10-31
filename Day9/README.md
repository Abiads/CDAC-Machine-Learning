# Day 9: Unsupervised Learning — Hierarchical Clustering & DBSCAN

Welcome to **Day 9** of the CDAC Machine Learning curriculum. This module dives deep into advanced unsupervised learning techniques, focusing on **Agglomerative Hierarchical Clustering**, **Dendrogram Cut Analysis**, **Clustering Validation Metrics**, **Density-Based Spatial Clustering (DBSCAN)**, and **Automated $\varepsilon$ Hyperparameter Optimization via KneeLocator**.

---

## 📌 Table of Contents
1. [Core Theoretical Concepts](#-core-theoretical-concepts)
   - [Agglomerative Hierarchical Clustering](#1-agglomerative-hierarchical-clustering)
   - [Linkage Criteria Comparison](#2-linkage-criteria-comparison)
   - [Cophenetic Correlation Coefficient](#3-cophenetic-correlation-coefficient)
   - [Clustering Evaluation Metrics (DBI, Silhouette, CHI)](#4-clustering-evaluation-metrics)
   - [DBSCAN (Density-Based Spatial Clustering)](#5-dbscan-density-based-spatial-clustering)
   - [Automated Epsilon Selection via KneeLocator](#6-automated-epsilon-selection-via-kneelocator)
2. [File & Script Manifest](#-file--script-manifest)
3. [Step-by-Step Execution Guide ("What To Do")](#-step-by-step-execution-guide-what-to-do)
4. [Generated Artifacts & Plots](#-generated-artifacts--plots)

---

## 🧠 Core Theoretical Concepts

### 1. Agglomerative Hierarchical Clustering
Unlike Partitioning algorithms like K-Means which require a pre-specified $k$ and assume spherical clusters, **Agglomerative Hierarchical Clustering** operates bottom-up:
1. Every observation starts in its own individual singleton cluster: $C_i = \{x_i\}$.
2. At each iteration, the two clusters with the minimum linkage distance are merged:
   $$C_{\text{new}} = C_a \cup C_b$$
3. This process repeats until all points coalesce into a single root cluster, forming a hierarchical tree (**Dendrogram**).
4. Discrete cluster assignments are obtained by cutting the tree at a chosen horizontal distance threshold ($t$) or by specifying $k$ clusters.

---

### 2. Linkage Criteria Comparison

The distance between two clusters $A$ and $B$ depends on the chosen linkage method:

| Linkage Method | Mathematical Definition | Characteristics & Use Cases |
| :--- | :--- | :--- |
| **Ward** (`'ward'`) | $d(A, B) = \sqrt{\frac{2 |A||B|}{|A| + |B|}} \|\mu_A - \mu_B\|$ | Minimizes within-cluster variance. Produces compact, balanced, spherical clusters. Requires Euclidean distance. |
| **Complete** (`'complete'`) | $d(A, B) = \max_{u \in A, v \in B} \|u - v\|$ | Measures maximum distance between cluster pairs. Prevents chaining, handles non-spherical shapes, sensitive to outliers. |
| **Average** (`'average'` / UPGMA) | $d(A, B) = \frac{1}{\|A\|\|B\|} \sum_{u \in A}\sum_{v \in B} \|u - v\|$ | Measures average pairwise distance. Robust compromise between single and complete linkage. |
| **Single** (`'single'`) | $d(A, B) = \min_{u \in A, v \in B} \|u - v\|$ | Measures minimum pairwise distance. Can detect arbitrary shapes but suffers from *chaining* (forming stringy clusters). |

---

### 3. Cophenetic Correlation Coefficient
The **Cophenetic Distance** ($c_{ij}$) is the dendrogram height at which observation $i$ and observation $j$ are first united into the same cluster.

The **Cophenetic Correlation Coefficient** ($r$) is the Pearson correlation between the original pairwise Euclidean distances $d_{ij}$ and the cophenetic distances $c_{ij}$:
$$r = \frac{\sum_{i < j} (d_{ij} - \bar{d})(c_{ij} - \bar{c})}{\sqrt{\sum_{i < j} (d_{ij} - \bar{d})^2 \sum_{i < j} (c_{ij} - \bar{c})^2}}$$

- **Range**: $[-1, 1]$ (practically $[0, 1]$).
- **Interpretation**: A coefficient closer to $1.0$ indicates that the dendrogram faithfully preserves the true geometry and pairwise distances of the original feature space.

---

### 4. Clustering Evaluation Metrics

When ground-truth labels are unavailable, internal validation metrics quantify cluster quality:

#### A. Davies-Bouldin Index (DBI) — *Lower is Better*
Evaluates similarity between each cluster and its most similar counterpart based on cluster dispersion $s_i$ and centroid distance $d(\mu_i, \mu_j)$:
$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \quad DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} R_{ij}$$
- Lower values indicate tighter, well-separated clusters.

#### B. Silhouette Coefficient ($s$) — *Higher is Better*
Measures how similar point $i$ is to its own cluster compared to the nearest neighboring cluster:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where:
- $a(i)$: Mean intra-cluster distance from point $i$ to all other points in the same cluster.
- $b(i)$: Mean nearest-cluster distance from point $i$ to points in the closest neighboring cluster.
- Score ranges from $-1$ (misclassified) to $+1$ (dense, distinct separation). Scores above $0.5$ indicate strong cluster structure.

#### C. Calinski-Harabasz Index (CHI / Variance Ratio Criterion) — *Higher is Better*
Ratio of between-cluster dispersion ($B_k$) to within-cluster dispersion ($W_k$), scaled by degrees of freedom:
$$CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - k}{k - 1}$$
- Higher values represent denser, better-separated clusters.

---

### 5. DBSCAN (Density-Based Spatial Clustering)

**DBSCAN** groups points based on local density and discovers clusters of arbitrary shapes without requiring the number of clusters $k$ in advance.

#### Core Definitions:
- **$\varepsilon$ (eps)**: Radius of the neighborhood around a sample.
- **$\text{min\_samples}$ (MinPts)**: Minimum number of samples within the $\varepsilon$-neighborhood to form a dense region.
- **Core Point**: A point with at least $\text{min\_samples}$ within distance $\varepsilon$.
- **Border Point**: A point with fewer than $\text{min\_samples}$ within $\varepsilon$, but reachable from a Core Point.
- **Noise / Outlier**: A point neither Core nor Border. DBSCAN flags these with label **`-1`**.

> [!IMPORTANT]
> **Golden Rule of DBSCAN**: Always scale your features (e.g. using `StandardScaler`). Because DBSCAN computes Euclidean distance to determine whether points lie inside $\varepsilon$, unscaled features with large numerical magnitudes will overwhelm features on smaller scales.

---

### 6. Automated Epsilon Selection via KneeLocator

In DBSCAN, manually guessing $\varepsilon$ often leads to either merging distinct clusters (when $\varepsilon$ is too large) or labeling valid points as noise (when $\varepsilon$ is too small).
- **The $k$-NN Distance Method**: For a chosen $\text{min\_samples} = k$, compute the distance from each point to its $k$-th nearest neighbor. Sort these distances in ascending order.
- **Knee Point Detection**: The curve exhibits a point of maximum curvature (the "knee" or "elbow").
  - Points to the left of the knee have small $k$-NN distances and reside in dense clusters.
  - Points to the right of the knee exhibit rapid distance spikes and represent sparse noise.
- Using `kneed.KneeLocator(x, k_distances, curve='convex', direction='increasing')`, the algorithm automatically locates the exact mathematical threshold $\varepsilon^*$.

---

## 📁 File & Script Manifest

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| [`dataset/hierarchical_sample_data.csv`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/dataset/hierarchical_sample_data.csv) | CSV Dataset | 150 samples across 4 continuous numerical features engineered for hierarchical clustering. |
| [`dataset/spatial_density_data.csv`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/dataset/spatial_density_data.csv) | CSV Dataset | 200 spatial observations with 3 dense clusters and 20 noise outliers for range query benchmarking. |
| [`config.yaml`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/config.yaml) | YAML Config | Pipeline configuration specifying input paths, features, cluster count ($n=3$), metric (`euclidean`), and linkage (`ward`). |
| [`01_exercise_linkage_comparison.ipynb`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/01_exercise_linkage_comparison.ipynb) | Jupyter Notebook | Compares Ward, Complete, Average, and Single linkage; calculates Cophenetic correlations and plots 4 dendrogram subplots. |
| [`02_exercise_density_based_outliers.ipynb`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/02_exercise_density_based_outliers.ipynb) | Jupyter Notebook | Custom range query implementation `detect_density_outliers(X, eps, min_samples)` benchmarked 100% against Scikit-Learn DBSCAN. |
| [`O01_hiearchical_clustering.ipynb`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O01_hiearchical_clustering.ipynb) | Jupyter Notebook | Ice cream flavor clustering demonstration with distance matrix computation, dendrogram plotting, and distance cut. |
| [`O02_problem.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O02_problem.py) | Python Script | Flavor profiling with pairwise Euclidean distance calculation and SciPy linkage tree generation. |
| [`O03_hierarchical.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O03_hierarchical.py) | Python Script | Dendrogram visualization and threshold line cutting using SciPy. |
| [`O04_agglomerative_clustering_pipeline.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O04_agglomerative_clustering_pipeline.py) | Python Script | Production-grade pipeline using `sklearn.pipeline.Pipeline`, `StandardScaler`, and `AgglomerativeClustering`. Serializes model to `joblib`. |
| [`O05_metrics.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O05_metrics.py) | Python Script | Multi-metric cluster evaluation script computing DBI, Silhouette, and CHI across $k \in [2, 6]$, determining optimal $k=4$. |
| [`O06_verify_heiarchical_pipeline.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O06_verify_heiarchical_pipeline.py) | Python Script | Pipeline verification: loads serialized `.joblib` model, verifies preprocessing steps, and validates cluster labels. |
| [`O07_clustering_DBSCAN.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O07_clustering_DBSCAN.py) | Python Script | 4-panel customer segmentation dashboard using DBSCAN (`eps=0.5`, `min_samples=3`), identifying 5 clusters and noise (`-1`). |
| [`O08_optimaleps.py`](file:///c:/Users/conne/Downloads/ML_CDAC_ALILOG/Day9/O08_optimaleps.py) | Python Script | Automated Knee/Elbow detection using `kneed.KneeLocator` to auto-detect optimal $\varepsilon = 0.55$ on real Wine data. |

---

## 🚀 Step-by-Step Execution Guide ("What To Do")

Follow these steps to run and test all Day 9 modules:

### Step 1: Run DBSCAN Customer Segmentation Dashboard
Demonstrates density-based clustering, $\varepsilon$-neighborhoods, and outlier/noise detection:
```bash
python Day9/O07_clustering_DBSCAN.py
```
- **Expected Output**:
  - Displays customer income and spending dataframe with cluster assignments and personas.
  - Generates 4-panel high-res dashboard: `Day9/dbscan_customer_segmentation.png`.

---

### Step 2: Auto-Detect Optimal Epsilon on Real Wine Data
Uses `KneeLocator` to mathematically identify the optimal $\varepsilon$ parameter:
```bash
python Day9/O08_optimaleps.py
```
- **Expected Output**:
  - Automatically locates knee at index $160$ ($\varepsilon^* = 0.5468$).
  - Fits DBSCAN, forming 2 wine clusters with 10 noise points.
  - Generates 2-panel knee curve and wine cluster figure: `Day9/dbscan_wine_optimal_eps.png`.

---

### Step 3: Run Density-Based Outlier Detection Notebook
Execute the range query and custom noise classifier:
```bash
jupyter nbconvert --to notebook --execute Day9/02_exercise_density_based_outliers.ipynb
```
- **Expected Output**:
  - Custom `detect_density_outliers(X, eps=0.8, min_samples=5)` matches Scikit-Learn DBSCAN noise classification 100%.
  - Generates spatial visualization: `Day9/spatial_density_outliers_plot.png`.

---

### Step 4: Run Production Agglomerative Pipeline
Fits the pipeline on `hierarchical_sample_data.csv`, performs standardization, fits 3 clusters, and saves the joblib artifact:
```bash
python Day9/O04_agglomerative_clustering_pipeline.py
```
- **Expected Output**:
  - Serializes model to `hierarchical_pipeline.joblib`.
  - Generates seaborn pairplot: `Day9/agglomerative_clusters_pairplot.png`.

---

### Step 5: Verify Serialized Model Pipeline
Loads the serialized `.joblib` model and tests inference:
```bash
python Day9/O06_verify_heiarchical_pipeline.py
```

---

### Step 6: Run Clustering Validation Metrics
Calculates Davies-Bouldin Index, Silhouette Score, and Calinski-Harabasz Index across candidate $k$ values ($k \in [2, 6]$):
```bash
python Day9/O05_metrics.py
```
- **Expected Output**:
  - Identifies optimal $k=4$ based on minimum DBI ($0.3708$), maximum Silhouette ($0.6698$), and peak CHI ($326.68$).
  - Generates multi-panel evaluation plot: `Day9/clustering_metrics_evaluation.png`.

---

### Step 7: Run Linkage & Cophenetic Comparison Notebook
Open and run `Day9/01_exercise_linkage_comparison.ipynb`:
```bash
jupyter nbconvert --to notebook --execute Day9/01_exercise_linkage_comparison.ipynb
```
- **Expected Output**:
  - Generates 4-panel dendrogram comparison: `Day9/linkage_comparison_dendrograms.png`.

---

## 📊 Generated Artifacts & Plots

| Image File | Description |
| :--- | :--- |
| `Day9/dbscan_customer_segmentation.png` | 4-panel comprehensive DBSCAN dashboard: (1) Scaled space with $\epsilon$-neighborhood circles and Core/Border/Noise differentiation, (2) Real-world persona segmentation with cluster callouts, (3) 3-NN distance knee/elbow curve for $\epsilon$ selection, and (4) Cluster distribution & diagnostic breakdown card. |
| `Day9/dbscan_wine_optimal_eps.png` | 2-panel plot showing automated KneeLocator k-distance curve with detected optimal $\varepsilon = 0.55$ and real wine cluster scatter plot. |
| `Day9/spatial_density_outliers_plot.png` | Spatial scatter plot contrasting dense cluster observations against custom detected noise outliers. |
| `Day9/agglomerative_clusters_pairplot.png` | Seaborn pairplot of features 1–4 showing clear multi-dimensional separation of clusters. |
| `Day9/clustering_metrics_evaluation.png` | 3-panel line plot tracking DBI, Silhouette Score, and Calinski-Harabasz Index across $k \in [2, 6]$. |
| `Day9/linkage_comparison_dendrograms.png` | 4-way dendrogram comparison showing cluster merging heights across Ward, Complete, Average, and Single linkage. |
| `Day9/dendrogram_plot.png` | Dendrogram tree with horizontal threshold cut line separating sample groupings. |
