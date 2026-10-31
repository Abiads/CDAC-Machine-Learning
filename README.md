# CDAC Machine Learning Training (October 2026)

Comprehensive Machine Learning repository covering end-to-end data pipelines, supervised learning algorithms, unsupervised clustering, model evaluation metrics, and deployment pipelines developed during the CDAC Machine Learning curriculum.

---

## 📚 Curriculum & Repository Roadmap

| Module | Topic | Key Concepts & Techniques | Highlights & Files |
| :--- | :--- | :--- | :--- |
| [**Day 1**](./Day1/) | Data Ingestion & Setup | Data ingestion pipelines, Pandas/NumPy fundamentals, Telco customer churn dataset setup. | `O2_Data_Ingestion/`, `O2_Telco_Data_Ingestion/` |
| [**Day 2**](./Day2/) | Data Preprocessing & EDA | Missing value imputation, outlier detection, numerical scaling, categorical encoding, exploratory data visualization. | `O3_Data_Preprocessing_Cleaning_Eda/` |
| [**Day 3**](./Day3/) | Regression Algorithms | Simple & Multiple Linear Regression, Polynomial Regression, Cost functions, Gradient Descent, $R^2$ and RMSE evaluation. | Regression pipelines, feature scaling |
| [**Day 4**](./Day4/) | Classification & Features | Logistic Regression, Sigmoid activation, Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC curves, Stage-4 feature engineering. | `stage-04-feature-engineering/` |
| [**Day 5**](./Day5/) | Tree & Ensemble Methods | Decision Trees, Information Gain, Gini Impurity, Random Forests, Bagging, Out-of-Bag (OOB) error estimation. | Ensemble modeling, tree pruning |
| [**Day 6**](./Day6/) | Support Vector Machines & Tuning | Hard vs. Soft Margin SVM, Kernel Trick (RBF, Polynomial, Sigmoid), Hyperparameter Optimization (`GridSearchCV`, `RandomizedSearchCV`), Cross-Validation. | Classification boundary visualization |
| [**Day 7**](./Day7/) | Dimensionality Reduction | Principal Component Analysis (PCA), Eigenvalue decomposition, Scree plots, Explained Variance Ratio, t-SNE manifold learning. | Multi-dimensional projection |
| [**Day 8**](./Day8/) | Unsupervised Part 1: K-Means | Centroid-based clustering, Lloyd's algorithm, Inertia & Elbow Method, Silhouette Analysis, Customer Segmentation. | `Day_8_Machine_Learning_Summary.pdf`, customer datasets |
| [**Day 9**](./Day9/) | Unsupervised Part 2: Hierarchical & DBSCAN | Agglomerative Hierarchical Clustering, Linkage criteria (Ward, Complete, Average, Single), Dendrogram Cuts, Cophenetic Correlation, Metrics (DBI, Silhouette, CHI), DBSCAN (`eps`, `min_samples`), and Automated Epsilon Optimization (`kneed.KneeLocator`). | [`Day9/README.md`](./Day9/README.md), `hierarchical_pipeline.joblib`, `O07_clustering_DBSCAN.py`, `O08_optimaleps.py` |
| [**Day 10**](./Day10/) | Advanced Trees, Ensembles & Recommenders | Information Theory, Shannon Entropy ($H(X) = -\sum p \log_2 p$), Gini Impurity ($1 - \sum p^2$), Decision Trees, Random Forest Ensembles, Variance Reduction, and Item-Based Collaborative Filtering (MovieLens Recommendations). | `O01_entropy.ipynb`, `O02_decissiontree.ipynb`, `O03_random_forest.ipynb`, `O04_problem.py`, `O04_problem_to_solve.py`, `O05_problem_to_solve.py`, `O07_problem_to_solve.py`, `O08_problem_to_solve.ipynb`, `O08_problem_to_solve.py` |

---

## 🌟 Day 9 Deep Dive: Hierarchical Clustering & DBSCAN

For detailed documentation, mathematical equations, and usage guides, refer to [`Day9/README.md`](./Day9/README.md).

### Key Highlights of Day 9:
1. **Agglomerative Hierarchical Clustering**:
   - Bottom-up iterative clustering with distance matrices and linkage computation (`scipy.cluster.hierarchy.linkage`).
   - Dendrogram visualization and distance threshold cuts to determine optimal cluster counts.
2. **Linkage Benchmark & Cophenetic Correlation**:
   - Compared **Ward**, **Complete**, **Average**, and **Single** linkage methods.
   - Evaluated the **Cophenetic Correlation Coefficient ($r$)** measuring how faithfully dendrogram branch heights reflect original Euclidean distances.
3. **Clustering Validation Metrics**:
   - **Davies-Bouldin Index (DBI)** (lower is better): Measures worst-case cluster overlap.
   - **Silhouette Score** (higher is better, $[-1, +1]$): Measures intra-cluster cohesion vs. nearest-cluster separation.
   - **Calinski-Harabasz Index (CHI)** (higher is better): Ratio of between-cluster to within-cluster dispersion.
4. **Density-Based Spatial Clustering (DBSCAN)**:
   - Density connectivity through neighborhood radius ($\varepsilon$) and density threshold (`min_samples`).
   - Discovers non-linear clusters and automatically identifies outliers/noise points (labeled `-1`).
   - Implemented real-world customer segmentation on Annual Income vs. Spending Score.
5. **Production Pipelines**:
   - YAML-configured pipeline (`Day9/config.yaml`) integrating `StandardScaler` with `AgglomerativeClustering`.
   - Model artifact serialization with `joblib` and automated verification script (`O06_verify_heiarchical_pipeline.py`).

---

## 🛠️ Environment & Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed with the following packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml joblib kneed
```

### 2. Running Day 9 Scripts

```bash
# Run DBSCAN Customer Segmentation Dashboard (4-panel diagnostic)
python Day9/O07_clustering_DBSCAN.py

# Auto-Detect Optimal Epsilon using KneeLocator on Real Wine Data
python Day9/O08_optimaleps.py

# Run Density-Based Outlier Detection Notebook
jupyter notebook Day9/02_exercise_density_based_outliers.ipynb

# Run Production Agglomerative Pipeline (trains & serializes model)
python Day9/O04_agglomerative_clustering_pipeline.py

# Verify the Serialized Pipeline & Test Inference
python Day9/O06_verify_heiarchical_pipeline.py

# Run Clustering Validation Metrics (DBI, Silhouette, CHI)
python Day9/O05_metrics.py

# Run Hierarchical Linkage Comparison Notebook
jupyter notebook Day9/01_exercise_linkage_comparison.ipynb
```

---

## 📈 Visual Outputs

| Visualization | Script / Source | Purpose |
| :--- | :--- | :--- |
| `Day9/dbscan_customer_segmentation.png` | `Day9/O07_clustering_DBSCAN.py` | Visualizes 5 customer segments and outlier noise points (`-1`). |
| `Day9/agglomerative_clusters_pairplot.png` | `Day9/O04_agglomerative_clustering_pipeline.py` | 4D feature pairplot demonstrating cluster separation. |
| `Day9/clustering_metrics_evaluation.png` | `Day9/O05_metrics.py` | DBI, Silhouette, and CHI curves identifying optimal $k=4$. |
| `Day9/linkage_comparison_dendrograms.png` | `Day9/01_exercise_linkage_comparison.ipynb` | 4-way dendrogram comparison (Ward, Complete, Average, Single). |
| `Day10/cricket_scout_classification.png` | `Day10/O04_problem.py` | 3-panel scouting dashboard: feature space separation & Random Forest feature importances. |
| `Day10/cricket_rf_solution.png` | `Day10/O04_problem_to_solve.py` | 2-panel dashboard: feature importance rankings & single Decision Tree structure inside the Forest. |
| `Day10/hiring_decision_tree.png` | `Day10/O05_problem_to_solve.py` | 2-panel dashboard: full candidate hiring Decision Tree structure & Random Forest feature importances. |
| `Day10/random_forest_hiring.png` | `Day10/O07_problem_to_solve.py` | 2-panel dashboard: Random Forest feature importances & single tree structure for candidate hiring. |

---

## 👥 Contributors
- **abs6187** (`23f2000876@ds.study.iitm.ac.in`)
- **Instructor / Upstream**: Mohamed Ali ([@alilog](https://github.com/alilog))
