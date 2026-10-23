# 📘 Practical Machine Learning — Comprehensive Course Notes
### *CDAC ACTS Pune — PGCP-AI (August 2026 Batch)*
**Source Document:** `04_Practical Machine Learning.pdf`

---

## 📌 Course Structure & Evaluation Schema

- **Total Duration:** **120 Hours**
  - **Theory (T):** 60 Hours
  - **Lab / Hands-on (L):** 40 Hours
  - **Self-Learning (SL):** 20 Hours
- **Prerequisites:** Python Programming, Linear Algebra, Probability & Statistics.
- **Evaluation Weightage:**
  - 📝 **Theory Examination:** **40%**
  - 💻 **Lab Examination:** **40%**
  - 📊 **Internal Assessment:** **20%**
- **Primary Textbook:**
  - *Machine Learning using Python* — Manaranjan Pradhan & U Dinesh Kumar (Wiley India, 1st Edition).
- **Key References:**
  - Andreas Müller & Sarah Guido — *Introduction to Machine Learning with Python*
  - Sebastian Raschka — *Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2*
  - Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

---

## 🧭 Module-by-Module Technical Notes

### Module 1: ML Foundations & Data Engineering (Sessions 1–3)
- **ML Paradigm Breakdown:**
  - *Supervised Learning:* Mapping input $X$ to labeled ground truth $y$ (Regression for continuous outputs, Classification for discrete classes).
  - *Unsupervised Learning:* Discovering latent geometric structure, clusters, or distributions without target labels $y$.
  - *Reinforcement Learning:* Agent learning optimal policy $\pi(s)$ via reward signals from environment transitions.
- **Data Preprocessing & Hygiene:**
  - Missing Value Imputation: Univariate (Mean, Median, Mode) vs Multivariate (KNNImputer, IterativeImputer).
  - Categorical Encoding: Nominal (One-Hot / Dummy Encoding with $k-1$ degrees of freedom) vs Ordinal (Integer rank mapping).
  - Feature Scaling: 
    $$\text{StandardScaler (Z-score): } z = \frac{x - \mu}{\sigma}, \quad \text{MinMaxScaler: } x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
  - Outlier Detection: IQR rule ($[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) and Z-score thresholding ($|z| > 3$).
- **Feature Selection:** Filter methods (Variance threshold, Pearson Correlation, Mutual Information), Wrapper methods (RFE, Sequential Forward/Backward Selection), Embedded methods (Lasso L1 penalty, Tree feature importances).

---

### Module 2: Dimensionality Reduction & Unsupervised Clustering (Sessions 4–7)
- **Dimensionality Reduction:**
  - **Principal Component Analysis (PCA):** Unsupervised orthogonal linear projection maximizing variance. Solves eigendecomposition of the covariance matrix $\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$. Sensitive to scale; features must be standardized first.
  - **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear probabilistic technique optimal for low-dimensional (2D/3D) visualization. Preserves local neighborhood manifold structures using Student-t distribution in the low-dimensional map.
- **Clustering Algorithms:**
  - **K-Means Clustering:**
    - Centroid-based iterative optimization minimizing Within-Cluster Sum of Squares (WCSS / Inertia):
      $$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$
    - Determining optimal $K$: **Elbow Method** (inflection point of WCSS) and **Silhouette Coefficient** ($s = \frac{b - a}{\max(a, b)}$, where $s \in [-1, +1]$).
  - **Hierarchical Clustering:** Agglomerative (bottom-up) vs Divisive (top-down). Linkage criteria: Single (minimum distance), Complete (maximum distance), Average, and Ward's (minimum variance). Visualized using a **Dendrogram**.
  - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
    - Parameters: $\epsilon$ (neighborhood radius) and `minPts` (minimum samples required to form a dense region).
    - Point categorization: Core points, Border points, and Noise (outliers).
    - Advantages: Finds arbitrary non-spherical shapes, automatically identifies noise, does not require pre-specifying $K$.

---

### Module 3: Supervised Regression & Regularization (Sessions 8–9)
- **Linear Regression:** Minimizes Ordinary Least Squares (OLS) residual loss: $\mathcal{L}_{\text{OLS}} = \sum (y_i - \hat{y}_i)^2$.
  - Assumptions: Linearity, Independence (no autocorrelation), Homoscedasticity (constant variance of residuals), Normality of residuals, No multicollinearity ($VIF < 5$).
- **Polynomial & Stepwise Regression:** Modeling non-linear polynomial features ($\sum w_d x^d$); Forward selection, backward elimination, bidirectional stepwise testing $p$-values / AIC / BIC.
- **Regularization Frameworks (Preventing Overfitting & High Variance):**
  - **Ridge Regression (L2 Regularization):**
    $$\mathcal{L}_{\text{Ridge}} = \text{OLS} + \lambda \sum_{j=1}^p w_j^2$$
    Shrinks coefficients asymptotically toward zero; handles high multicollinearity.
  - **Lasso Regression (L1 Regularization):**
    $$\mathcal{L}_{\text{Lasso}} = \text{OLS} + \lambda \sum_{j=1}^p |w_j|$$
    Drives non-informative coefficients exactly to zero; performs automated feature selection.
  - **Elastic Net:** Convex combination of L1 and L2 penalties: $\lambda_1 ||w||_1 + \lambda_2 ||w||_2^2$.
- **Logistic Regression:** Generalized linear model for binary classification mapping linear predictor to probability space via the Sigmoid (Logit) function:
  $$P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = w^T X + b$$
  Loss function: Binary Cross-Entropy (Log-Loss).

---

### Module 4: Advanced Classification, SVM & Tuning (Sessions 10–14)
- **Support Vector Machines (SVM):**
  - Finds the optimal maximum-margin hyperplane separating classes: Margin $= \frac{2}{||w||}$.
  - Support vectors are the critical data points closest to the separating hyperplane.
  - Non-linear separation using Kernel Trick: Projects inputs to higher-dimensional Hilbert space without computing explicit coordinates:
    - Polynomial Kernel: $K(x, x') = (x^T x' + c)^d$
    - Radial Basis Function (RBF / Gaussian): $K(x, x') = \exp(-\gamma ||x - x'||^2)$
  - Regularization parameter $C$: High $C \rightarrow$ low bias, high variance (narrow margin); Low $C \rightarrow$ larger margin, permits misclassifications.
- **Hyperparameter Optimization & Validation:**
  - **K-Fold / Stratified K-Fold Cross-Validation:** Splits dataset into $k$ balanced folds to guard against train/validation leakage.
  - **GridSearchCV:** Exhaustive cartesian search over specified parameter grid.
  - **RandomizedSearchCV:** Stochastic sampling over parameter distributions; faster and scales better to high-dimensional parameter spaces.
- **K-Nearest Neighbors (KNN):** Instance-based (lazy) non-parametric classifier. Classifies queries by majority vote of $k$ closest neighbors using Euclidean / Manhattan / Minkowski distance metrics. Sensitive to feature scale.
- **Discriminant Analysis:**
  - Linear Discriminant Analysis (LDA): Assumes Gaussian distributions with identical class covariance matrices.
  - Quadratic Discriminant Analysis (QDA): Relaxes assumption; computes individual class covariance matrices (quadratic boundary).
- **Decision Trees (CART):**
  - Recursive binary splitting based on impurity reduction:
    - Gini Impurity: $I_G = 1 - \sum p_i^2$
    - Entropy & Information Gain: $H(S) = -\sum p_i \log_2(p_i); \quad IG = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$
  - Pruning: Pre-pruning (`max_depth`, `min_samples_split`, `min_samples_leaf`) and Cost-Complexity Post-pruning ($\alpha$).
- **Naïve Bayes Classifier:**
  - Probabilistic classifier rooted in Bayes' Theorem:
    $$P(C_k | X) = \frac{P(C_k) \prod_{i=1}^n P(x_i | C_k)}{P(X)}$$
  - "Naïve" assumption: Strong conditional independence between all input features given class label $C_k$.
  - Variants: GaussianNB (continuous), MultinomialNB (text/word counts), BernoulliNB (binary binary flags).

---

### Module 5: Ensemble Learning & Association Rule Mining (Sessions 15–18)
- **Ensemble Architectures:**
  - **Bagging (Bootstrap Aggregating — Random Forest):**
    - Trains $B$ parallel independent trees on bootstrap samples with random feature subspacing (`max_features`).
    - Aggregation reduces variance without increasing bias.
  - **Boosting (Sequential Error Correction):**
    - **AdaBoost:** Sequentially reweights misclassified observations.
    - **Gradient Boosting (GBM / XGBoost / LightGBM):** Sequentially fits new base learners to the pseudo-residuals (negative gradient of loss) of preceding models.
  - **Stacking (Stacked Generalization):** Trains multiple heterogeneous base models (e.g., SVM + RF + KNN) whose out-of-fold predictions feed into a meta-classifier (e.g., Logistic Regression).
- **Association Rule Mining & Market Basket Analysis:**
  - **Apriori Algorithm:** Exploits downward-closure property (any subset of a frequent itemset must be frequent).
  - Core Metrics:
    $$\text{Support}(A \rightarrow B) = P(A \cup B) = \frac{\text{freq}(A, B)}{N}$$
    $$\text{Confidence}(A \rightarrow B) = P(B | A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
    $$\text{Lift}(A \rightarrow B) = \frac{P(A \cup B)}{P(A) \cdot P(B)} = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
    *(Lift $> 1$ indicates positive association; Lift $= 1$ indicates statistical independence).*

---

### Module 6: Time Series Forecasting (Sessions 19–22)
- **Time Series Decomposition:** Trend ($T_t$), Seasonality ($S_t$), Cyclical ($C_t$), and Irregular/Noise ($\epsilon_t$) components (Additive: $Y_t = T+S+I$; Multiplicative: $Y_t = T \times S \times I$).
- **Smoothing Techniques:**
  - Simple Moving Average (SMA) and Weighted Moving Average (WMA).
  - Simple Exponential Smoothing (SES): Single parameter $\alpha$ for level (stationary series).
  - Holt’s Linear Trend Method: Two parameters ($\alpha$ for level, $\beta$ for trend).
  - Holt-Winters’ Method: Three parameters ($\alpha$ for level, $\beta$ for trend, $\gamma$ for seasonal cycle; Additive vs Multiplicative).
- **ARIMA Modeling Framework:**
  - **Stationarity:** Constant mean, constant variance, and autocovariance independent of time. Tested using **Augmented Dickey-Fuller (ADF) Test** ($p < 0.05 \rightarrow$ stationary).
  - **Differencing ($d$):** Number of non-seasonal differences required to induce stationarity.
  - **Auto-Regressive Component ($p$):** Model depends on its own lagged values: $Y_t = c + \sum_{i=1}^p \phi_i Y_{t-i} + \epsilon_t$. Identified via **PACF (Partial Autocorrelation Function)** cutoff.
  - **Moving Average Component ($q$):** Model depends on lagged forecast errors: $Y_t = \mu + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$. Identified via **ACF (Autocorrelation Function)** cutoff.
  - Combined Notation: **$\text{ARIMA}(p, d, q)$** (and **$\text{SARIMA}(p,d,q)(P,D,Q)_s$** for seasonal periods).

---

### Module 7: Real-Time ML & Performance Evaluation Metrics (Session 23)
- **Classification Evaluation Metrics:**
  - **Confusion Matrix:**
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
    $$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Quality of positive alarms})$$
    $$\text{Recall / Sensitivity} = \frac{TP}{TP + FN} \quad (\text{Coverage of actual positives})$$
    $$\text{Specificity} = \frac{TN}{TN + FP} \quad (\text{True negative rate})$$
    $$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{Harmonic mean})$$
  - **ROC Curve & AUC Score:** Plot of True Positive Rate ($Y$-axis) vs False Positive Rate ($X$-axis) across all operating decision thresholds $\theta \in [0, 1]$. Invariant to class distribution.
  - **Precision-Recall Curve (PR-AUC):** Superior diagnostic for heavily imbalanced target distributions.
- **Regression Evaluation Metrics:**
  - $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$ (Linear penalty, robust to outliers)
  - $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$ (Penalizes large errors heavily)
  - $\text{RMSE} = \sqrt{\text{MSE}}$ (Interpretable in original target units)
  - $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (Variance explained)

---

### Module 8: Recommendation Systems (Sessions 24–25)
- **Recommender Paradigms:**
  - *Content-Based Filtering:* Recommends items similar to user's historical preferences using item metadata and cosine similarity:
    $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||}$$
  - *Collaborative Filtering (CF):* Relies on collective user-item interaction matrix.
    - User-Based CF: "Users who agreed in the past agree in the future."
    - Item-Based CF: "Items liked together remain correlated."
- **Matrix Factorization (Latent Factor Models):**
  - Singular Value Decomposition (SVD): Decomposes interaction matrix $\mathbf{R} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
  - Low-rank factorization representing users and items in $k$-dimensional latent space $\mathbf{R} \approx \mathbf{P} \mathbf{Q}^T$.
- **Ranking Evaluation Metrics:**
  - **Mean Reciprocal Rank (MRR):** $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ (Reward for high position of first relevant item).
  - **Mean Average Precision at K (MAP@K):** Evaluates precision across top-$K$ recommendations.
  - **Normalized Discounted Cumulative Gain (NDCG@K):** Accounts for graded relevance and discounts positions exponentially:
    $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

### Module 9: Anomaly Detection (Session 26)
- **Anomaly Taxonomies:**
  - *Point Anomaly:* Individual instance anomalous relative to entire dataset (e.g., single credit card charge of \$50,000).
  - *Contextual Anomaly:* Instance anomalous within a specific context (e.g., \$500 winter heating bill in Arizona summer).
  - *Collective Anomaly:* Collection of related data points anomalous collectively even if individually normal (e.g., sudden flatline in ECG).
- **Isolation Forest Algorithm:**
  - Core Principle: Anomalies are few and structurally different; hence, they are isolated closer to tree roots with fewer random partitions.
  - Builds an ensemble of Isolation Trees (`iTrees`) using random splits.
  - Anomaly score derived from path length $h(x)$: shorter average path length $\rightarrow$ higher probability of anomaly.

---

## 🔬 Course Lab Assignments Matrix

| Session Range | Lab Topic / Objective | Typical Dataset | Core Tasks & Expected Deliverables |
|:---|:---|:---|:---|
| **Sessions 4 & 5** | K-Means Clustering & WCSS | Kaggle Customer / Mall Data | Scale features, compute WCSS for $K \in [1, 10]$, generate Elbow plot, calculate Silhouette score, profile clusters. |
| **Session 6** | Hierarchical Clustering | Kaggle Segmentation Dataset | Compute distance matrix, generate Dendrogram, apply Agglomerative clustering, compare linkage criteria (Ward vs Complete). |
| **Sessions 8 & 9** | Regression Comparison & Metrics | Real Estate / Housing Data | Train OLS, Ridge, Lasso, Polynomial, and Logistic Regression; evaluate and compare MAE, MSE, RMSE, $R^2$, and F1. |
| **Sessions 10 & 11** | Air Quality AQI Prediction & SVM | Kaggle Air Quality Dataset | Continuous AQI prediction via Linear Regression + 5-class classification via SVM (Very Good $\rightarrow$ Worst); GridSearch vs RandomizedSearch. |
| **Session 12** | KNN Preprocessing & Classification | Kaggle Classification Dataset | Feature scaling (Z-score), optimal $k$ search, distance metric benchmarking (Euclidean vs Manhattan). |
| **Session 13** | CART Decision Tree Modeling | UCI Machine Learning Repository | Train classification tree, extract feature importance, tune `max_depth` & `min_samples_split`, visualize tree splits. |
| **Session 14** | Naïve Bayes Classification | Text / Document Dataset | Text vectorization (TF-IDF), MultinomialNB training, prior & posterior probability calculation. |
| **Sessions 15 & 16** | Ensemble Models (RF & GBM) | Kaggle Tabular Dataset | Implement Random Forest and Gradient Boosting (XGBoost/LightGBM), evaluate Out-Of-Bag (OOB) score, compare feature importances. |
| **Sessions 17 & 18** | Market Basket Analysis via Apriori | Retail / Supermarket Receipts | Extract transaction lists, set `min_support` and `min_confidence`, generate association rules, rank by Lift. |
| **Sessions 19 & 20** | Time Series Smoothing | Kaggle Sales / Demand Data | Apply Simple Moving Average, Holt’s linear trend, and Holt-Winters’ seasonal smoothing; compare residual errors. |
| **Sessions 21 & 22** | ARIMA / SARIMA Modeling | Stock / Climate Time Series | Check ADF stationarity, plot ACF and PACF correlograms, determine $(p, d, q)$, train ARIMA and evaluate forecast residuals. |
| **Session 23** | Classification Diagnostics | Any Binary/Multiclass Data | Generate Confusion Matrix, classification report (Precision, Recall, F1), plot ROC-AUC curve. |
| **Sessions 24 & 25** | Movie Recommendation Engine | MovieLens Dataset | Build User-Item interaction matrix, implement Collaborative Filtering / SVD Matrix Factorization, evaluate Recall@K & NDCG. |
| **Session 26** | Anomaly Detection via Isolation Forest | Credit Card / Network Intrusion | Train Isolation Forest, tune contamination rate, extract anomaly scores, isolate and inspect outliers. |

---

## ⚡ Quick Formula & Metric Cheatsheet for Exams

- **WCSS (K-Means):** $\sum_{k=1}^K \sum_{x \in C_k} ||x - \mu_k||^2$
- **Silhouette Coefficient:** $s = \frac{b - a}{\max(a, b)} \in [-1, 1]$
- **Ridge (L2 Penalty):** $\text{Loss} = \text{OLS} + \lambda \sum w_j^2$ (asymptotic shrinkage, retains all features)
- **Lasso (L1 Penalty):** $\text{Loss} = \text{OLS} + \lambda \sum |w_j|$ (exact sparse zeroing, automated selection)
- **Support:** $P(A \cap B)$
- **Confidence:** $P(B | A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$
- **Lift:** $\frac{P(A \cap B)}{P(A)P(B)} = \frac{\text{Confidence}}{\text{Support}(B)}$
- **ARIMA Components:** $p$ = AR order (from PACF), $d$ = differencing order (from ADF), $q$ = MA order (from ACF)
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$
- **Cosine Similarity:** $\frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$
