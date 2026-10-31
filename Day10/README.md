# Day 10: Advanced Decision Trees & Random Forest Ensembles

Welcome to **Day 10** of the CDAC Machine Learning curriculum. This module focuses on Information Theory in Machine Learning, impurity criteria (**Shannon Entropy** and **Gini Impurity**), **Decision Tree Induction**, and **Random Forest Ensemble Learning**.

---

## 📌 Table of Contents
1. [Core Theoretical Concepts](#-core-theoretical-concepts)
   - [Shannon Entropy & Information Gain](#1-shannon-entropy--information-gain)
   - [Gini Impurity](#2-gini-impurity)
   - [Gini vs. Entropy Comparison](#3-gini-vs-entropy-comparison)
   - [Decision Trees vs. Random Forests](#4-decision-trees-vs-random-forests)
   - [Ensemble Variance Reduction (Effect of n_estimators)](#5-ensemble-variance-reduction-effect-of-n_estimators)
2. [Notebook & Script Manifest](#-notebook--script-manifest)
3. [Execution Guide](#-execution-guide)

---

## 🧠 Core Theoretical Concepts

### 1. Shannon Entropy & Information Gain
**Entropy** is a fundamental thermodynamic and information-theoretic measure of disorder, uncertainty, or impurity in a system.
For a target variable with $C$ discrete classes, where $p_i$ is the empirical proportion of samples belonging to class $i$:
$$H(X) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

- **Pure Node** (all samples belong to the same class): $H(X) = - (1.0 \cdot \log_2(1.0)) = 0.0$ bits.
- **Maximum Impurity** (50/50 binary split): $H(X) = - (0.5 \log_2(0.5) + 0.5 \log_2(0.5)) = 1.0$ bit.
- **Information Gain (IG)**: The expected reduction in entropy achieved by partitioning dataset $T$ on attribute $A$:
  $$\text{IG}(T, A) = H(T) - \sum_{v \in \text{Values}(A)} \frac{|T_v|}{|T|} H(T_v)$$

---

### 2. Gini Impurity
**Gini Impurity** measures the probability that a randomly selected observation would be incorrectly labeled if classified according to the empirical class distribution of the node:
$$\text{Gini}(X) = 1 - \sum_{i=1}^{C} (p_i)^2$$

- **Theoretical Minimum**: $0.0$ (node is completely pure).
- **Theoretical Maximum** (for binary classification with equal split $0.5 / 0.5$): $1 - (0.5^2 + 0.5^2) = 0.5$.

---

### 3. Gini vs. Entropy Comparison

| Feature | Gini Impurity | Shannon Entropy |
| :--- | :--- | :--- |
| **Formula** | $1 - \sum p_i^2$ | $-\sum p_i \log_2(p_i)$ |
| **Range (Binary)** | $[0.0, 0.5]$ | $[0.0, 1.0]$ |
| **Computation Speed** | Faster (no logarithms required) | Slightly slower due to $\log_2$ evaluations |
| **Tree Balance** | Tends to isolate the largest class into pure nodes | Tends to generate slightly more balanced splits |
| **Scikit-Learn Default** | `criterion='gini'` | `criterion='entropy'` |

---

### 4. Decision Trees vs. Random Forests

A single Decision Tree recursively partitions the feature space into orthogonal hyperplanes. While highly interpretable:
- A single unconstrained tree tends to memorize noise and **overfit** (high variance, low bias).
- **Random Forests** solve this limitation through **Bootstrap Aggregation (Bagging)** and **Random Subspace Projection**:
  1. **Row Sampling with Replacement (Bootstrap)**: Each of the $B$ trees trains on a randomly drawn bootstrap sample of size $N$.
  2. **Feature Subsampling**: At each candidate split, only a random subset of $\sqrt{D}$ features is evaluated, decorrelating the trees.
  3. **Ensemble Aggregation**: Predictions are aggregated via majority voting (classification) or mean averaging (regression).

---

### 5. Ensemble Variance Reduction (Effect of n_estimators)

As demonstrated in [`Day10/O03_random_forest.ipynb`](./O03_random_forest.ipynb) on the noisy interleaving moons dataset:

- **$n\_estimators = 1$ (Single Decision Tree)**: Exhibits sharp, jagged, rectangular decision boundaries that wrap around individual noisy outliers.
- **$n\_estimators = 5$**: Decision boundaries begin to smooth out as competing trees vote down idiosyncratic errors.
- **$n\_estimators = 20$**: Boundaries conform smoothly to the true underlying non-linear manifold.
- **$n\_estimators = 100$**: Asymptotic variance reduction is achieved; the decision surface is robust against random perturbations and noise.

---

## 📁 Notebook & Script Manifest

| Notebook | Description | Key Modules / Output |
| :--- | :--- | :--- |
| [`O01_entropy.ipynb`](./O01_entropy.ipynb) | Pure Python and SciPy calculation of Shannon Entropy across pure, mixed, and skewed distributions. Basic `DecisionTreeClassifier(criterion='entropy')` role prediction. | `scipy.stats.entropy`, `collections.Counter`, `DecisionTreeClassifier` |
| [`O02_decissiontree.ipynb`](./O02_decissiontree.ipynb) | Theoretical derivations, mathematical curve plotting, and comparison table contrasting Gini Impurity vs. Shannon Entropy across binary class distributions. | Gini vs. Entropy curve plot, `1 - Σp²`, `-Σp log₂p` |
| [`O03_random_forest.ipynb`](./O03_random_forest.ipynb) | 4-panel visual comparison comparing a single Decision Tree vs. Random Forests ($n \in \{1, 5, 20, 100\}$) on noisy synthetic 2D moons data, mapping decision surfaces and classification accuracies. | `make_moons`, `RandomForestClassifier`, contour meshgrid plots |
| [`O04_problem.py`](./O04_problem.py) | **Real-World Case Study**: Cricket Scout & Coach Classification Challenge. Classifies players into `Batsman`, `Bowler`, `All-Rounder`, and `Wicket-Keeper` across `matches`, `runs`, `wickets`, `catches`, and `dominant_hand` using Decision Trees & Random Forests. | `RandomForestClassifier`, `DecisionTreeClassifier`, Feature Importance, `cricket_scout_classification.png` |
| [`O04_problem_to_solve.py`](./O04_problem_to_solve.py) | **Instructor Classroom Script**: Complete end-to-end Random Forest classification on raw cricket statistics with `LabelEncoder`, `StandardScaler`, stratified split, single tree architecture visualization, and prospective player prediction. | `RandomForestClassifier`, `StandardScaler`, `plot_tree`, `cricket_rf_solution.png` |
| [`O05_problem_to_solve.py`](./O05_problem_to_solve.py) | **PastHires Candidate Selection**: Predicting job hiring outcomes (`Hired`) from resume features (`Years Experience`, `Employed`, `Previous Employers`, `Level of Education`, `Top-tier School`, `Interned`) using Decision Trees & Random Forests. | `DecisionTreeClassifier`, `RandomForestClassifier`, `plot_tree`, `hiring_decision_tree.png` |

---

## 🏏 Case Study: Cricket Scouting Classification Challenge (`O04_problem.py`)

### 1. Feature Space & Attributes
- **Input Features**:
  - `matches`: Career matches played (exposure metric).
  - `runs`: Total career runs scored.
  - `wickets`: Total career wickets taken.
  - `catches`: Total career catches taken (isolates wicket-keepers & slip specialists).
  - `dominant_hand`: Batting / bowling handedness (`Right` or `Left`).
- **Target Variable (`role`)**:
  - `Batsman`, `Bowler`, `All-Rounder`, `Wicket-Keeper`.

### 2. End-to-End Machine Learning Workflow
1. **Feature Engineering & Preprocessing**:
   - $\text{Runs per Match (RPM)} = \frac{\text{Runs}}{\text{Matches}}$ (Batting efficiency)
   - $\text{Wickets per Match (WPM)} = \frac{\text{Wickets}}{\text{Matches}}$ (Bowling efficiency)
   - $\text{Catches per Match (CPM)} = \frac{\text{Catches}}{\text{Matches}}$ (Fielding/keeping efficiency)
   - $\text{Hand Encoded} = 0 \text{ for Right}, 1 \text{ for Left}$ (Categorical binary encoding)
2. **Interpretability via Decision Tree (Entropy Split)**:
   - Clear deterministic coach guidelines derived from Shannon Entropy & Information Gain:
     - $\text{wickets} \le 15.00$:
       - $\text{catches\_per\_match} \le 0.98 \rightarrow$ **Batsman**
       - $\text{catches\_per\_match} > 0.98 \rightarrow$ **Wicket-Keeper**
     - $\text{wickets} > 15.00$:
       - $\text{runs\_per\_match} \le 15.68 \rightarrow$ **Bowler**
       - $\text{runs\_per\_match} > 15.68 \rightarrow$ **All-Rounder**
3. **Robustness via Random Forest Ensemble (100 Trees)**:
   - Decorrelates individual trees using random bootstrap aggregation and feature subsets.
   - Out-of-Bag (OOB) accuracy: **100.00%**.
   - **Feature Selection Discovery**: The model reveals that `hand_encoded` has only $\approx 0.1\%$ importance (near-zero Information Gain), proving mathematically that handedness does not bias player role, while rate metrics (`catches_per_match`, `runs_per_match`, `wickets_per_match`) supply over $58\%$ of predictive information.
4. **Live Automated Scout Inference Engine**:
   - `scout_evaluate_player(player_name, matches, runs, wickets, catches, dominant_hand)` provides instantaneous role prediction and confidence distribution across all roles.


---

## 🚀 Execution Guide

To execute and inspect all Day 10 notebooks:

```bash
# Execute Entropy calculations
jupyter notebook Day10/O01_entropy.ipynb

# Execute Gini vs. Entropy derivations and curve comparisons
jupyter notebook Day10/O02_decissiontree.ipynb

# Execute Decision Tree vs. Random Forest Boundary Visualizations
jupyter notebook Day10/O03_random_forest.ipynb
```
