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
| [`O04_problem.py`](./O04_problem.py) | **Real-World Case Study**: Cricket Scout & Coach Classification Challenge. Classifies players into `Batsman`, `Bowler`, and `All-Rounder` using Decision Trees & Random Forests, featuring domain engineering and live scout inference. | `RandomForestClassifier`, `DecisionTreeClassifier`, Feature Importance, `cricket_scout_classification.png` |

---

## 🏏 Case Study: Cricket Scouting Classification Challenge (`O04_problem.py`)

### 1. Challenge Context
Cricket scouts manually inspecting raw scorecards (`matches`, `runs`, `wickets`) face heavy cognitive burden and bias:
- **Career Longevity Bias**: 1,000 runs scored over 25 matches indicates an elite batsman, whereas 1,000 runs over 200 matches indicates a lower-order bowler.
- **Borderline Ambiguity**: Differentiating genuine all-rounders from tail-end bowlers who bat occasionally.

### 2. End-to-End Machine Learning Approach
1. **Feature Engineering**:
   - $\text{Runs per Match} = \frac{\text{Runs}}{\text{Matches}}$
   - $\text{Wickets per Match} = \frac{\text{Wickets}}{\text{Matches}}$
   - Eliminates career length bias and exposes intrinsic player performance rates.
2. **Interpretability via Decision Tree (Entropy Split)**:
   - Provides clear, transparent decision rules coaches can follow:
     - `runs_per_match <= 15.91` $\rightarrow$ **Bowler**
     - `runs_per_match > 15.91` AND `wickets <= 12` $\rightarrow$ **Batsman**
     - `runs_per_match > 15.91` AND `wickets > 12` $\rightarrow$ **All-Rounder**
3. **Robustness via Random Forest Ensemble**:
   - 100 estimators with out-of-bag scoring (`oob_score=True`) achieving 100% accuracy.
   - Extracts feature importances showing that engineered rate ratios (`wickets_per_match`: 37.3%, `runs_per_match`: 31.6%) provide over 68% of predictive information.
4. **Live Automated Scout Inference Engine**:
   - Function `scout_evaluate_player(name, matches, runs, wickets)` predicting role, probability confidence, and distribution.


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
