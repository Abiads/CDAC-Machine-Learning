'''
The Challenge: PastHires Candidate Selection
=============================================
Goal: Predict whether a prospective job candidate will be HIRED (Yes/No)
      based on their qualifications and employment background.

Features:
- Years Experience   : Total years of professional work experience
- Employed           : Whether the candidate is currently employed ('Y'/'N')
- Previous Employers : Number of companies previously worked for
- Level of Education : Highest academic degree ('BS', 'MS', 'PhD')
- Top-tier School    : Graduated from a top-tier university ('Y'/'N')
- Interned           : Whether the candidate completed an internship ('Y'/'N')
Target:
- Hired              : Hiring outcome ('Y'/'N')
'''

import os
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# =====================================================================
# STEP 1: Download & Load the PastHires Dataset
# =====================================================================
# Define target input file paths (handles ./assets/ and ./assests/ gracefully)
base_dir = Path(__file__).parent
assets_dir = base_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

input_file = assets_dir / "PastHires.csv"
url = "https://raw.githubusercontent.com/ApnaClassroom/Dataset/main/PastHires.csv"

# Download dataset if not already present locally
if not input_file.exists():
    print(f"Downloading dataset from: {url}")
    urllib.request.urlretrieve(url, input_file)
    print(f"Saved dataset to: {input_file}")

# Load CSV into pandas DataFrame
df = pd.read_csv(input_file)

print("=" * 70)
print("PAST HIRES CANDIDATE DATASET (First 5 Rows):")
print("=" * 70)
print(df.head())

print("\nDataset Info & Missing Values:")
print(df.info())
print("\nTarget Class Distribution (Hired):")
print(df['Hired'].value_counts())

# =====================================================================
# STEP 2: Data Preprocessing & Categorical Feature Encoding
# =====================================================================
# Machine Learning models require numeric values.
# We map binary 'Y'/'N' columns to 1/0, and ordinal education degrees to integers.

# 2.1 Binary Mapping for Yes/No columns
binary_mapping = {'Y': 1, 'N': 0}

df['Employed'] = df['Employed'].map(binary_mapping)
df['Top-tier School'] = df['Top-tier School'].map(binary_mapping)
df['Interned'] = df['Interned'].map(binary_mapping)
df['Hired'] = df['Hired'].map(binary_mapping)

# 2.2 Ordinal Mapping for Level of Education: BS=0, MS=1, PhD=2
education_mapping = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(education_mapping)

print("\n" + "=" * 70)
print("ENCODED DATASET (Numeric Features):")
print("=" * 70)
print(df.head())

# =====================================================================
# STEP 3: Define Features (X) and Target (y)
# =====================================================================
feature_columns = [
    'Years Experience',
    'Employed',
    'Previous Employers',
    'Level of Education',
    'Top-tier School',
    'Interned'
]

X = df[feature_columns]
y = df['Hired']

print("\nFeature Matrix X:")
print(X.head())
print("\nTarget Vector y (1 = Hired, 0 = Not Hired):")
print(y.head())

# =====================================================================
# STEP 4: Train Decision Tree Classifier (Baseline & Interpretable Rules)
# =====================================================================
# Using entropy (Information Gain) to find the most informative root split
dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_clf.fit(X, y)

print("\n" + "=" * 70)
print("DECISION TREE SPLIT RULES:")
print("=" * 70)
# export_text accepts feature_names across all scikit-learn versions
tree_rules = export_text(dt_clf, feature_names=feature_columns)
# Map numeric class codes (0 -> Not Hired, 1 -> Hired) for readability
readable_rules = tree_rules.replace("class: 0", "class: Not Hired").replace("class: 1", "class: Hired")
print(readable_rules)

# =====================================================================
# STEP 5: Train Random Forest Classifier (Ensemble Model)
# =====================================================================
# 10 Decision Trees voting together to make robust hiring predictions
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X, y)

# Accuracy on the historical dataset
y_pred_rf = rf_clf.predict(X)
print("=" * 70)
print(f"Random Forest Training Accuracy: {accuracy_score(y, y_pred_rf) * 100:.2f}%")
print("=" * 70)

# Feature Importance ranking
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFEATURE IMPORTANCES (Scoring Signals):")
print("-" * 70)
for idx in indices:
    bar = "█" * int(importances[idx] * 40)
    print(f"  {feature_columns[idx]:20s} : {importances[idx]:.4f} ({importances[idx]*100:.1f}%) {bar}")

# =====================================================================
# STEP 6: Predict on New Job Applicants
# =====================================================================
# Candidate 1: 10 yrs experience, currently employed, 3 prev employers, BS (0), top-tier school (1), did intern (1)
# Candidate 2: 0 yrs experience, not employed, 0 prev employers, BS (0), not top-tier (0), did not intern (0)
# Candidate 3: 5 yrs experience, not employed, 2 prev employers, MS (1), top-tier (1), did intern (1)

new_candidates = pd.DataFrame([
    [10, 1, 3, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [5, 0, 2, 1, 1, 1]
], columns=feature_columns)

predictions = rf_clf.predict(new_candidates)
probabilities = rf_clf.predict_proba(new_candidates)

print("\n" + "=" * 70)
print("HIRING PREDICTIONS FOR NEW APPLICANTS:")
print("=" * 70)

candidate_profiles = [
    "Candidate 1 (10 Yrs Exp, Employed, 3 Prev, BS, Top-Tier, Interned)",
    "Candidate 2 (0 Yrs Exp, Unemployed, 0 Prev, BS, Non-Top-Tier, No Intern)",
    "Candidate 3 (5 Yrs Exp, Unemployed, 2 Prev, MS, Top-Tier, Interned)"
]

for name, pred, prob in zip(candidate_profiles, predictions, probabilities):
    result = ">>> HIRED <<<" if pred == 1 else ">>> NOT HIRED <<<"
    confidence = max(prob) * 100
    print(f"\n{name}")
    print(f"  Decision   : {result} (Confidence: {confidence:.1f}%)")
    print(f"  Probability: Not Hired = {prob[0]*100:.1f}%, Hired = {prob[1]*100:.1f}%")

# =====================================================================
# STEP 7: Visualization (Decision Tree Structure & Feature Importances)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Subplot 1: Visual Decision Tree
plot_tree(
    dt_clf,
    feature_names=feature_columns,
    class_names=['Not Hired', 'Hired'],
    filled=True,
    rounded=True,
    ax=axes[0],
    fontsize=9
)
axes[0].set_title("Decision Tree Structure for Candidate Hiring", fontsize=13, fontweight='bold')

# Subplot 2: Random Forest Feature Importances
axes[1].bar(
    range(len(feature_columns)),
    importances[indices],
    color='#2ecc71',
    edgecolor='black',
    linewidth=1.2
)
axes[1].set_xticks(range(len(feature_columns)))
axes[1].set_xticklabels([feature_columns[i] for i in indices], rotation=30, fontsize=10)
axes[1].set_title("Random Forest - Feature Importances", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Importance Score", fontsize=11)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for i, idx in enumerate(indices):
    score = importances[idx]
    axes[1].text(i, score + 0.01, f"{score*100:.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save figure to file
output_plot = base_dir / "hiring_decision_tree.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n[+] Visualization saved successfully to: {output_plot}")

# Safe display
try:
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
except Exception:
    plt.close()

