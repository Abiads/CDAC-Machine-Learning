'''
The Challenge: PastHires Candidate Selection using Random Forest
================================================================
Goal: Predict whether a prospective job candidate will be HIRED (Yes/No)
      using a Random Forest Classifier ensemble.

Dataset Features:
- Years Experience   : Total years of professional work experience
- Employed           : Whether the candidate is currently employed ('Y'/'N')
- Previous Employers : Number of companies previously worked for
- Level of Education : Highest academic degree ('BS', 'MS', 'PhD')
- Top-tier School    : Graduated from a top-tier university ('Y'/'N')
- Interned           : Whether the candidate completed an internship ('Y'/'N')
Target:
- Hired              : Hiring outcome ('Y'/'N' -> 1/0)
'''

import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# =====================================================================
# Step 1: Load the PastHires Dataset
# =====================================================================
# Candidate input file path handling (works whether run from root or Day10)
input_file = "./assets/PastHires.csv"
if not os.path.exists(input_file):
    input_file = Path(__file__).parent / "assets" / "PastHires.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("PAST HIRES DATASET (First 5 Rows):")
print("=" * 60)
print(df.head())
print("=" * 60)

# =====================================================================
# Step 2: Data Preprocessing & Categorical Encoding
# =====================================================================
# Map text categories to numeric integers:
# 'Y' -> 1, 'N' -> 0, 'BS' -> 0, 'MS' -> 1, 'PhD' -> 2
d = {'Y': 1, 'N': 0, 'BS': 0, 'MS': 1, 'PhD': 2}
df.replace(d, inplace=True)

# Ensure all object columns are cast to integer type
for col in df.select_dtypes(include=np.object_).columns.to_list():
    df[col] = df[col].astype(int)

print("Encoded Data Sample:")
print(df.head())
print("=" * 60)

# =====================================================================
# Step 3: Feature Scaling & Train-Test Split
# =====================================================================
features = list(df.columns[:6])
target = df.columns[6]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df[target]

# 80% Training, 20% Testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape : {X_test.shape}")
print("=" * 60)

# =====================================================================
# Step 4: Train the Random Forest Classifier
# =====================================================================
# Initialize an ensemble of 10 decision trees
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X_train, y_train)

print("Random Forest Classifier trained successfully.")
print("=" * 60)

# =====================================================================
# Step 5: Evaluate Model Accuracy
# =====================================================================
# Accuracy on the training set
y_train_pred = rf_clf.predict(X_train)
train_acc = (y_train_pred == y_train).mean()
print(f"Accuracy on the training data: {train_acc * 100:.2f}%")

# Accuracy on the test set
y_test_pred = rf_clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on the testing data : {test_acc * 100:.2f}%")
print("=" * 60)

print("Classification Report (Testing Data):")
print(classification_report(y_test, y_test_pred, target_names=['Not Hired', 'Hired'], zero_division=0))
print("=" * 60)

# =====================================================================
# Step 6: Feature Importances
# =====================================================================
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("FEATURE IMPORTANCE RANKINGS:")
print("-" * 60)
for idx in indices:
    bar = "█" * int(importances[idx] * 40)
    print(f"  {features[idx]:20s} : {importances[idx]:.4f} ({importances[idx]*100:.1f}%) {bar}")
print("=" * 60)

# =====================================================================
# Step 7: Predict for Prospective Candidates
# =====================================================================
# Candidate 1: 10 yrs exp, employed (1), 4 prev employers, BS (0), non-top-tier (0), no intern (0)
# Candidate 2: 0 yrs exp, unemployed (0), 0 prev employers, BS (0), top-tier (1), interned (1)
new_candidates_raw = pd.DataFrame([
    [10, 1, 4, 0, 0, 0],
    [0, 0, 0, 0, 1, 1]
], columns=features)

# Apply the same fitted scaler transform
new_candidates_scaled = pd.DataFrame(
    scaler.transform(new_candidates_raw),
    columns=features
)

predictions = rf_clf.predict(new_candidates_scaled)
probabilities = rf_clf.predict_proba(new_candidates_scaled)

print("PREDICTIONS FOR NEW CANDIDATES:")
print("-" * 60)
profiles = [
    "Candidate 1 (10 Yrs Exp, Employed, 4 Prev, BS, Non-Top-Tier, No Intern)",
    "Candidate 2 (0 Yrs Exp, Unemployed, 0 Prev, BS, Top-Tier, Interned)"
]

for name, pred, prob in zip(profiles, predictions, probabilities):
    outcome = ">>> HIRED <<<" if pred == 1 else ">>> NOT HIRED <<<"
    conf = max(prob) * 100
    print(f"{name}")
    print(f"  Outcome: {outcome} (Confidence: {conf:.1f}%)")
    print(f"  Probabilities: Not Hired = {prob[0]*100:.1f}%, Hired = {prob[1]*100:.1f}%\n")
print("=" * 60)

# =====================================================================
# Step 8: Visualizations (Feature Importances & Individual Tree Structure)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature Importance Bar Chart
axes[0].bar(
    range(len(features)),
    importances[indices],
    color='#27ae60',
    edgecolor='black',
    linewidth=1.2
)
axes[0].set_xticks(range(len(features)))
axes[0].set_xticklabels([features[i] for i in indices], rotation=30, fontsize=10)
axes[0].set_title("Random Forest - Feature Importances", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Importance Score", fontsize=11)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

for i, idx in enumerate(indices):
    score = importances[idx]
    axes[0].text(i, score + 0.01, f"{score*100:.1f}%", ha='center', fontsize=9, fontweight='bold')

# Subplot 2: Structure of Tree #1 from the Forest
plot_tree(
    rf_clf.estimators_[0],
    feature_names=features,
    class_names=['Not Hired', 'Hired'],
    filled=True,
    rounded=True,
    ax=axes[1],
    fontsize=9
)
axes[1].set_title("Structure of Tree #1 inside the Random Forest", fontsize=13, fontweight='bold')

plt.tight_layout()

output_plot = Path(__file__).parent / "random_forest_hiring.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Visualization saved to: {output_plot}")
