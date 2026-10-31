'''
The Challenge:
Imagine you are a cricket data analyst.
Your organization has collected performance statistics for various international cricket players 
(including matches played, total runs, wickets taken, and catches). 
However, sports scouts are tired of manually categorizing players.

Goal: Categorize cricket players based on their performance statistics and
      accurately predict their primary playing role: Batsman, Bowler, or Allrounder.

Approach:
1. Clean and prepare raw data.
2. Feature Encoding & Scaling
3. Train a Model : use 80% of the data for training
4. Evaluate the Model : use the remaining 20% of the data for testing
5. Predict role for a new player using inverse_transform
6. Visualize Feature Importances and Tree Structure
'''

import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# =====================================================================
# Raw Cricket Dataset (In-Memory CSV)
# =====================================================================
raw_csv_data = """matches,runs,wickets,catches,dominant_hand,role
150,6500,4,45,Right,Batsman
120,4800,10,38,Left,Batsman
200,8500,12,60,Right,Batsman
90,3200,2,25,Right,Batsman
180,7200,8,50,Left,Batsman
110,4200,5,30,Right,Batsman
75,2100,0,18,Right,Batsman
140,5900,15,42,Right,Batsman
130,250,180,20,Right,Bowler
160,400,220,15,Right,Bowler
95,150,130,12,Left,Bowler
210,600,310,25,Right,Bowler
85,100,110,8,Right,Bowler
175,500,260,18,Left,Bowler
105,300,145,10,Right,Bowler
190,800,290,22,Right,Bowler
140,3200,120,35,Right,Allrounder
165,4100,150,40,Left,Allrounder
115,2500,95,28,Right,Allrounder
195,5100,185,55,Right,Allrounder
80,1800,70,20,Right,Allrounder
135,3100,110,32,Left,Allrounder
170,4500,140,48,Right,Allrounder
100,2200,80,22,Right,Allrounder
160,3900,135,36,Right,Allrounder
"""

# =====================================================================
# Step 1 : Load the data into a pandas DataFrame
# =====================================================================
# Read string data directly using StringIO
df = pd.read_csv(StringIO(raw_csv_data))

# Display the first few rows of the DataFrame to verify the data
print("First 5 rows of data:")
print(df.head())

print("\nThe Target feature is 'role':")
print(df['role'].value_counts())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# =====================================================================
# Step 2 : Data Preprocessing (Encoding & Feature Scaling)
# =====================================================================

# 2.1 Encode target column 'role' into numeric integers (0, 1, 2)
role_encoder = LabelEncoder()
df['role_encoded'] = role_encoder.fit_transform(df['role'])

# 2.2 Encode categorical feature 'dominant_hand' into numbers (Left=0, Right=1)
hand_encoder = LabelEncoder()
df['hand_encoded'] = hand_encoder.fit_transform(df['dominant_hand'])

print("\nEncoder mapping for role:")
for idx, label in enumerate(role_encoder.classes_):
    print(f"  {label} -> {idx}")

print("\nHand Encoder mapping:", list(zip(hand_encoder.classes_, range(len(hand_encoder.classes_)))))

# 2.3 Select feature columns (X) and target column (y)
feature_columns = ['matches', 'runs', 'wickets', 'catches', 'hand_encoded']
X = df[feature_columns]
y = df['role_encoded']

print("\nSelected Features (First 5 rows):")
print(X.head())

# 2.4 Feature Scaling using StandardScaler
# Standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

print("\nScaled Features (First 5 rows):")
print(X_scaled_df.head())

# =====================================================================
# Step 3 : Split the data into training (80%) and testing (20%) sets
# =====================================================================
# stratify=y ensures all roles are proportionately split between train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("\n" + "=" * 60)
print("Training set size:", X_train.shape)
print("Testing set size :", X_test.shape)
print("=" * 60)

# =====================================================================
# Step 4 : Train the Random Forest Model
# =====================================================================
# n_estimators=50: build an ensemble of 50 decision trees
# criterion="gini": split based on Gini Impurity (1 - sum(p^2))
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, criterion="gini")
rf_clf.fit(X_train, y_train)

# =====================================================================
# Step 5 : Evaluate the Model
# =====================================================================
y_pred = rf_clf.predict(X_test)

# Calculate test accuracy:
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=role_encoder.classes_))

# =====================================================================
# Step 6 : Predict role for a new prospective player
# =====================================================================
# Player stats: 130 matches, 3500 runs, 105 wickets, 30 catches, Right handed (hand_encoded=1)
new_player_df = pd.DataFrame([[130, 3500, 105, 30, 1]], columns=feature_columns)

# Scale using the exact same fitted StandardScaler
new_player_scaled = scaler.transform(new_player_df)
new_player_scaled_df = pd.DataFrame(new_player_scaled, columns=feature_columns)

# Predict role code (e.g., 0, 1, or 2)
predicted_code = rf_clf.predict(new_player_scaled_df)

# Decode integer back to string role ('Allrounder', 'Batsman', or 'Bowler')
predicted_role = role_encoder.inverse_transform(predicted_code)

# Predict probability breakdown
probs = rf_clf.predict_proba(new_player_scaled_df)[0]

print("=" * 60)
print(f"Predicted role for the new player: >>> {predicted_role[0]} <<<")
print(f"Confidence: {max(probs)*100:.1f}%")
print("Class Probabilities:")
for role_name, prob in zip(role_encoder.classes_, probs):
    print(f"  {role_name:12s}: {prob * 100:.1f}%")
print("=" * 60)

# =====================================================================
# Step 7 : Visualizations (Feature Importance & Decision Tree Structure)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature Importance Bar Chart
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

axes[0].bar(range(len(feature_columns)), importances[indices], color='skyblue', edgecolor='black')
axes[0].set_xticks(range(len(feature_columns)))
axes[0].set_xticklabels([feature_columns[i] for i in indices], rotation=30)
axes[0].set_title("Random Forest - Feature Importances", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Importance Score")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Plot one of the Decision Trees inside the Forest
plot_tree(
    rf_clf.estimators_[0],
    feature_names=feature_columns,
    class_names=role_encoder.classes_,
    filled=True,
    rounded=True,
    ax=axes[1]
)
axes[1].set_title("Structure of Tree #1 inside the Random Forest", fontsize=14, fontweight='bold')

plt.tight_layout()

# Save figure to file
plt.savefig("Day10/cricket_scout_classification.png", dpi=300, bbox_inches='tight')
print("\n[+] Plot saved to: Day10/cricket_scout_classification.png")

# Display plot safely
try:
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
except Exception:
    plt.close()
