'''
The Challenge:
Imagine you are a cricket data analyst.
Your organization has collected performance statistics for various international cricket players 
(including matches played, total runs, wickets taken, and catches). 
However, sports scouts are tired of manually categorizing players.
'''

'''
Goal: Categorize cricket players based on their performance statistics and
      accurately predict their primary playing role: Batsman, Bowler, or Allrounder.
'''

'''
Approach:
1. Clean and prepare raw data.
2. Feature Encoding and Scaling (StandardScaler)
3. Train a Model : use 80% of the data for training
4. Evaluate the Model : use the remaining 20% of the data for testing
5. Predict for new prospective players using inverse_transform
6. Visualize Feature Importances and Decision Tree Architecture
'''

from io import StringIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree

# =====================================================================
# STEP 1: Load and Inspect the Raw Cricket Data
# =====================================================================
# In-memory CSV dataset containing real-world performance statistics:
# matches, runs, wickets, catches, dominant hand (Right/Left), and playing role.
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

# Load CSV string into a pandas DataFrame using StringIO
df = pd.read_csv(StringIO(raw_csv_data))

# Display the first few rows of the DataFrame to inspect data structure
print("=" * 70)
print("CRICKET PERFORMANCE DATASET (First 5 Rows):")
print("=" * 70)
print(df.head())

# Inspect the target distribution (class balance)
print("\nThe Target feature is 'role'. Class Counts:")
print(df['role'].value_counts())

# Data validation: Check for null/missing values across all columns
print("\nMissing values in each column:")
print(df.isnull().sum())

# =====================================================================
# STEP 2: Data Preprocessing (Encoding & Feature Scaling)
# =====================================================================
# LabelEncoder converts categorical text labels into discrete numeric integers.

# 2.1 Encode Target Role: ['Allrounder', 'Batsman', 'Bowler'] -> [0, 1, 2]
role_encoder = LabelEncoder()
df['role_encoded'] = role_encoder.fit_transform(df['role'])

# 2.2 Encode Categorical Feature 'dominant_hand': ['Left', 'Right'] -> [0, 1]
hand_encoder = LabelEncoder()
df['hand_encoded'] = hand_encoder.fit_transform(df['dominant_hand'])

print("\n" + "=" * 70)
print("ENCODER MAPPINGS:")
print("=" * 70)
print("Role Encoder mapping:")
for idx, label in enumerate(role_encoder.classes_):
    print(f"  {label} -> {idx}")

print("Hand Encoder mapping:", list(zip(hand_encoder.classes_, range(len(hand_encoder.classes_)))))

# 2.3 Separate input features (X) and target variable (y)
feature_columns = ['matches', 'runs', 'wickets', 'catches', 'hand_encoded']
X = df[feature_columns]
y = df['role_encoded']

print("\nRaw Features Sample (First 5 rows):")
print(X.head())

# 2.4 Feature Scaling using StandardScaler:
# Centers data to mean = 0 and scales to unit variance (standard deviation = 1)
# Formula: z = (x - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

print("\nStandardized Features Sample (First 5 rows):")
print(X_scaled_df.head())

# =====================================================================
# STEP 3: Train-Test Split (80% Train, 20% Test)
# =====================================================================
# stratify=y guarantees that both training and testing partitions retain
# identical class proportions (e.g. balanced proportions of Batsmen, Bowlers, Allrounders).
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("DATASET PARTITION SIZES:")
print("=" * 70)
print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set size : {X_test.shape[0]} samples, {X_test.shape[1]} features")

# =====================================================================
# STEP 4: Train the Random Forest Model
# =====================================================================
# Random Forest creates an ensemble of 50 decision trees using bootstrap
# aggregation (bagging) and random feature subspace projection.
# criterion='gini': uses Gini Impurity (1 - sum(p_i^2)) to evaluate splits.
rf_clf = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    criterion="gini"
)
rf_clf.fit(X_train, y_train)

# =====================================================================
# STEP 5: Model Evaluation on Held-Out Test Data
# =====================================================================
y_pred = rf_clf.predict(X_test)

# Calculate overall accuracy
acc = accuracy_score(y_test, y_pred)
print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS:")
print("=" * 70)
print(f"Accuracy on test set: {acc * 100:.2f}%\n")

# Detailed per-class precision, recall, and f1-score
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=role_encoder.classes_))

# =====================================================================
# STEP 6: Predict Role for a Prospective Recruit
# =====================================================================
# Incoming raw player scout stats:
# [matches=130, runs=3500, wickets=105, catches=30, hand_encoded=1 (Right)]
new_player_df = pd.DataFrame([[130, 3500, 105, 30, 1]], columns=feature_columns)

# Crucial: Apply the SAME fitted StandardScaler transform to new player data
new_player_scaled = scaler.transform(new_player_df)
new_player_scaled_df = pd.DataFrame(new_player_scaled, columns=feature_columns)

# 1. Predict numeric role code using Random Forest
predicted_code = rf_clf.predict(new_player_scaled_df)

# 2. Decode numeric code back to text label using role_encoder.inverse_transform()
predicted_role = role_encoder.inverse_transform(predicted_code)

# 3. Predict class probability distribution (predict_proba)
predicted_probs = rf_clf.predict_proba(new_player_scaled_df)[0]
confidence = max(predicted_probs) * 100

print("=" * 70)
print("LIVE INFERENCE FOR NEW PLAYER:")
print("=" * 70)
print(f"Raw Stats       : 130 Matches | 3,500 Runs | 105 Wickets | 30 Catches | Hand: Right")
print(f"Predicted Role  : >>> {predicted_role[0]} <<< (Confidence: {confidence:.1f}%)")
print("Class Probability Breakdown:")
for cls_name, prob in zip(role_encoder.classes_, predicted_probs):
    print(f"  {cls_name:12s}: {prob * 100:.1f}%")

# =====================================================================
# STEP 7: Visualization (Feature Importances & Tree Architecture)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature Importance Bar Chart
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

axes[0].bar(
    range(len(feature_columns)),
    importances[indices],
    color='#3498db',
    edgecolor='black',
    linewidth=1.2
)
axes[0].set_xticks(range(len(feature_columns)))
axes[0].set_xticklabels([feature_columns[i] for i in indices], rotation=30, fontsize=10)
axes[0].set_title("Random Forest - Feature Importances", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Gini Importance Score", fontsize=11)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Add numerical percentage labels atop bars
for idx, bar_idx in enumerate(indices):
    score = importances[bar_idx]
    axes[0].text(idx, score + 0.01, f"{score*100:.1f}%", ha='center', fontsize=9, fontweight='bold')

# Subplot 2: Plot one of the Decision Trees inside the Forest
plot_tree(
    rf_clf.estimators_[0],
    feature_names=feature_columns,
    class_names=role_encoder.classes_,
    filled=True,
    rounded=True,
    ax=axes[1],
    fontsize=9
)
axes[1].set_title("Structure of Tree #1 inside the Random Forest", fontsize=13, fontweight='bold')

plt.tight_layout()
# Save visualization figure to disk
output_plot = Path(__file__).parent / "cricket_rf_solution.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n[+] Visualization dashboard saved successfully to: {output_plot}")

plt.close()

'''
Chronological Step-by-Step Workflow (The Roadmap)
Step 1: Load and Inspect the Raw Data
What to do: Read the raw CSV file into a Pandas DataFrame.

Why: Inspect the data types (text vs. numbers) and check for missing values or weird formatting.

Step 2: Data Preprocessing & Cleaning (The "Last Bit" Cleaning)
What to do:

Convert text categories like dominant_hand ("Right"/"Left") into numbers using LabelEncoder.

Convert the target labels (role: "Batsman", "Bowler", "All-Rounder") into numeric codes so the machine learning model can process them.

Why: Machine learning algorithms only understand numbers, not words.

Step 3: Feature Scaling (Squeezing the Values)
What to do: Use StandardScaler to transform features like runs (which can be 10,000+) and wickets (which might be 150) onto a uniform scale (mean = 0, variance = 1).

Why: It prevents large-number columns from dominating smaller-number columns during distance or impurity calculations.

Step 4: Split Data into Training and Testing Sets
What to do: Separate your features (X = matches, runs, wickets, etc.) from your target (y = role), and use train_test_split() to reserve 80% for training and 20% for testing.

Why: You must test your model on players it has never seen before to check if it has genuinely learned patterns rather than just memorizing the answers.

Step 5: Train the Random Forest Classifier
What to do: Initialize RandomForestClassifier(n_estimators=50) and .fit() it using your training data.

Why: The algorithm will build 50 different decision trees, introducing randomness into the data and features each time to ensure diversity.

Step 6: Evaluate Model Performance
What to do: Predict the roles for the test set and generate an accuracy_score and classification_report.

Why: To measure how well your "forest of trees" performs on unseen data.

Step 7: Make a Prediction on a New Player
What to do: Feed a custom player profile (e.g., someone with 150 matches, 4,500 runs, and 90 wickets) into your trained pipeline.

Why: To see the model in action answering real-world scouting questions.

Step 8: Visualize the Insights
What to do:

Plot a bar chart of Feature Importances to see whether runs or wickets drove the model's decisions.

Plot one of the individual decision trees using plot_tree().

Why: To make the "black box" of machine learning transparent and easy for your audience to interpret.
'''

