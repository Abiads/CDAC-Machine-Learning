"""
The Challenge:
==============
Imagine you are a cricket data analyst. Your organisation's coach and head of
scouting want an automated, data-driven system to classify player roles
(Batsman, Bowler, All-Rounder, Wicket-Keeper) based on player performance metrics:
- matches: Total career matches played
- runs: Total career runs scored
- wickets: Total career wickets taken
- catches: Total career catches taken
- dominant_hand: Batting / bowling handedness ('Right' or 'Left')
- role: Target role to predict

Scouts are tired of manually categorising players from raw scorecards.
Manual categorization is prone to human bias, ignores career length differences,
and struggles to separate specialized roles (e.g. wicket-keepers vs. batsmen,
or bowling all-rounders vs. pure bowlers).

================================================================================
CRICKET ANALYTICS APPROACH: MACHINE LEARNING WORKFLOW
================================================================================

1. Business & Domain Understanding:
   - Raw counts (runs, wickets, catches) scale with career longevity.
   - Rate metrics (Runs/Match, Wickets/Match, Catches/Match) isolate true player skill.
   - Categorical attributes like dominant_hand (Right vs Left) and role
     must be encoded using LabelEncoder and analyzed for predictive value.

2. Problem Formulation:
   - Task: Multi-Class Supervised Classification.
   - Target (y): ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper']
   - Features (X): ['matches', 'runs', 'wickets', 'catches', 'hand_encoded',
                    'runs_per_match', 'wickets_per_match', 'catches_per_match']

3. Feature Engineering & Preprocessing:
   - runs_per_match    = runs / matches (Batting efficiency)
   - wickets_per_match = wickets / matches (Bowling efficiency)
   - catches_per_match = catches / matches (Fielding/keeping efficiency)
   - hand_encoded      = 0 for 'Right', 1 for 'Left' (Binary encoding)
   - role_encoded      = LabelEncoder mapping roles to integer class IDs [0, 1, 2, 3]

4. Model Architecture & Selection:
   - Baseline: Single Decision Tree (criterion='entropy', max_depth=4)
     * Provides clear, transparent IF-THEN rules for coaches.
     * Uses Shannon Entropy & Information Gain to determine optimal split features.
   - Production Model: Random Forest Ensemble (n_estimators=100, max_depth=5)
     * Decorrelates individual trees using random bootstrap samples and feature subsets.
     * Calculates Out-Of-Bag (OOB) generalization accuracy and feature importances.

5. Model Evaluation:
   - Stratified Train-Test Split (75% Train, 25% Test).
   - Metrics: Precision, Recall, F1-Score, Confusion Matrix, OOB Accuracy.

6. Deployment / Scout Inference:
   - Automated helper function scout_evaluate_player() predicting role via
     role_encoder.inverse_transform() and confidence via predict_proba().
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# =====================================================================
# STEP 1: Benchmark Dataset Generation (Matches, Runs, Wickets, Catches, Hand, Role)
# =====================================================================
np.random.seed(42)
n_per_class = 60

# 1. Pure Batsmen: High runs (~35-52/match), near-zero wickets, moderate catches (slips ~0.35-0.75/match)
bat_matches = np.random.randint(15, 200, n_per_class)
bat_runs = np.maximum(50, bat_matches * np.random.uniform(35, 52, n_per_class) + np.random.normal(0, 80, n_per_class))
bat_wickets = np.random.poisson(lam=1.0, size=n_per_class)
bat_catches = np.maximum(2, bat_matches * np.random.uniform(0.35, 0.75, n_per_class) + np.random.normal(0, 3, n_per_class))
bat_hand = np.random.choice(['Right', 'Left'], size=n_per_class, p=[0.70, 0.30])

# 2. Pure Bowlers: Low runs (~4-11/match), high wickets (~1.2-2.2/match), low catches (~0.15-0.35/match)
bowl_matches = np.random.randint(15, 200, n_per_class)
bowl_runs = np.maximum(10, bowl_matches * np.random.uniform(4, 11, n_per_class) + np.random.normal(0, 30, n_per_class))
bowl_wickets = np.maximum(5, bowl_matches * np.random.uniform(1.2, 2.2, n_per_class) + np.random.normal(0, 4, n_per_class))
bowl_catches = np.maximum(1, bowl_matches * np.random.uniform(0.15, 0.35, n_per_class) + np.random.normal(0, 2, n_per_class))
bowl_hand = np.random.choice(['Right', 'Left'], size=n_per_class, p=[0.75, 0.25])

# 3. Genuine All-Rounders: Significant runs (~23-38/match) AND significant wickets (~0.8-1.5/match), moderate catches
all_matches = np.random.randint(15, 200, n_per_class)
all_runs = np.maximum(100, all_matches * np.random.uniform(23, 38, n_per_class) + np.random.normal(0, 60, n_per_class))
all_wickets = np.maximum(8, all_matches * np.random.uniform(0.8, 1.5, n_per_class) + np.random.normal(0, 4, n_per_class))
all_catches = np.maximum(2, all_matches * np.random.uniform(0.35, 0.75, n_per_class) + np.random.normal(0, 3, n_per_class))
all_hand = np.random.choice(['Right', 'Left'], size=n_per_class, p=[0.70, 0.30])

# 4. Specialized Wicket-Keepers: Good runs (~28-45/match), near-zero wickets, high catches (~1.2-2.2/match)
wk_matches = np.random.randint(15, 200, n_per_class)
wk_runs = np.maximum(80, wk_matches * np.random.uniform(28, 45, n_per_class) + np.random.normal(0, 60, n_per_class))
wk_wickets = np.random.poisson(lam=0.1, size=n_per_class)
wk_catches = np.maximum(15, wk_matches * np.random.uniform(1.2, 2.2, n_per_class) + np.random.normal(0, 4, n_per_class))
wk_hand = np.random.choice(['Right', 'Left'], size=n_per_class, p=[0.80, 0.20])

df = pd.DataFrame({
    'matches': np.concatenate([bat_matches, bowl_matches, all_matches, wk_matches]),
    'runs': np.round(np.concatenate([bat_runs, bowl_runs, all_runs, wk_runs])).astype(int),
    'wickets': np.round(np.concatenate([bat_wickets, bowl_wickets, all_wickets, wk_wickets])).astype(int),
    'catches': np.round(np.concatenate([bat_catches, bowl_catches, all_catches, wk_catches])).astype(int),
    'dominant_hand': np.concatenate([bat_hand, bowl_hand, all_hand, wk_hand]),
    'role': ['Batsman'] * n_per_class + ['Bowler'] * n_per_class + ['All-Rounder'] * n_per_class + ['Wicket-Keeper'] * n_per_class
})

# =====================================================================
# STEP 2: Feature Engineering & Preprocessing
# =====================================================================
# Domain efficiency rate metrics
df['runs_per_match'] = np.round(df['runs'] / df['matches'], 2)
df['wickets_per_match'] = np.round(df['wickets'] / df['matches'], 2)
df['catches_per_match'] = np.round(df['catches'] / df['matches'], 2)

# Categorical Feature Encoding (dominant_hand: 0 = Right, 1 = Left)
df['hand_encoded'] = (df['dominant_hand'] == 'Left').astype(int)

# Target Variable Encoding with LabelEncoder:
# Maps string target roles into integers: 0, 1, 2, 3
role_encoder = LabelEncoder()
df['role_encoded'] = role_encoder.fit_transform(df['role'])

print("=" * 80)
print("CRICKET SCOUTING DATASET SAMPLE (Matches, Runs, Wickets, Catches, Hand, Role):")
print("=" * 80)
print(df[['matches', 'runs', 'wickets', 'catches', 'dominant_hand', 'role', 'role_encoded', 'runs_per_match', 'wickets_per_match', 'catches_per_match']].head(8))

print("\n" + "=" * 80)
print("LABEL ENCODER CLASS MAPPING:")
print("=" * 80)
for idx, label in enumerate(role_encoder.classes_):
    print(f"  Class Index {idx} -> {label}")

print("\n" + "=" * 80)
print("CLASS PERFORMANCE PROFILE (Mean Ratios by Role):")
print("=" * 80)
profile = df.groupby('role')[['runs_per_match', 'wickets_per_match', 'catches_per_match', 'runs', 'wickets', 'catches']].mean().round(2)
print(profile)

feature_cols = [
    'matches', 'runs', 'wickets', 'catches', 'hand_encoded',
    'runs_per_match', 'wickets_per_match', 'catches_per_match'
]
X = df[feature_cols]
y = df['role_encoded']

# Stratified Split (guarantees balanced representation across all 4 roles)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =====================================================================
# STEP 3: Baseline Single Decision Tree Modeling (Entropy Criterion)
# =====================================================================
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)
dt_model.fit(X_train, y_train)

print("\n" + "=" * 80)
print("DECISION TREE SPLIT RULES (Human-Interpretable Scout Guidelines):")
print("=" * 80)
tree_rules = export_text(
    dt_model,
    feature_names=feature_cols,
    class_names=list(role_encoder.classes_)
)
print(tree_rules)

# =====================================================================
# STEP 4: Random Forest Ensemble Modeling (100 Estimators)
# =====================================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=5,
    oob_score=True,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=" * 80)
print(f"RANDOM FOREST OUT-OF-BAG (OOB) ACCURACY: {rf_model.oob_score_ * 100:.2f}%")
print("=" * 80)

print("\nRANDOM FOREST CLASSIFICATION REPORT (Test Set):")
print("-" * 80)
print(classification_report(y_test, y_pred_rf, target_names=role_encoder.classes_, digits=4))

print("FEATURE IMPORTANCE RANKINGS (Information Value Breakdown):")
print("-" * 80)
importances = rf_model.feature_importances_
feature_rank = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

for _, row in feature_rank.iterrows():
    bar = "█" * int(row['Importance'] * 40)
    print(f"  {row['Feature']:20s} : {row['Importance']:.4f} ({row['Importance']*100:.1f}%) {bar}")

# =====================================================================
# STEP 5: Live Automated Scout Inference Engine
# =====================================================================
def scout_evaluate_player(
    player_name: str,
    matches: int,
    runs: int,
    wickets: int,
    catches: int,
    dominant_hand: str = 'Right'
):
    """
    Automated scouting function to classify any incoming player profile
    given matches, runs, wickets, catches, and dominant hand.
    
    Demonstrates:
    1. Input feature formatting and domain ratio calculation
    2. Model prediction returning numeric code: rf_model.predict(input_data)
    3. Decoding numeric code back to text: role_encoder.inverse_transform(pred_code)
    4. Probabilistic confidence estimation: rf_model.predict_proba(input_data)
    """
    rpm = runs / matches if matches > 0 else 0
    wpm = wickets / matches if matches > 0 else 0
    cpm = catches / matches if matches > 0 else 0
    hand_enc = 1 if dominant_hand.strip().capitalize() == 'Left' else 0
    
    input_data = pd.DataFrame([{
        'matches': matches,
        'runs': runs,
        'wickets': wickets,
        'catches': catches,
        'hand_encoded': hand_enc,
        'runs_per_match': rpm,
        'wickets_per_match': wpm,
        'catches_per_match': cpm
    }])

    # -----------------------------------------------------------------
    # STEP 5.1: Predict Encoded Role (User's Change Implemented)
    # -----------------------------------------------------------------
    # The Random Forest outputs the predicted integer class ID [0, 1, 2, or 3]
    predicted_code = rf_model.predict(input_data)

    # -----------------------------------------------------------------
    # STEP 5.2: Decode with role_encoder.inverse_transform()
    # -----------------------------------------------------------------
    # Decode the numeric ID back to the human-readable string role:
    # 0 -> 'All-Rounder', 1 -> 'Batsman', 2 -> 'Bowler', 3 -> 'Wicket-Keeper'
    pred_role = role_encoder.inverse_transform(predicted_code)[0]

    # -----------------------------------------------------------------
    # STEP 5.3: Compute Posterior Class Probabilities (predict_proba)
    # -----------------------------------------------------------------
    # predict_proba returns the array of probability estimates for each role
    probs = rf_model.predict_proba(input_data)[0]
    classes = role_encoder.classes_
    prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probs)}
    
    return {
        'Player': player_name,
        'Matches': matches,
        'Runs': runs,
        'Wickets': wickets,
        'Catches': catches,
        'Hand': dominant_hand,
        'Runs/Match': round(rpm, 1),
        'Wickets/Match': round(wpm, 2),
        'Catches/Match': round(cpm, 2),
        'Predicted Role': pred_role,
        'Confidence': f"{max(probs)*100:.1f}%",
        'Class Probabilities': prob_dict
    }

print("\n" + "=" * 80)
print("LIVE SCOUT INFERENCE ENGINE (Evaluating Prospective Recruits):")
print("=" * 80)

prospects = [
    ("Prospect A (Top-Order Specialist)", 85, 3850, 4, 48, "Right"),
    ("Prospect B (Strike Bowler)", 110, 620, 195, 25, "Left"),
    ("Prospect C (Premier All-Rounder)", 95, 2950, 105, 52, "Right"),
    ("Prospect D (Wicket-Keeper Batsman)", 75, 2480, 0, 125, "Right"),
    ("Prospect E (Borderline Bowling All-Rounder)", 45, 780, 52, 18, "Left")
]

for p in prospects:
    eval_result = scout_evaluate_player(p[0], p[1], p[2], p[3], p[4], p[5])
    print(f"\nPlayer: {eval_result['Player']}")
    print(f"  Profile: {eval_result['Matches']} Matches | {eval_result['Runs']} Runs ({eval_result['Runs/Match']} R/M) | {eval_result['Wickets']} Wickets ({eval_result['Wickets/Match']} W/M) | {eval_result['Catches']} Catches ({eval_result['Catches/Match']} C/M) | Hand: {eval_result['Hand']}")
    print(f"  Coach Decision: >>> {eval_result['Predicted Role']} <<< (Confidence: {eval_result['Confidence']})")
    print(f"  Distribution: {eval_result['Class Probabilities']}")

# =====================================================================
# STEP 6: Multi-Panel Visual Analytics Dashboard
# =====================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

color_map = {
    'Batsman': '#2ca02c',
    'Bowler': '#d62728',
    'All-Rounder': '#1f77b4',
    'Wicket-Keeper': '#ff7f0e'
}

# Subplot 1: Batting vs Bowling Impact
for role in role_encoder.classes_:
    sub = df[df['role'] == role]
    ax1.scatter(
        sub['runs_per_match'],
        sub['wickets_per_match'],
        c=color_map[role],
        label=role,
        edgecolor='k',
        s=60,
        alpha=0.8
    )
ax1.set_title("1. Batting vs Bowling Impact (R/M vs W/M)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Runs Per Match", fontsize=10)
ax1.set_ylabel("Wickets Per Match", fontsize=10)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Fielding / Wicket-Keeping Specialization
for role in role_encoder.classes_:
    sub = df[df['role'] == role]
    ax2.scatter(
        sub['catches_per_match'],
        sub['runs_per_match'],
        c=color_map[role],
        label=role,
        edgecolor='k',
        s=60,
        alpha=0.8
    )
ax2.axvline(x=0.98, color='#666666', linestyle='--', linewidth=1.5, label='Tree Keeper Split (C/M = 0.98)')
ax2.set_title("2. Fielding Impact Isolating Wicket-Keepers", fontsize=11, fontweight='bold')
ax2.set_xlabel("Catches Per Match", fontsize=10)
ax2.set_ylabel("Runs Per Match", fontsize=10)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Random Forest Feature Importance Pareto Chart
bars = ax3.barh(feature_rank['Feature'], feature_rank['Importance'] * 100, color='#1f77b4', edgecolor='black', linewidth=1.2)
ax3.set_title("3. Feature Importance (Scouting Information Value)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Predictive Importance (%)", fontsize=10)
ax3.grid(axis='x', linestyle='--', alpha=0.5)

for idx, val in enumerate(feature_rank['Importance'] * 100):
    ax3.text(val + 0.8, idx, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold')

plt.suptitle("Cricket Player Role Scouting: Matches, Runs, Wickets, Catches, Hand & Role", fontsize=13, fontweight='bold')
plt.tight_layout()

output_plot = Path(__file__).parent / "cricket_scout_classification.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[+] Visualization dashboard saved successfully to: {output_plot}")
