"""
The Challenge:
==============
Imagine you are a cricket data analyst. Your organisation's coach and head of
scouting want an automated, data-driven system to classify player roles
(Batsman, Bowler, All-Rounder) based on player performance metrics:
- matches: Total matches played
- runs: Total career runs scored
- wickets: Total career wickets taken

However, scouts are tired of manually categorising players from raw scorecards.
Manual categorization is prone to human bias, ignores career length differences,
and struggles to distinguish borderline players (e.g., lower-order bowling
all-rounders vs. pure bowlers).

================================================================================
CRICKET ANALYTICS APPROACH: MACHINE LEARNING WORKFLOW
================================================================================

1. Business & Domain Understanding:
   - Pure counts (runs, wickets) are heavily confounded by career longevity
     (a player with 1,500 runs across 30 matches is elite; across 200 matches is a bowler).
   - Rate metrics (Runs per Match, Wickets per Match) isolate player capability
     from total exposure.

2. Problem Formulation:
   - Task: Multi-Class Supervised Classification.
   - Classes (y): ['Batsman', 'Bowler', 'All-Rounder'].
   - Features (X): ['matches', 'runs', 'wickets', 'runs_per_match', 'wickets_per_match'].

3. Feature Engineering:
   - runs_per_match    = runs / matches (Batting impact per game)
   - wickets_per_match = wickets / matches (Bowling impact per game)
   - all_rounder_ratio = (runs_per_match) * (wickets_per_match)

4. Model Architecture & Selection:
   - Baseline: Single Decision Tree (criterion='entropy', max_depth=3)
     * Provides clear, interpretable IF-THEN split rules that coaches can inspect.
     * Uses Shannon Entropy & Information Gain to pick the root feature.
   - Production Model: Random Forest Ensemble (n_estimators=100, max_depth=4)
     * Reduces single-tree variance through Bootstrap Aggregation (Bagging).
     * Decorrelates trees via random feature subsets at each split (sqrt(D)).
     * Delivers robust probabilities and Feature Importance rankings.

5. Model Evaluation:
   - Stratified Train-Test Split (75% Train, 25% Test).
   - Metrics: Precision, Recall, F1-Score, Confusion Matrix, and OOB Score.

6. Deployment / Scout Inference:
   - Automated helper function scout_evaluate_player() predicting role and confidence.
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
from sklearn.tree import DecisionTreeClassifier, export_text

# =====================================================================
# STEP 1: Benchmark Dataset Ingestion / Generation
# =====================================================================
np.random.seed(42)
n_per_class = 75

# 1. Pure Batsmen: High runs (~35-55/match), near-zero wickets
bat_matches = np.random.randint(15, 220, n_per_class)
bat_runs = np.maximum(50, bat_matches * np.random.uniform(34, 52, n_per_class) + np.random.normal(0, 100, n_per_class))
bat_wickets = np.random.poisson(lam=1.2, size=n_per_class)

# 2. Pure Bowlers: Low runs (~3-12/match), high wickets (~1.2-2.3/match)
bowl_matches = np.random.randint(15, 220, n_per_class)
bowl_runs = np.maximum(10, bowl_matches * np.random.uniform(4, 11, n_per_class) + np.random.normal(0, 40, n_per_class))
bowl_wickets = np.maximum(5, bowl_matches * np.random.uniform(1.2, 2.2, n_per_class) + np.random.normal(0, 4, n_per_class))

# 3. Genuine All-Rounders: Significant runs (~22-38/match) AND significant wickets (~0.7-1.5/match)
all_matches = np.random.randint(15, 220, n_per_class)
all_runs = np.maximum(100, all_matches * np.random.uniform(23, 37, n_per_class) + np.random.normal(0, 80, n_per_class))
all_wickets = np.maximum(8, all_matches * np.random.uniform(0.7, 1.4, n_per_class) + np.random.normal(0, 4, n_per_class))

df = pd.DataFrame({
    'matches': np.concatenate([bat_matches, bowl_matches, all_matches]),
    'runs': np.round(np.concatenate([bat_runs, bowl_runs, all_runs])).astype(int),
    'wickets': np.round(np.concatenate([bat_wickets, bowl_wickets, all_wickets])).astype(int),
    'role': ['Batsman'] * n_per_class + ['Bowler'] * n_per_class + ['All-Rounder'] * n_per_class
})

# =====================================================================
# STEP 2: Feature Engineering
# =====================================================================
# Domain ratios isolate player efficiency from total matches played
df['runs_per_match'] = np.round(df['runs'] / df['matches'], 2)
df['wickets_per_match'] = np.round(df['wickets'] / df['matches'], 2)

print("=" * 70)
print("CRICKET SCOUTING DATASET SAMPLE (First 5 records):")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("CLASS PERFORMANCE PROFILE (Mean Ratios per Role):")
print("=" * 70)
print(df.groupby('role')[['runs_per_match', 'wickets_per_match', 'runs', 'wickets']].mean().round(2))

feature_cols = ['matches', 'runs', 'wickets', 'runs_per_match', 'wickets_per_match']
X = df[feature_cols]
y = df['role']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =====================================================================
# STEP 3: Baseline Single Decision Tree Modeling (Entropy Criterion)
# =====================================================================
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n" + "=" * 70)
print("DECISION TREE SPLIT RULES (Human-Interpretable Scout Guidelines):")
print("=" * 70)
tree_rules = export_text(dt_model, feature_names=feature_cols)
print(tree_rules)

# =====================================================================
# STEP 4: Random Forest Ensemble Modeling (100 Estimators)
# =====================================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=4,
    oob_score=True,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=" * 70)
print(f"RANDOM FOREST OUT-OF-BAG (OOB) ACCURACY: {rf_model.oob_score_ * 100:.2f}%")
print("=" * 70)

print("\nRANDOM FOREST CLASSIFICATION REPORT (Test Set):")
print("-" * 70)
print(classification_report(y_test, y_pred_rf, digits=4))

print("FEATURE IMPORTANCE RANKINGS (Information Value):")
print("-" * 70)
importances = rf_model.feature_importances_
feature_rank = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

for _, row in feature_rank.iterrows():
    bar = "█" * int(row['Importance'] * 40)
    print(f"  {row['Feature']:20s} : {row['Importance']:.4f}  {bar}")

# =====================================================================
# STEP 5: Live Automated Scout Inference Engine
# =====================================================================
def scout_evaluate_player(player_name: str, matches: int, runs: int, wickets: int):
    """
    Automated scouting function to classify any incoming player profile.
    """
    rpm = runs / matches if matches > 0 else 0
    wpm = wickets / matches if matches > 0 else 0
    
    input_data = pd.DataFrame([{
        'matches': matches,
        'runs': runs,
        'wickets': wickets,
        'runs_per_match': rpm,
        'wickets_per_match': wpm
    }])
    
    pred_role = rf_model.predict(input_data)[0]
    probs = rf_model.predict_proba(input_data)[0]
    classes = rf_model.classes_
    prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probs)}
    
    return {
        'Player': player_name,
        'Matches': matches,
        'Runs': runs,
        'Wickets': wickets,
        'Runs/Match': round(rpm, 1),
        'Wickets/Match': round(wpm, 2),
        'Predicted Role': pred_role,
        'Confidence': f"{max(probs)*100:.1f}%",
        'Class Probabilities': prob_dict
    }

print("\n" + "=" * 70)
print("LIVE SCOUT INFERENCE ENGINE (Test Prospects):")
print("=" * 70)

prospects = [
    ("Prospect A (High Runs, Low Wickets)", 85, 3850, 4),
    ("Prospect B (Low Runs, High Wickets)", 110, 620, 195),
    ("Prospect C (Balanced Impact)", 95, 2950, 105),
    ("Prospect D (Borderline Lower Order)", 45, 780, 52)
]

for p in prospects:
    eval_result = scout_evaluate_player(p[0], p[1], p[2], p[3])
    print(f"\nPlayer: {eval_result['Player']}")
    print(f"  Stats: {eval_result['Matches']} Matches | {eval_result['Runs']} Runs ({eval_result['Runs/Match']} R/M) | {eval_result['Wickets']} Wickets ({eval_result['Wickets/Match']} W/M)")
    print(f"  Coach Recommendation: >>> {eval_result['Predicted Role']} <<< (Confidence: {eval_result['Confidence']})")
    print(f"  Distribution: {eval_result['Class Probabilities']}")

# =====================================================================
# STEP 6: Multi-Panel Visual Analytics Dashboard
# =====================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

color_map = {'Batsman': '#2ca02c', 'Bowler': '#d62728', 'All-Rounder': '#1f77b4'}

# Subplot 1: Feature Space Separation (Runs/Match vs Wickets/Match)
for role in ['Batsman', 'Bowler', 'All-Rounder']:
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

ax1.axvline(x=16.87, color='#666666', linestyle='--', linewidth=1.5, label='Tree Threshold (Runs/M)')
ax1.set_title("1. Cricket Player Role Clusters (Engineered Ratios)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Runs Per Match (Batting Metric)", fontsize=11)
ax1.set_ylabel("Wickets Per Match (Bowling Metric)", fontsize=11)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Random Forest Feature Importance Pareto Chart
ax2.barh(feature_rank['Feature'], feature_rank['Importance'] * 100, color='#1f77b4', edgecolor='black', linewidth=1.2)
ax2.set_title("2. Random Forest Feature Importance (Scouting Signals)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predictive Importance (%)", fontsize=11)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

for idx, val in enumerate(feature_rank['Importance'] * 100):
    ax2.text(val + 1, idx, f"{val:.1f}%", va='center', fontsize=10, fontweight='bold')

plt.suptitle("Automated Cricket Player Scouting Classification (Decision Tree vs Random Forest)", fontsize=14, fontweight='bold')
plt.tight_layout()

output_plot = Path(__file__).parent / "cricket_scout_classification.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n[+] Visualization dashboard saved successfully to: {output_plot}")

plt.close()
