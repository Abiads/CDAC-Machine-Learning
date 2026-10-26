import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# DATASET: Predicting Job Hiring (20 records)
# ==========================================
# Designed to intentionally showcase:
# 1. Quasi-constant feature (laptop_owned -> almost all 1s)
# 2. Redundant / highly correlated feature (coding_score & typing_speed)
# 3. Missing values (interview_score has NaNs)
# 4. Outlier (years_experience has an extreme value: 35)
# 5. Categorical with rare categories (preferred_os has 'Linux' only once)
# 6. Skewed distribution (salary_expectation)
# ==========================================

data = {
    "years_experience": [1, 2, 3, 2, 4, 5, 6, 3, 5, 7, 8, 2, 4, 6, 5, 3, 35, 4, 5, 2],
    "coding_score":     [45, 50, 60, 55, 65, 70, 75, 58, 72, 80, 85, 52, 68, 78, 74, 59, 95, 66, 71, 48],
    "typing_speed":     [46, 51, 61, 56, 66, 71, 76, 59, 73, 81, 86, 53, 69, 79, 75, 60, 96, 67, 72, 49], 
    "laptop_owned":     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
    "interview_score":  [70, 65, np.nan, 80, 85, 90, 75, 60, np.nan, 95, 88, 72, 78, 82, 88, 65, 92, 70, 80, 62], 
    "preferred_os":     ["Windows", "Mac", "Windows", "Linux", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", 
                         "Windows", "Mac", "Windows", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", "Windows"], 
    "salary_expectation": [30000, 32000, 35000, 33000, 40000, 45000, 50000, 34000, 46000, 55000, 
                           60000, 32000, 41000, 52000, 48000, 35000, 150000, 42000, 47000, 31000], 
    "hired":            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0] 
}

df = pd.DataFrame(data)
print(f"Original Dataset Shape: {df.shape}")
print("-" * 60)

# Separate Predictors (X) and Target (y)
X = df.drop(columns=["hired"])
y = df["hired"]
print("Initial Predictor Columns:", X.columns.tolist())
print("-" * 60)

# ==========================================
# 1. Quasi-Constant Feature Removal (laptop_owned)
# ==========================================
# laptop_owned is 95% ones -> variance = 0.95 * 0.05 = 0.0475 < threshold
vt_quasi = VarianceThreshold(threshold=0.05)
X_num = X.select_dtypes(include=[np.number])
vt_quasi.fit(X_num)
X_num = X_num.loc[:, vt_quasi.get_support()]
print("1. After Quasi-Constant Removal (Dropped 'laptop_owned'):", X_num.columns.tolist())

# ==========================================
# 2. Redundant / Highly Correlated Feature Removal (coding_score & typing_speed)
# ==========================================
# typing_speed is 100% redundant with coding_score (typing_speed = coding_score + 1)
corr = X_num[["coding_score", "typing_speed"]].corr().abs()
if corr.iloc[0, 1] > 0.90:
    X_num = X_num.drop(columns=["typing_speed"])
print("2. After Redundancy Removal (Dropped 'typing_speed'):", X_num.columns.tolist())

# ==========================================
# 3. Missing Value Imputation & Indicators (interview_score)
# ==========================================
X_num["interview_score_is_missing"] = X_num["interview_score"].isna().astype(int)
interview_median = X_num["interview_score"].median()
X_num["interview_score"] = X_num["interview_score"].fillna(interview_median)
print(f"3. Missing Imputed: 'interview_score' filled with median ({interview_median:.0f}) | Nulls left: {X_num['interview_score'].isna().sum()}")

# ==========================================
# 4. Outlier Treatment (years_experience extreme outlier = 35)
# ==========================================
cap = X_num["years_experience"].quantile(0.95)
X_num["years_experience"] = np.where(X_num["years_experience"] > cap, cap, X_num["years_experience"])
print(f"4. Outlier Capped: 'years_experience' capped at 95th percentile ({cap:.1f}) | Max: {X_num['years_experience'].max():.1f}")

# ==========================================
# 5. Categorical Encoding & Rare Categories (preferred_os)
# ==========================================
X_cat = X.select_dtypes(include=['object']).copy()
counts = X_cat["preferred_os"].value_counts()
# Linux appears only once (< 2 occurrences) -> group into 'Other'
X_cat["preferred_os"] = X_cat["preferred_os"].replace(counts[counts < 2].index, "Other")
X_cat = pd.get_dummies(X_cat, columns=["preferred_os"], drop_first=True, dtype=int)
print("5. Categorical Encoded 'preferred_os' (Rare 'Linux' -> 'Other'):", X_cat.columns.tolist())

# ==========================================
# 6. Skewed Distribution Transformation (salary_expectation)
# ==========================================
pt = PowerTransformer(method="yeo-johnson")
X_num["salary_expectation"] = pt.fit_transform(X_num[["salary_expectation"]])
print("6. Power Transformation (Yeo-Johnson) applied to 'salary_expectation'")

# ==========================================
# 7. Feature Scaling & Normalization (StandardScaler)
# ==========================================
scaler = StandardScaler()
scale_cols = ["years_experience", "coding_score", "interview_score", "salary_expectation"]
X_num[scale_cols] = scaler.fit_transform(X_num[scale_cols])
print("7. Standardization (Z-score) applied to:", scale_cols)

# Combine all processed numeric and categorical features
X_processed = pd.concat([X_num, X_cat], axis=1)
print(f"8. Final Combined Processed Matrix Shape: {X_processed.shape}")
print("-" * 60)

# ==========================================
# 8. Automated Feature Selection via ANOVA F-value
# ==========================================
selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(X_processed, y)
selected_mask = selector.get_support()
selected_features = X_processed.columns[selected_mask].tolist()
feature_scores = selector.scores_[selected_mask]

print("9. Top 4 Selected Features (via ANOVA F-value):")
for feat, score in zip(selected_features, feature_scores):
    print(f"   - {feat}: F-score = {score:.4f}")
print("-" * 60)

# ==========================================
# 9. Model Training, Prediction & Evaluation
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed[selected_features], y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("10. Model Evaluation on Test Set:")
print(f"    - Accuracy Score: {accuracy * 100:.2f}%")
print(f"    - Predicted Labels: {y_pred.tolist()}")
print(f"    - Actual Labels:    {y_test.values.tolist()}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("=" * 60)
print("Assignment 1 Completed Successfully!")