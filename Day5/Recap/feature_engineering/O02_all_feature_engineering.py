'''
1. Zero-Variance Feature Removal
2. Quasi-Constant Feature Removal
3. Highly Correlated & Redundant Feature Removal
4. Missing Value Imputation & Missingness Indicators
5. Outlier Treatment (Winsorization / IQR Capping)
6. Categorical Encoding & Rare Category Handling
7. High-Cardinality Feature Management (Smoothed Target Encoding)
8. Mathematical & Power Transformations (Yeo-Johnson)
9. Feature Scaling & Normalization (StandardScaler)
10. Date-Time & Temporal Decomposition
11. Binning & Discretization
12. Dimensionality Reduction (PCA) & Automated Feature Selection (SelectKBest)
13. Model Training, Prediction & Evaluation (Following O01 Pattern)
'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    "always_zero": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], # 1. Zero-Variance
    "laptop_badge": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # 2. Quasi-Constant
    "hours_worked": [35, 40, 45, 38, 42, 50, 44, 39, 41, 48, 46, 37, 43, 49, 40, 36, 55, 42, 47, 38],
    "minutes_worked": [2100, 2400, 2700, 2280, 2520, 3000, 2640, 2340, 2460, 2880, 
                       2760, 2220, 2580, 2940, 2400, 2160, 3300, 2520, 2820, 2280], # 3. Highly Correlated with hours_worked
    "salary": [50000, 52000, np.nan, 53000, 60000, 75000, 62000, 51000, np.nan, 80000, 
               85000, 52000, 61000, 78000, 65000, 50000, 180000, 63000, 70000, 49000], # 4. Missing values & 5. Outlier (180000)
    "department": ["IT", "HR", "IT", "Sales", "IT", "HR", "Sales", "IT", "HR", "IT", 
                   "Sales", "IT", "HR", "IT", "Sales", "IT", "QuantumPhysics", "IT", "HR", "Sales"], # 6. Categorical with rare category ('QuantumPhysics')
    "postal_code": ["56001", "56002", "56003", "56004", "56005", "56006", "56007", "56008", "56009", "56010",
                    "56011", "56012", "56013", "56014", "56015", "56016", "56017", "56018", "56019", "56020"], # 7. High-Cardinality
    "bonus_amount": [100, 150, 200, 120, 300, 800, 400, 110, 250, 900, 
                     1000, 1300, 350, 850, 450, 90, 5000, 32050, 500, 105], # 8. Skewed continuous distribution
    "satisfaction_score": [3.2, 4.1, 4.5, 3.8, 4.0, 4.9, 4.2, 3.5, 3.9, 4.8, 
                           4.7, 3.6, 4.1, 4.8, 4.3, 3.1, 5.0, 4.2, 4.6, 3.4], # 11. Continuous for Binning
    "join_date": ["2022-01-10", "2021-05-15", "2023-03-20", "2020-11-05", "2022-07-12", "2019-01-01", "2021-08-19", "2023-02-14", "2022-09-30", "2018-04-11",
                  "2020-06-25", "2023-01-05", "2021-10-10", "2019-12-15", "2022-04-01", "2023-05-01", "2017-06-18", "2021-03-12", "2022-11-22", "2023-04-10"], # 10. Date-Time
    "promoted": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0] # Target variable
}

df = pd.DataFrame(data)
print("Original Dataset (df):")
print(df.head(5))
print("_" * 60)

# Separate Predictors (X) and Target (Y)
X = df.drop(columns=["promoted"])
Y = df["promoted"]

print("Original Features (X):")
print(X.head(5))
print("Target (Y):")
print(Y.values)
print("_" * 60)

# Step 1 : Zero-Variance Feature Removal
print("Step 1: Zero-Variance Feature Removal")
vt_zero = VarianceThreshold(threshold=0)
X_num = X.select_dtypes(include=[np.number])
print("Numeric Columns Before Zero-Variance Filter:")
print(X_num.columns.tolist())

vt_zero.fit(X_num)
surviving_step1 = X_num.columns[vt_zero.get_support()]
dropped_step1 = X_num.columns[~vt_zero.get_support()]
print("Surviving Features after Zero-Variance Filter:")
print(surviving_step1.tolist())
print("Dropped Features (Zero Variance):")
print(dropped_step1.tolist())
print("_" * 60)

# Rebuild Numeric DataFrame after Step 1
X_num = X_num[surviving_step1]
print("Rebuilt Numeric DataFrame (after Zero-Variance Filter):")
print(X_num.head(4))
print("_" * 60)

# Step 2 : Quasi-Constant Feature Removal
print("Step 2: Quasi-Constant Feature Removal")
quasi_threshold = 0.06 # Binary threshold for quasi-constant features p * (1-p) for 94% threshold
vt_quasi = VarianceThreshold(threshold=quasi_threshold)
vt_quasi.fit(X_num)
surviving_step2 = X_num.columns[vt_quasi.get_support()]
dropped_step2 = X_num.columns[~vt_quasi.get_support()]
print("Surviving Features after Quasi-Constant Filter:")
print(surviving_step2.tolist())
print("Dropped Features (Quasi-Constant):")
print(dropped_step2.tolist())
print("_" * 60)

# Rebuild Numeric DataFrame after Step 2
X_num = X_num[surviving_step2]
print("Rebuilt Numeric DataFrame (after Quasi-Constant Filter):")
print(X_num.head(4))
print("_" * 60)

# Step 3 : Highly Correlated & Redundant Feature Removal
print("Step 3: Highly Correlated & Redundant Feature Removal")
corr_matrix = X_num.corr().abs()
print("Correlation Matrix:")
print(corr_matrix)
print("_" * 60)
corr_threshold = 0.90
to_drop_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > corr_threshold:
            to_drop_corr.add(corr_matrix.columns[i])

print("Features to Drop due to High Correlation:")
print(list(to_drop_corr))
print("_" * 60)

X_num = X_num.drop(columns=list(to_drop_corr))
print("Final Numeric Features after Removing Highly Correlated Features:")
print(X_num.head(4))
print("_" * 60)

# Step 4 : Missing Value Imputation & Indicators
print("Step 4: Missing Value Imputation & Indicators")
print("Missing Values Count Before Imputation:")
print(X_num.isnull().sum())

# Create a binary missingness indicator column (1 if missing, 0 if present)
X_num["salary_is_missing"] = X_num["salary"].isna().astype(int)

# Median imputation for continuous numerical feature
salary_median = X_num["salary"].median()
X_num["salary"] = X_num["salary"].fillna(salary_median)
print(f"Salary Median Imputed: {salary_median}")
print("Missing Values Count After Imputation:")
print(X_num.isnull().sum())
print("Rebuilt DataFrame with Imputation & Missingness Indicator:")
print(X_num.head(4))
print("_" * 60)

# Step 5 : Outlier Treatment (IQR Capping / Winsorization)
print("Step 5: Outlier Treatment (IQR Capping / Winsorization)")
Q1 = X_num["salary"].quantile(0.25)
Q3 = X_num["salary"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print(f"Salary IQR Bounds: Lower = {lower_limit}, Upper = {upper_limit}")
print(f"Salary Max Before Outlier Capping: {X_num['salary'].max()}")

# Clip / Winsorize salary values within the IQR limits
X_num["salary"] = X_num["salary"].clip(lower=lower_limit, upper=upper_limit)
print(f"Salary Max After Outlier Capping: {X_num['salary'].max()}")
print("Salary Column Preview After Winsorization:")
print(X_num[["salary"]].head(4))
print("_" * 60)

# Step 6 : Categorical Encoding & Rare Category Handling
print("Step 6: Categorical Encoding & Rare Category Handling")
dept = X["department"].copy()
dept_freq = dept.value_counts(normalize=True)
print("Department Value Counts (Normalized Frequency):")
print(dept_freq)

# Group rare categories (<= 5% frequency) into 'Other'
rare_cats = dept_freq[dept_freq <= 0.05].index.tolist()
print("Rare Categories Detected (<= 5%):", rare_cats)
dept = dept.replace(rare_cats, "Other")
print("Department Categories after Grouping into 'Other':", dept.unique().tolist())

# One-Hot Encoding (drop_first=True to avoid dummy variable trap / multicollinearity)
dept_encoded = pd.get_dummies(dept, prefix="dept", drop_first=True, dtype=int)
print("One-Hot Encoded Department DataFrame (drop_first=True):")
print(dept_encoded.head(4))
print("_" * 60)

# Step 7 : High-Cardinality Feature Management (Smoothed Target Encoding)
print("Step 7: High-Cardinality Feature Management (Smoothed Target Encoding)")
smoothing = 5.0
global_mean = Y.mean()
postal_stats = pd.DataFrame({"postal_code": X["postal_code"], "target": Y})
stats = postal_stats.groupby("postal_code")["target"].agg(["count", "mean"])
smoothed_target = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)

postal_encoded = X["postal_code"].map(smoothed_target).fillna(global_mean)
postal_encoded_df = pd.DataFrame({"postal_code_encoded": postal_encoded})
print("Smoothed Target Encoded Postal Code DataFrame Preview:")
print(postal_encoded_df.head(4))
print("_" * 60)

# Step 8 : Mathematical & Power Transformations (Yeo-Johnson)
print("Step 8: Mathematical & Power Transformations (Yeo-Johnson)")
pt = PowerTransformer(method="yeo-johnson")
skew_before = X_num["bonus_amount"].skew()
bonus_transformed = pt.fit_transform(X_num[["bonus_amount"]])
X_num["bonus_transformed"] = bonus_transformed
skew_after = X_num["bonus_transformed"].skew()
print(f"Bonus Amount Skewness: Before = {skew_before:.2f}, After = {skew_after:.2f}")
print("Bonus Amount Transformed Preview:")
print(X_num[["bonus_amount", "bonus_transformed"]].head(4))
X_num = X_num.drop(columns=["bonus_amount"])
print("_" * 60)

# Step 9 : Feature Scaling & Normalization (StandardScaler)
print("Step 9: Feature Scaling & Normalization (StandardScaler)")
scaler = StandardScaler()
cols_to_scale = ["hours_worked", "salary", "bonus_transformed"]
scaled_vals = scaler.fit_transform(X_num[cols_to_scale])
X_scaled_df = pd.DataFrame(scaled_vals, columns=[f"{c}_scaled" for c in cols_to_scale], index=X.index)
print("Scaled Numerical Features (mean=0, std=1):")
print(X_scaled_df.head(4))
print("_" * 60)

# Step 10 : Date-Time & Temporal Decomposition
print("Step 10: Date-Time & Temporal Decomposition")
dates = pd.to_datetime(X["join_date"])
ref_date = pd.to_datetime("2026-01-01")
date_features = pd.DataFrame({
    "join_year": dates.dt.year,
    "join_month": dates.dt.month,
    "join_dayofweek": dates.dt.dayofweek,
    "tenure_years": ((ref_date - dates).dt.days / 365.25).round(2)
}, index=X.index)
print("Decomposed Temporal Features DataFrame:")
print(date_features.head(4))
print("_" * 60)

# Step 11 : Binning & Discretization
print("Step 11: Binning & Discretization")
satisfaction_binned = pd.cut(
    X["satisfaction_score"],
    bins=[0, 3.5, 4.5, 5.0],
    labels=["Low", "Medium", "High"]
)
print("Binned Satisfaction Distribution:")
print(satisfaction_binned.value_counts())
satisfaction_encoded = pd.get_dummies(satisfaction_binned, prefix="sat", drop_first=True, dtype=int)
print("Discretized & Encoded Satisfaction DataFrame:")
print(satisfaction_encoded.head(4))
print("_" * 60)

# Step 12 : Dimensionality Reduction & Automated Feature Selection
print("Step 12: Dimensionality Reduction & Automated Feature Selection")
# Assemble all processed and engineered feature components into final matrix
X_final = pd.concat([
    X_scaled_df,
    dept_encoded,
    postal_encoded_df,
    date_features,
    satisfaction_encoded,
    X_num[["salary_is_missing"]]
], axis=1)

print("Final Rebuilt Feature Matrix (Shape: {}):".format(X_final.shape))
print(X_final.head(4))
print("_" * 60)

# 12.A: Automated Feature Selection via SelectKBest (ANOVA F-statistic)
k_best = 4
selector = SelectKBest(score_func=f_classif, k=k_best)
X_selected = selector.fit_transform(X_final, Y)
selected_columns = X_final.columns[selector.get_support()].tolist()
print(f"Top {k_best} Features Selected by SelectKBest:")
print(selected_columns)
print("_" * 60)

# 12.B: Dimensionality Reduction via PCA (Principal Component Analysis)
n_pca = 2
pca = PCA(n_components=n_pca)
X_pca = pca.fit_transform(X_final)
print(f"PCA Reduced Features (Shape: {X_pca.shape}):")
print("Explained Variance Ratio by 2 Components:", np.round(pca.explained_variance_ratio_, 4))
print(f"Total Variance Retained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
print("_" * 60)

# Step 13 : Split Data into Training and Testing Sets & Evaluate Prediction (Following O01 Pattern)
print("Step 13: Model Training, Prediction & Evaluation (Following O01 Pattern)")
X_train, X_test, y_train, y_test = train_test_split(
    X_final[selected_columns], Y, test_size=0.3, random_state=42, stratify=Y
)

print("Training Features (X_train):")
print(X_train)
print("Testing Features (X_test):")
print(X_test)
print("Training Labels (y_train):")
print(y_train)
print("Testing Labels (y_test):")
print(y_test)
print("_" * 60)

# Fit Logistic Regression on the engineered and selected features
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model on test data
accuracy = model.score(X_test, y_test)
print("Model Accuracy on Test Data:", accuracy)

# Predict labels for test set
y_pred = model.predict(X_test)
print("Predicted Labels for Test Data:")
print(y_pred)
print("Actual Labels for Test Data:")
print(y_test.values)
print("_" * 60)

# Performance metrics (Accuracy Score & Classification Report)
print(f"Accuracy Score : {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))