'''
Comprehensive Feature Engineering Tutorial (12 Core Steps)
1. Zero-Variance Feature Removal
2. Quasi-Constant Feature Removal
3. Highly Correlated & Redundant Feature Removal
4. Missing Value Imputation & Missingness Indicators
5. Outlier Treatment (Winsorization / IQR Capping)
6. Categorical Encoding & Rare Category Handling
7. High-Cardinality Feature Management (Target Encoding)
8. Mathematical & Power Transformations (Yeo-Johnson)
9. Feature Scaling & Normalization (StandardScaler)
10. Date-Time & Temporal Decomposition
11. Binning & Discretization
12. Dimensionality Reduction (PCA) & Automated Feature Selection (SelectKBest)
'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA

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
X = df.drop(columns=["promoted"])
Y = df["promoted"]

print("=" * 60)
print("Initial Predictor Columns:", X.columns.tolist())
print("=" * 60)

# 1. Zero-Variance Feature Removal
vt_zero = VarianceThreshold(threshold=0)
X_num = X.select_dtypes(include=[np.number])
vt_zero.fit(X_num)
X_num = X_num.loc[:, vt_zero.get_support()]
print("1. After Zero-Variance Removal:", X_num.columns.tolist())

# 2. Quasi-Constant Feature Removal
quasi_threshold = 0.06 # Binary threshold: p * (1 - p) for 94% constant
vt_quasi = VarianceThreshold(threshold=quasi_threshold)
vt_quasi.fit(X_num)
X_num = X_num.loc[:, vt_quasi.get_support()]
print("2. After Quasi-Constant Removal:", X_num.columns.tolist())

# 3. Highly Correlated & Redundant Feature Removal
corr = X_num.corr().abs()
to_drop = set()
for i in range(len(X_num.columns)):
    for j in range(i):
        if corr.iloc[i, j] > 0.90:
            to_drop.add(X_num.columns[i])
X_num = X_num.drop(columns=list(to_drop))
print("3. After Correlation Pruning (>0.90):", X_num.columns.tolist())

# 4. Missing Value Imputation & Missingness Indicators
X_num["salary_is_missing"] = X_num["salary"].isna().astype(int)
X_num["salary"] = X_num["salary"].fillna(X_num["salary"].median())
print(f"4. Missing Imputed: salary filled with median | Nulls left: {X_num['salary'].isna().sum()}")

# 5. Outlier Treatment (Winsorization via IQR Capping)
q1 = X_num["salary"].quantile(0.25)
q3 = X_num["salary"].quantile(0.75)
iqr = q3 - q1
X_num["salary"] = X_num["salary"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
print(f"5. Outlier Capped: salary clipped between [{q1 - 1.5 * iqr:.0f}, {q3 + 1.5 * iqr:.0f}] | Max: {X_num['salary'].max():.0f}")

# 6. Categorical Encoding & Rare Category Handling
dept_freq = X["department"].value_counts(normalize=True)
rare_cats = dept_freq[dept_freq <= 0.05].index
dept_cleaned = X["department"].replace(rare_cats, "Other")
dept_encoded = pd.get_dummies(dept_cleaned, prefix="dept", drop_first=True, dtype=int)
print("6. One-Hot Encoded Department:", dept_encoded.columns.tolist())

# 7. High-Cardinality Feature Management (Target Encoding)
smoothing = 5.0
prior = Y.mean()
stats = pd.DataFrame({"code": X["postal_code"], "target": Y}).groupby("code")["target"].agg(["count", "mean"])
smoothed_target = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
postal_encoded = X["postal_code"].map(smoothed_target).fillna(prior).to_frame(name="postal_code_encoded")
print("7. Target Encoded postal_code (smoothed)")

# 8. Mathematical & Power Transformations (Yeo-Johnson)
pt = PowerTransformer(method="yeo-johnson")
X_num["bonus_transformed"] = pt.fit_transform(X_num[["bonus_amount"]])
X_num = X_num.drop(columns=["bonus_amount"])
print("8. Power Transformed bonus_amount (Yeo-Johnson)")

# 9. Feature Scaling & Normalization (StandardScaler)
scaler = StandardScaler()
scale_cols = ["hours_worked", "salary", "bonus_transformed"]
X_scaled = pd.DataFrame(scaler.fit_transform(X_num[scale_cols]), columns=[f"{c}_scaled" for c in scale_cols])
print("9. StandardScaler Applied to:", [f"{c}_scaled" for c in scale_cols])

# 10. Date-Time & Temporal Decomposition
dates = pd.to_datetime(X["join_date"])
date_features = pd.DataFrame({
    "join_year": dates.dt.year,
    "join_month": dates.dt.month,
    "join_dayofweek": dates.dt.dayofweek,
    "tenure_years": ((pd.to_datetime("2026-01-01") - dates).dt.days / 365.25).round(2)
})
print("10. Date-Time Extracted:", date_features.columns.tolist())

# 11. Binning & Discretization
sat_binned = pd.cut(X["satisfaction_score"], bins=[0, 3.5, 4.5, 5.0], labels=["Low", "Medium", "High"])
sat_encoded = pd.get_dummies(sat_binned, prefix="sat", drop_first=True, dtype=int)
print("11. Satisfaction Binned & Encoded:", sat_encoded.columns.tolist())

# 12. Dimensionality Reduction & Automated Feature Selection
X_final = pd.concat([
    X_scaled,
    dept_encoded,
    postal_encoded,
    date_features,
    sat_encoded,
    X_num[["salary_is_missing"]]
], axis=1)

selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(X_final, Y)
print("12. Top 4 Features Selected (SelectKBest):", X_final.columns[selector.get_support()].tolist())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_final)
print("    PCA 2 Components Variance Retained: {:.2f}%".format(pca.explained_variance_ratio_.sum() * 100))
print("=" * 60)
print(f"Final Feature Matrix Shape: {X_final.shape}")
print("All 12 Feature Engineering Steps Completed Successfully!")