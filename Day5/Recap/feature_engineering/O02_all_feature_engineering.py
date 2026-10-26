'''
1. Zero-Variance Feature Removal
2. Quasi-Constant Feature Removal
3. Highly Correlated & Redundant Feature Removal
4. Missing Value Imputation & Missingness Indicators
5. Outlier Treatment
6. Categorical Encoding & Rare Category Handling
7. High-Cardinality Feature Management
8. Mathematical & Power Transformations
9. Feature Scaling & Normalization
10. Date-Time & Temporal Decomposition
11. Binning & Discretization
12. Dimensionality Reduction & Automated Feature Selection
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
# print(df)
# print("_"*60)
# print("DataFrame Info:")
# print(df.info())
# print("_"*60)
# print("DataFrame Description:")
# print(df.describe(include='all').T)
# print("_"*60)
# print("_"*60)
# print("Checking for Missing Values:")
# print(df.isnull().sum())
print("="*60)
print("All Columns in predictor list : ", X.columns.tolist())

# 1 Zero-Variance Feature Removal
print("_"*60)
print("Step 1: Zero-Variance Feature Removal")
vt_zero = VarianceThreshold(threshold=0)
X_num = X.select_dtypes(include=[np.number])
print("Numeric Columns Before Zero-Variance Features Removed: ", X_num.columns.tolist())
vt_zero.fit(X_num)
X_num = X_num.loc[:, vt_zero.get_support()]
print("Numeric Columns After Zero-Variance Features Removed: ", X_num.columns.tolist())

# 2. Quasi-Constant Feature Removal
print("_"*60)
print("Step 2: Quasi-Constant Feature Removal")
quasi_threshold = 0.06 # Binary threshold for quasi-constant features p * (1-p) for 94% threshold
vt_quasi = VarianceThreshold(threshold=quasi_threshold)
vt_quasi.fit(X_num)
X_num = X_num.loc[:, vt_quasi.get_support()]
print("Numeric Columns After Quasi-Constant Features Removed: ", X_num.columns.tolist())

# 3.  Highly Correlated & Redundant Removal
print("_"*60)
print("Step 3: Highly Correlated & Redundant Feature Removal")
corr_threshold = 0.9
# corr_matrix = X_num.corr().abs()
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
# X_num = X_num.drop(columns=to_drop)

corr = X_num.corr().abs()
# print("Correlation Matrix: ")
# print(corr)
to_drop = set()
for i in range(len(X_num.columns)):
    for j in range(i):
        if corr.iloc[i, j] > corr_threshold:
            to_drop.add(X_num.columns[i])
print("Highly Correlated Features to Drop:", list(to_drop))
X_num = X_num.drop(columns=list(to_drop))
print("Numeric Columns After Highly Correlated Features Removed: ", X_num.columns.tolist())

# 4. Missing Value Imputation & Indicators
print("_"*60)
print("Step 4: Missing Value Imputation & Indicators")
# Create a binary missingness indicator column (1 if missing, 0 if present)
X_num["salary_is_missing"] = X_num["salary"].isna().astype(int)

# Median imputation for continuous numerical feature
salary_median = X_num["salary"].median()
X_num["salary"] = X_num["salary"].fillna(salary_median)
print("Salary median used for imputation:", salary_median)
print("Missing values in salary after imputation:", X_num["salary"].isna().sum())

# 5. Outlier Treatment (Winsorization / IQR Capping)
print("_"*60)
print("Step 5: Outlier Treatment (IQR Capping / Winsorization)")
# Calculate IQR for salary
Q1 = X_num["salary"].quantile(0.25)
Q3 = X_num["salary"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"Salary bounds: Lower = {lower_limit}, Upper = {upper_limit}")
print("Salary before outlier treatment (Max):", X_num["salary"].max())

# Clip / Winsorize salary values within the [lower_limit, upper_limit] bounds
X_num["salary"] = X_num["salary"].clip(lower=lower_limit, upper=upper_limit)
print("Salary after outlier capping (Max):", X_num["salary"].max())

# 6. Categorical Encoding & Rare Categories
print("_"*60)
print("Step 6: Categorical Encoding & Rare Category Handling")
dept = X["department"].copy()
dept_freq = dept.value_counts(normalize=True)
print("Department Frequency Distribution:\n", dept_freq)

# Identify categories appearing in 5% or fewer records and replace with 'Other'
rare_cats = dept_freq[dept_freq <= 0.05].index
dept = dept.replace(rare_cats, "Other")
print("Unique departments after rare category grouping:", dept.unique().tolist())

# One-Hot Encoding (drop_first=True to avoid dummy variable trap / multicollinearity)
dept_encoded = pd.get_dummies(dept, prefix="dept", drop_first=True, dtype=int)
print("One-Hot Encoded Department Columns:")
print(dept_encoded.head(3))

# 7. High-Cardinality Feature Management
print("_"*60)
print("Step 7: High-Cardinality Feature Management")
# postal_code has 20 unique values (high cardinality relative to sample size)
# Using Target Encoding with smoothing:
# formula: (count * mean + smoothing * global_mean) / (count + smoothing)
smoothing = 5.0
global_mean = Y.mean()
postal_stats = pd.DataFrame({"postal_code": X["postal_code"], "target": Y})
stats = postal_stats.groupby("postal_code")["target"].agg(["count", "mean"])
smoothed_target = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)

postal_encoded = X["postal_code"].map(smoothed_target).fillna(global_mean)
postal_encoded = pd.DataFrame({"postal_code_encoded": postal_encoded})
print("Smoothed Target Encoded Postal Code Sample:")
print(postal_encoded.head(3))

# 8. Mathematical & Power Transformations
print("_"*60)
print("Step 8: Mathematical & Power Transformations (Yeo-Johnson)")
# bonus_amount is right-skewed; apply Yeo-Johnson PowerTransformer to normalize
pt = PowerTransformer(method="yeo-johnson")
skew_before = X_num["bonus_amount"].skew()
X_num["bonus_transformed"] = pt.fit_transform(X_num[["bonus_amount"]])
skew_after = X_num["bonus_transformed"].skew()
print(f"Bonus Amount Skewness: Before = {skew_before:.2f}, After = {skew_after:.2f}")

# 9. Feature Scaling & Normalization
print("_"*60)
print("Step 9: Feature Scaling & Normalization")
# Scale numerical features using StandardScaler (mean=0, variance=1)
scaler = StandardScaler()
num_cols_to_scale = ["hours_worked", "salary", "bonus_transformed"]
scaled_array = scaler.fit_transform(X_num[num_cols_to_scale])
X_scaled = pd.DataFrame(
    scaled_array,
    columns=[f"{col}_scaled" for col in num_cols_to_scale],
    index=X.index
)
print("Scaled Numerical Features Sample:")
print(X_scaled.head(3))

# 10. Date-Time & Temporal Decomposition
print("_"*60)
print("Step 10: Date-Time & Temporal Decomposition")
dates = pd.to_datetime(X["join_date"])
# Extract year, month, and day of week
join_year = dates.dt.year
join_month = dates.dt.month
join_dayofweek = dates.dt.dayofweek

# Compute tenure in years from reference date (2026-01-01)
ref_date = pd.to_datetime("2026-01-01")
tenure_years = ((ref_date - dates).dt.days / 365.25).round(2)

date_features = pd.DataFrame({
    "join_year": join_year,
    "join_month": join_month,
    "join_dayofweek": join_dayofweek,
    "tenure_years": tenure_years
}, index=X.index)
print("Extracted Temporal Features:")
print(date_features.head(3))

# 11. Binning & Discretization
print("_"*60)
print("Step 11: Binning & Discretization")
# Convert continuous satisfaction_score into Low, Medium, High categories
satisfaction_binned = pd.cut(
    X["satisfaction_score"],
    bins=[0, 3.5, 4.5, 5.0],
    labels=["Low", "Medium", "High"]
)
print("Binned Satisfaction Distribution:\n", satisfaction_binned.value_counts())
# One-hot encode the binned satisfaction levels
satisfaction_encoded = pd.get_dummies(satisfaction_binned, prefix="sat", drop_first=True, dtype=int)
print("Satisfaction Binned & Encoded Sample:")
print(satisfaction_encoded.head(3))

# 12. Dimensionality Reduction & Automated Feature Selection
print("_"*60)
print("Step 12: Dimensionality Reduction & Automated Feature Selection")
# Combine all processed features into one final feature matrix
final_features = pd.concat([
    X_scaled,
    dept_encoded,
    postal_encoded,
    date_features,
    satisfaction_encoded,
    X_num[["salary_is_missing"]]
], axis=1)

print("Final Combined Feature Matrix Shape:", final_features.shape)
print("Features List:", final_features.columns.tolist())

# 12.A: Automated Feature Selection using SelectKBest (ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(final_features, Y)
selected_mask = selector.get_support()
selected_columns = final_features.columns[selected_mask].tolist()
print("\nTop 4 Features Selected by SelectKBest:")
print(selected_columns)

# 12.B: Dimensionality Reduction using PCA (Principal Component Analysis)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(final_features)
print("\nPCA Reduced Features (Shape: {}):".format(X_pca.shape))
print(f"Explained Variance Ratio by 2 Components: {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
print("_"*60)
print("All 12 Feature Engineering Steps Completed Successfully!")