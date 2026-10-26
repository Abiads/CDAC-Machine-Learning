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
print(df)