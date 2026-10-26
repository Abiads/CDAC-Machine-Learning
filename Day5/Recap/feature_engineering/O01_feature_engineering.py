import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# predict Exam success
# Features: hours_studied, has_pencil, minutes_studied, breakfast_eaten
# Target: passed_exam

# 1. hours_studied --> useful! Driectly impacts exam success
# 2. has_pencil --> (0 variance) not useful for prediction (type of feature : quasi-constant)
# 3. minutes_studied ---> redundant (correlated with hours_studied) (100% correlation)
# if minutes_studied is considered , it would be overfitting the model, and introduce multicollinearity, so it should be dropped.
# Does it introduce VIF (Variance Inflation Factor) --> Yes, because it is perfectly correlated with hours_studied
# 4. breakfast_eaten ---> Binary Feature (0 = not eaten or 1 = eaten).
# should it be dropped? --> Not necessarily, it could be useful for prediction depending on the context
# 5. passed_exam --> Target variable (Y), should not be used as a feature for prediction 
data = {
    "hours_studied": [1, 2, 5, 6, 8, 9, 10, 12, 14, 15],
    "has_pencil": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
    "minutes_studied": [60, 120, 300, 360, 480, 540, 600, 720, 840, 900],
    "breakfast_eaten": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    "passed_exam": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)
print(df)
print("_" * 60)
# step 1 Separate Predictors and Target
# X --> Predictors (Features)
# y --> Target (Label)

# Separte Predictors (X) and Target (y)
X = df[["hours_studied", "has_pencil", "minutes_studied", "breakfast_eaten"]]
y = df["passed_exam"]

print("Original Features (X):")
print(X)    
print("_" * 60)

# step 2 Filter method 1 --> Remove quasi-constant features
# quasi are features that very rarely change their value (low variance)
# we can remove them using a threshold for variance (e.g. features > 85% of identical values)
# formula for binary threshold: variance = p * (1 - p) : where p is the proportion of ones in the feature
# if 85% of the values are identical, the variance threshold would be:
# threshold = 0.85 * (1 - 0.85) = 0.1275

variance_selector = VarianceThreshold(threshold=0.1275)
X_variance_filtered = variance_selector.fit_transform(X)
print("Features after Variance Threshold Filtering:")
print(X_variance_filtered)
print("_" * 60)
# retrieve the selected feature names after variance threshold filtering
surviving_features = X.columns[variance_selector.get_support()]
print("Surviving Features after Variance Threshold Filtering:")
print(surviving_features)
dropped_features = X.columns[~variance_selector.get_support()]
print("Dropped Features after Variance Threshold Filtering:")
print(dropped_features)
print("_" * 60)

# Rebuild DataFrame with surviving features
X_step1_df = pd.DataFrame(X_variance_filtered, columns=surviving_features)
print("Rebuilt DataFrame with Surviving Features:")
print(X_step1_df)

# Step 3 : Remove correlated features (multicollinearity) : Removing redundancy
corr_matrix = X_step1_df.corr().abs()
print("Correlation Matrix:")
print(corr_matrix)
print("_" * 60)
threshold = 0.85
fetures_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > threshold:
            fetures_to_drop.add(corr_matrix.columns[i])

print("Features to Drop due to High Correlation:")
print(fetures_to_drop)
print("_" * 60)

# Drop the redundant columns due to high correlation
X_final = X_step1_df.drop(columns=fetures_to_drop)
print("Final Features after Removing Highly Correlated Features:")
print(X_final)
print("_" * 60)




