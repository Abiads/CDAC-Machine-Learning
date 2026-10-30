import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Create a DataFrame for the sample data points
data = {
    'Sample_ID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'Feature_A': [1, 1, 1, 4, 4, 4],
    'Feature_B': [2, 4, 0, 2, 4, 0]
}
df = pd.DataFrame(data)
print(df)