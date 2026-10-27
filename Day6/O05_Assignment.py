# Assignment 2: Multi-Feature Tech Salary Prediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


# 1. Synthetic Dataset Generation
np.random.seed(102)
n_tech_samples = 100

years_experience = np.random.uniform(1, 12, size=n_tech_samples)
projects_completed = np.random.poisson(lam=5, size=n_tech_samples)
code_review_score = np.random.uniform(3.5, 5.0, size=n_tech_samples)

# Target: Salary = Base (40k) + 10k*(exp) + 3k*(projects) + 15k*(review_score) + Noise
annual_salary = (
    40000 
    + 10000 * years_experience 
    + 3000 * projects_completed 
    + 15000 * code_review_score 
    + np.random.normal(0, 4000, size=n_tech_samples)
)

df_tech = pd.DataFrame({
    "experience_years": years_experience.round(1),
    "projects_completed": projects_completed,
    "code_review_score": code_review_score.round(2),
    "annual_salary": annual_salary.round(2)
})