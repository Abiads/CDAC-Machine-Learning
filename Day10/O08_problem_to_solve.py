'''
Day 10: Item-Based Collaborative Filtering (Movie Recommendation System)
========================================================================
Dataset: MovieLens 100k (u.data & u.item)
Goal: Recommend movies similar to 'Star Wars (1977)' using user rating correlations.
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Robust path handling for running from workspace root or Day10 directory
base_dir = Path(__file__).parent if "__file__" in locals() else Path(".")
data_path = base_dir / "assets" / "u.data"
item_path = base_dir / "assets" / "u.item"

if not data_path.exists():
    data_path = Path("./assets/u.data")
    item_path = Path("./assets/u.item")

# 1. Load Ratings
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv(str(data_path), sep="\t", usecols=range(3), names=r_cols, encoding='ISO-8859-1')
print("Ratings head:")
print(ratings.head())
print("=" * 60)

# 2. Load Movies
m_cols = ['movie_id', 'title']
movies = pd.read_csv(str(item_path), sep="|", usecols=range(2), names=m_cols, encoding='ISO-8859-1')
print("Movies head:")
print(movies.head())
print("Shape of movies:", movies.shape)
print("=" * 60)

# 3. Merge Movies and Ratings
ratings = pd.merge(movies, ratings)
print("Merged ratings head:")
print(ratings.head())
print("=" * 60)

# 4. Create User-Item Pivot Table
moviesRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
print("Movie Ratings Pivot Table head:")
print(moviesRatings.head())
print("=" * 60)

# 5. Extract ratings for Target Movie: 'Star Wars (1977)'
myMovie = moviesRatings['Star Wars (1977)']
print("Star Wars ratings sample:")
print(myMovie.head())
print("=" * 60)

# 6. Compute correlation of all movies with Star Wars
similarMovies = moviesRatings.corrwith(myMovie)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies, columns=['similarity'])
print("Raw Correlations head:")
print(df.head())
print("=" * 60)

print("Top Raw Correlations (contains outliers/single-viewer ratings):")
print(similarMovies.sort_values(ascending=False).head(10))
print("=" * 60)

# 7. Aggregate movie statistics: rating count (size) and average rating (mean)
movieStats = ratings.groupby('title').agg({'rating': ['count', 'mean']})
movieStats.columns = ['size', 'mean']
print("Movie Stats head:")
print(movieStats.head())
print("=" * 60)

# 8. Filter popular movies with size >= 100
popularMovies = movieStats['size'] >= 100
print("Top 15 Highest Rated Popular Movies (>= 100 ratings):")
print(movieStats[popularMovies].sort_values(by='mean', ascending=False)[:15])
print("=" * 60)

# 9. Solve Problem: Join correlations with rating counts to filter out spurious outliers
df = df.join(movieStats)
recommendations = df[df['size'] >= 100].sort_values(by='similarity', ascending=False)
print("=" * 60)
print("FINAL RECOMMENDATIONS FOR 'Star Wars (1977)' (Movies with >= 100 ratings):")
print("=" * 60)
print(recommendations.head(10))
print("=" * 60)
