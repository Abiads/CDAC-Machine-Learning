"""
O04_cardinality_dimensionality_reducer.py
Groups rare categorical levels into 'Other' and performs PCA/MCA dimensionality reduction.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CardinalityReducer")

class CardinalityDimensionalityReducer:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.frequent_categories = {}
        self.pca_transformer = None

    def _load_config(self) -> dict:
        """Loads cardinality reduction parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("cardinality_reduction", {})

    def group_rare_categories(self, df: pd.DataFrame, categorical_cols: list = None, is_train: bool = True) -> pd.DataFrame:
        """Groups low-frequency categories (< rare_label_threshold) into an 'Other' category."""
        df_clean = df.copy()
        exclude_patterns = ["id", "uuid", "key"]
        cols = categorical_cols or [
            c for c in df_clean.select_dtypes(include=["object", "category"]).columns
            if not any(pat in c.lower() for pat in exclude_patterns)
        ]
        rare_thresh = self.config.get("rare_label_threshold", 0.02)

        for col in cols:
            if col not in df_clean.columns:
                continue
            if is_train or col not in self.frequent_categories:
                freq = df_clean[col].value_counts(normalize=True)
                frequent = freq[freq >= rare_thresh].index.tolist()
                self.frequent_categories[col] = frequent

            frequent = self.frequent_categories.get(col, [])
            df_clean[col] = df_clean[col].apply(lambda x: x if x in frequent else "Other")
            logger.info(f"Grouped rare levels for '{col}'. Retained {len(frequent)} frequent categories.")

        return df_clean

    def apply_pca_reduction(self, df: pd.DataFrame, feature_cols: list = None, variance_ratio: float = 0.95, is_train: bool = True) -> pd.DataFrame:
        """Performs PCA dimensionality reduction on numeric feature space."""
        df_clean = df.copy()
        cols = feature_cols or [c for c in df_clean.select_dtypes(include=[np.number]).columns]
        if not cols or len(cols) < 2:
            return df_clean

        if is_train or self.pca_transformer is None:
            self.pca_transformer = PCA(n_components=variance_ratio, random_state=42)
            reduced_data = self.pca_transformer.fit_transform(df_clean[cols])
            pca_cols = [f"pca_component_{i+1}" for i in range(reduced_data.shape[1])]
            self.pca_feature_names = pca_cols
        else:
            reduced_data = self.pca_transformer.transform(df_clean[cols])
            pca_cols = self.pca_feature_names

        pca_df = pd.DataFrame(reduced_data, columns=pca_cols, index=df_clean.index)
        df_clean = df_clean.drop(columns=cols)
        df_clean = pd.concat([df_clean, pca_df], axis=1)
        logger.info(f"PCA reduced {len(cols)} features to {len(pca_cols)} components (variance={variance_ratio})")
        return df_clean

if __name__ == "__main__":
    pass
    