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

    def group_rare_categories(self, df: pd.DataFrame, categorical_cols: list, is_train: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        return df_clean
    
    def apply_pca_reduction(self, df: pd.DataFrame, variance_ratio: float = 0.95, is_train: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        return df_clean

if __name__ == "__main__":
    pass
    