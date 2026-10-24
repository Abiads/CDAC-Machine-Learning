"""
O02_categorical_encoder.py
Handles low and high cardinality categorical variables via One-Hot and out-of-fold Target Encoding.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CategoricalEncoder")

class CategoricalEncoder:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ohe_encoder = None
        self.target_encoding_maps = {}

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("categorical_encoding", {})

    def fit_transform_low_cardinality(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        return df_clean

    def fit_transform_target_encoding(self, df: pd.DataFrame, target_col: str, is_train: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        return df_clean

if __name__ == "__main__":
    pass