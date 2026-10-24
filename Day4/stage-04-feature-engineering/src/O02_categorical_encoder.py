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
        """Applies One-Hot Encoding to configured low-cardinality nominal columns."""
        df_clean = df.copy()
        cols = [c for c in self.config.get("low_cardinality_columns", []) if c in df_clean.columns]
        if not cols:
            logger.info("No low-cardinality columns found to encode.")
            return df_clean

        drop_first = self.config.get("drop_first", True)
        if is_train or self.ohe_encoder is None:
            self.ohe_cols = cols
            self.ohe_encoder = OneHotEncoder(
                drop="first" if drop_first else None,
                sparse_output=False,
                handle_unknown="ignore"
            )
            encoded_matrix = self.ohe_encoder.fit_transform(df_clean[cols])
            feature_names = self.ohe_encoder.get_feature_names_out(cols)
            self.ohe_feature_names = feature_names
        else:
            encoded_matrix = self.ohe_encoder.transform(df_clean[self.ohe_cols])
            feature_names = self.ohe_feature_names

        encoded_df = pd.DataFrame(encoded_matrix, columns=feature_names, index=df_clean.index)
        df_clean = df_clean.drop(columns=cols)
        df_clean = pd.concat([df_clean, encoded_df], axis=1)
        logger.info(f"OHE applied to {cols}. Generated {len(feature_names)} encoded columns.")
        return df_clean

    def fit_transform_target_encoding(self, df: pd.DataFrame, target_col: str, is_train: bool = True) -> pd.DataFrame:
        """Applies smoothed target encoding for high-cardinality categorical variables."""
        df_clean = df.copy()
        cols = [c for c in self.config.get("high_cardinality_columns", []) if c in df_clean.columns]
        if not cols:
            return df_clean

        smoothing = self.config.get("target_encoding", {}).get("smoothing", 10.0)
        global_mean = df_clean[target_col].mean() if is_train else self.target_encoding_maps.get("__global_mean__", 0.0)

        if is_train:
            self.target_encoding_maps["__global_mean__"] = global_mean
            for col in cols:
                stats = df_clean.groupby(col)[target_col].agg(["count", "mean"])
                # Empirical Bayes / m-estimate smoothing
                smooth_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
                self.target_encoding_maps[col] = smooth_map.to_dict()

        for col in cols:
            mapping = self.target_encoding_maps.get(col, {})
            encoded_col = f"{col}_target_encoded"
            df_clean[encoded_col] = df_clean[col].map(mapping).fillna(global_mean)
            df_clean = df_clean.drop(columns=[col])
            logger.info(f"Target encoding applied to '{col}' -> '{encoded_col}'")

        return df_clean

if __name__ == "__main__":
    pass