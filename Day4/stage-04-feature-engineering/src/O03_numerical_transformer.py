"""
O03_numerical_transformer.py
Applies power transformations (Yeo-Johnson/Box-Cox) and scaling (Standard/Robust).
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("NumericalTransformer")

class NumericalTransformer:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.power_transformer = None
        self.scaler = None
        self.transformed_cols = []

    def _load_config(self) -> dict:
        """Loads numerical transformation parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("numerical_transformation", {})

    def fit_transform_features(self, df: pd.DataFrame, exclude_cols: list = None, is_train: bool = True) -> pd.DataFrame:
        """Applies Yeo-Johnson power transformation and robust scaling to numeric columns."""
        df_clean = df.copy()
        exclude = set(exclude_cols or [])
        # Only transform continuous numerical columns (exclude binary 0/1 indicator flags)
        num_cols = [
            c for c in df_clean.select_dtypes(include=[np.number]).columns 
            if c not in exclude and (df_clean[c].nunique() > 2 if is_train else True)
        ]

        if not num_cols:
            return df_clean

        power_cfg = self.config.get("power_transform", {})
        scaling_cfg = self.config.get("scaling", {})

        if is_train or not self.transformed_cols:
            self.transformed_cols = num_cols
            if power_cfg.get("enabled", True):
                method = power_cfg.get("method", "yeo-johnson")
                self.power_transformer = PowerTransformer(method=method, standardize=False)
                df_clean[num_cols] = self.power_transformer.fit_transform(df_clean[num_cols])
                logger.info(f"Applied {method} power transform to {num_cols}")

            scale_method = scaling_cfg.get("method", "robust")
            self.scaler = RobustScaler() if scale_method == "robust" else StandardScaler()
            df_clean[num_cols] = self.scaler.fit_transform(df_clean[num_cols])
            logger.info(f"Applied {scale_method} scaler to {num_cols}")
        else:
            cols = [c for c in self.transformed_cols if c in df_clean.columns]
            if self.power_transformer:
                df_clean[cols] = self.power_transformer.transform(df_clean[cols])
            if self.scaler:
                df_clean[cols] = self.scaler.transform(df_clean[cols])
            logger.info(f"Applied fitted transforms to test set for {cols}")

        return df_clean

if __name__ == "__main__":
    pass