"""
O01_domain_feature_builder.py
Builds domain-specific interaction terms, ratios, and usage intensity indicators.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DomainFeatureBuilder")

class DomainFeatureBuilder:
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Loads domain feature engineering rules from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("domain_features", {})

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs domain features and ratio metrics specified in configuration."""
        df_clean = df.copy()
        ratios = self.config.get("ratios_to_build", [])
        for rule in ratios:
            num = rule.get("numerator")
            den = rule.get("denominator")
            new_col = rule.get("new_feature_name")
            if num in df_clean.columns and den in df_clean.columns:
                # Replace 0 denominators with NaN to avoid ZeroDivisionError, fillna with 0
                df_clean[new_col] = (df_clean[num] / df_clean[den].replace(0, np.nan)).fillna(0.0)
                logger.info(f"Engineered ratio feature '{new_col}' = {num} / {den}")
            else:
                logger.warning(f"Skipping ratio '{new_col}': columns '{num}' or '{den}' not present.")
        return df_clean

if __name__ == "__main__":
    pass