"""
O05_feature_selector.py
Multistage feature selection using variance thresholds, correlation filters, and Mutual Information.
"""


import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("FeatureSelector")

class FeatureSelector:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.variance_threshold = None
        self.selected_features = None

    def _load_config(self) -> dict:
        """Loads feature selection parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("feature_selection", {})

    def apply_variance_threshold(self, df: pd.DataFrame, exclude_cols: list = None, is_train: bool = True) -> pd.DataFrame:
        """Removes near-zero-variance features."""
        df_clean = df.copy()
        vt_cfg = self.config.get("variance_threshold", {})
        if not vt_cfg.get("enabled", True):
            return df_clean

        thresh = vt_cfg.get("threshold", 0.01)
        exclude = set(exclude_cols or [])
        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in exclude]

        if not numeric_cols:
            return df_clean

        if is_train or self.variance_threshold is None:
            self.variance_threshold = VarianceThreshold(threshold=thresh)
            self.variance_threshold.fit(df_clean[numeric_cols])
            selected_mask = self.variance_threshold.get_support()
            self.vt_retained_cols = [col for col, keep in zip(numeric_cols, selected_mask) if keep]
            dropped_cols = [col for col, keep in zip(numeric_cols, selected_mask) if not keep]
            logger.info(f"Variance threshold ({thresh}): retained {len(self.vt_retained_cols)} columns, dropped {dropped_cols}")

        kept_cols = [c for c in df_clean.columns if c not in numeric_cols or c in self.vt_retained_cols]
        return df_clean[kept_cols]

    def apply_correlation_filter(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """Drops one feature from collinear pairs exceeding the correlation threshold."""
        df_clean = df.copy()
        corr_cfg = self.config.get("correlation_filter", {})
        if not corr_cfg.get("enabled", True):
            return df_clean

        thresh = corr_cfg.get("threshold", 0.85)
        exclude = set(exclude_cols or [])
        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in exclude]

        if len(numeric_cols) < 2:
            return df_clean

        corr_matrix = df_clean[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > thresh)]

        if to_drop:
            df_clean = df_clean.drop(columns=to_drop)
            logger.info(f"Correlation filter ({thresh}): dropped redundant collinear features: {to_drop}")
        return df_clean

    def apply_statistical_scoring(self, df: pd.DataFrame, target_col: str, exclude_cols: list = None, is_train: bool = True) -> pd.DataFrame:
        """Selects top-K features using Mutual Information or Chi-Square scoring."""
        df_clean = df.copy()
        scoring_cfg = self.config.get("statistical_scoring", {})
        top_k = scoring_cfg.get("top_k", 15)
        exclude = set(exclude_cols or [])
        exclude.add(target_col)

        feature_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in exclude]

        if not feature_cols:
            return df_clean

        actual_k = min(top_k, len(feature_cols))

        if is_train or self.selected_features is None:
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            selector = SelectKBest(score_func=mutual_info_classif, k=actual_k)
            selector.fit(X, y)
            support = selector.get_support()
            self.selected_features = [col for col, keep in zip(feature_cols, support) if keep]
            logger.info(f"Mutual Information selection retained top {len(self.selected_features)} features: {self.selected_features}")

        retained_cols = [c for c in df_clean.columns if c not in feature_cols or c in self.selected_features]
        return df_clean[retained_cols]

if __name__ == "__main__":
    pass

