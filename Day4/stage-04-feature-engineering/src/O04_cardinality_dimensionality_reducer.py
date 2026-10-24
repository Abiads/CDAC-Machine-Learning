"""
================================================================================
MODULE: Cardinality Reduction & Dimensionality Reduction Engine
FILE: src/O04_cardinality_dimensionality_reducer.py
PURPOSE:
  Mitigates the 'Curse of Dimensionality' and avoids sparse, over-parameterized
  feature spaces through two complementary strategies:
  
  1. Rare-Label Grouping (Frequency Thresholding):
     - Consolidates long-tail, low-frequency categorical levels (< threshold)
       into a unified 'Other' bucket.
     - Prevents excessive feature explosion during subsequent One-Hot Encoding
       and stabilizes tree-based split criteria.
     - Strictly stores learned frequent categories on training data to prevent
       data leakage during evaluation.

  2. Principal Component Analysis (PCA):
     - Orthogonal linear projection of continuous feature space.
     - Retains components explaining a target cumulative variance (e.g., 95%).
     - Eliminates multicollinearity among dense numerical variables.

CONFIG DEPENDENCIES:
  - 'config/O01_feature_rules.yaml' -> section 'cardinality_reduction'
================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CardinalityReducer")


class CardinalityDimensionalityReducer:
    """
    Handles rare-category pooling and linear dimensionality reduction (PCA).
    Preserves state across train and test partitions to ensure reproducible transforms.
    """

    def __init__(self, config_path: Path):
        """
        Initializes the reducer by loading parameters from configuration.

        Args:
            config_path (Path): Path to 'config/O01_feature_rules.yaml'.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # State storage to prevent data leakage:
        # Maps column_name -> List of frequent category values learned from X_train
        self.frequent_categories: Dict[str, List[str]] = {}

        # Fitted PCA transformer instance from scikit-learn
        self.pca_transformer: Optional[PCA] = None
        self.pca_feature_names: List[str] = []

    def _load_config(self) -> dict:
        """
        Parses the YAML configuration file and extracts cardinality reduction rules.

        Returns:
            dict: Configuration dictionary under the 'cardinality_reduction' key.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("cardinality_reduction", {})

    def group_rare_categories(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Consolidates infrequent categories representing less than a configured percentage
        (e.g., < 2%) of total samples into a single 'Other' category.

        CRITICAL LEAKAGE PRECAUTION:
          - During training (is_train=True), frequent categories are calculated and stored.
          - During testing/inference (is_train=False), the previously learned frequent
            categories are reused; any novel or previously rare categories become 'Other'.

        Args:
            df (pd.DataFrame): Input dataset.
            categorical_cols (list, optional): Explicit list of columns to process.
                                               If None, auto-selects object/category columns.
            is_train (bool): Flag indicating whether to fit parameters or use frozen state.

        Returns:
            pd.DataFrame: Transformed DataFrame with rare levels replaced by 'Other'.
        """
        df_clean = df.copy()

        # Identifier columns (e.g., customer_id, uuid) must never be lumped into 'Other'
        exclude_patterns = ["id", "uuid", "key"]

        # Identify candidate categorical columns if not explicitly passed
        cols = categorical_cols or [
            c for c in df_clean.select_dtypes(include=["object", "category"]).columns
            if not any(pat in c.lower() for pat in exclude_patterns)
        ]

        # Read minimum frequency threshold from config (default: 0.02 = 2% of records)
        rare_thresh = self.config.get("rare_label_threshold", 0.02)

        for col in cols:
            if col not in df_clean.columns:
                continue

            # ------------------------------------------------------------------
            # Phase 1: Parameter Learning (Training Partition Only)
            # ------------------------------------------------------------------
            if is_train or col not in self.frequent_categories:
                # Calculate normalized value frequencies (proportions summing to 1.0)
                freq = df_clean[col].value_counts(normalize=True)

                # Filter categories meeting or exceeding the threshold
                frequent = freq[freq >= rare_thresh].index.tolist()

                # Persist learned frequent levels in reducer state
                self.frequent_categories[col] = frequent

            # ------------------------------------------------------------------
            # Phase 2: Category Transformation (Applied to Train & Test)
            # ------------------------------------------------------------------
            frequent = self.frequent_categories.get(col, [])

            # Replace any category not present in the frequent set with 'Other'
            df_clean[col] = df_clean[col].apply(lambda x: x if x in frequent else "Other")

            logger.info(
                f"Grouped rare levels for '{col}'. "
                f"Retained {len(frequent)} frequent categories (threshold >= {rare_thresh:.1%})."
            )

        return df_clean

    def apply_pca_reduction(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        variance_ratio: float = 0.95,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Performs Principal Component Analysis (PCA) to compress high-dimensional
        correlated numerical features into orthogonal principal components.

        Args:
            df (pd.DataFrame): Input dataset containing numerical columns.
            feature_cols (list, optional): Specific numerical columns to project.
                                           If None, selects all numerical columns.
            variance_ratio (float): Percentage of cumulative variance to preserve (default: 0.95).
            is_train (bool): If True, fits PCA on this data. If False, transforms using fitted state.

        Returns:
            pd.DataFrame: Transformed DataFrame with original numerical columns replaced
                          by orthogonal PCA components ('pca_component_1', 'pca_component_2', ...).
        """
        df_clean = df.copy()

        # Identify numerical columns for projection
        cols = feature_cols or [c for c in df_clean.select_dtypes(include=[np.number]).columns]

        # PCA requires at least 2 numerical features to perform dimensionality reduction
        if not cols or len(cols) < 2:
            logger.warning("PCA skipped: fewer than 2 numerical columns available.")
            return df_clean

        # ----------------------------------------------------------------------
        # Phase 1: Fit & Transform on Training Data
        # ----------------------------------------------------------------------
        if is_train or self.pca_transformer is None:
            # n_components as float represents desired cumulative explained variance
            self.pca_transformer = PCA(n_components=variance_ratio, random_state=42)
            reduced_data = self.pca_transformer.fit_transform(df_clean[cols])

            # Generate descriptive component names: ['pca_component_1', 'pca_component_2', ...]
            pca_cols = [f"pca_component_{i+1}" for i in range(reduced_data.shape[1])]
            self.pca_feature_names = pca_cols
        # ----------------------------------------------------------------------
        # Phase 2: Transform on Unseen Validation / Test Data
        # ----------------------------------------------------------------------
        else:
            reduced_data = self.pca_transformer.transform(df_clean[cols])
            pca_cols = self.pca_feature_names

        # Convert the NumPy reduced array back to a Pandas DataFrame
        pca_df = pd.DataFrame(reduced_data, columns=pca_cols, index=df_clean.index)

        # Drop the original redundant numerical features
        df_clean = df_clean.drop(columns=cols)

        # Horizontally concatenate the new orthogonal PCA components
        df_clean = pd.concat([df_clean, pca_df], axis=1)

        logger.info(
            f"PCA reduced {len(cols)} features to {len(pca_cols)} components "
            f"(preserving {variance_ratio:.0%} variance)."
        )
        return df_clean


# ------------------------------------------------------------------------------
# Local Verification & Demonstration
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Robust configuration path resolution
    config_file = Path(__file__).resolve().parent.parent / "config" / "O01_feature_rules.yaml"
    if not config_file.exists():
        config_file = Path("config/O01_feature_rules.yaml")

    print("\n" + "=" * 70)
    print(" CARDINALITY REDUCTION & PCA DEMONSTRATION")
    print("=" * 70)

    # Mock dataset with high-cardinality rare labels and collinear numerical features
    sample_df = pd.DataFrame({
        "customer_id": [f"CUST-{i:03d}" for i in range(1, 11)],
        # 'contract_type' contains frequent levels ('m2m', '1-year') and rare levels (< 2%)
        "contract_type": [
            "month-to-month", "month-to-month", "month-to-month", "month-to-month",
            "one-year", "one-year", "one-year",
            "two-year", "custom-enterprise-rare", "trial-promo-rare"
        ],
        # Numerical features suitable for PCA reduction
        "tenure_months": [12, 24, 6, 48, 18, 36, 60, 2, 72, 8],
        "monthly_charges": [29.95, 75.0, 45.0, 89.0, 55.0, 99.0, 110.0, 20.0, 120.0, 30.0],
        "total_charges": [359.4, 1800.0, 270.0, 4272.0, 990.0, 3564.0, 6600.0, 40.0, 8640.0, 240.0]
    })

    print("\n[INPUT] Raw Sample DataFrame:")
    print(sample_df)

    if config_file.exists():
        reducer = CardinalityDimensionalityReducer(config_path=config_file)

        # 1. Rare-category grouping test
        df_grouped = reducer.group_rare_categories(sample_df, is_train=True)
        print("\n[OUTPUT] After Rare-Category Grouping (Rare levels -> 'Other'):")
        print(df_grouped[["customer_id", "contract_type"]])

        # 2. PCA dimensionality reduction test
        df_pca = reducer.apply_pca_reduction(
            df_grouped,
            feature_cols=["tenure_months", "monthly_charges", "total_charges"],
            variance_ratio=0.95,
            is_train=True
        )
        print("\n[OUTPUT] After PCA Dimensionality Reduction:")
        print(df_pca)
    else:
        print(f"[ERROR] Config file not found at: {config_file}")