"""
O06_data_leakage_validator.py
Enforces strict train/test splitting before transformer fitting to prevent global data leakage.
"""
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DataLeakageValidator")

class DataLeakageValidator:
    def __init__(self, test_size:float = 0.2, random_state:int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def stratified_train_test_split(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Splits DataFrame into train and test sets using stratification on target_col."""
        if target_col not in df.columns:
            logger.warning(f"Target '{target_col}' not found. Performing standard non-stratified split.")
            stratify_series = None
        else:
            stratify_series = df[target_col]

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_series
        )
        # Reset indices for clean downstream indexing
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logger.info(f"Dataset split complete: Train shape={train_df.shape}, Test shape={test_df.shape}")
        return train_df, test_df

if __name__ == "__main__":
    pass
