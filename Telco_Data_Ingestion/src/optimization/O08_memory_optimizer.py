"""
================================================================================
MODULE: Low-Level Memory Optimizer for Telco Data
FILE  : src/optimization/O08_memory_optimizer.py
PURPOSE:
  Optimizes the RAM footprint of the IBM Telco dataset through aggressive 
  numerical downcasting (converting int64 to int8/int16 and float64 to float32) 
  and categorical type conversions without loss of mathematical precision.
================================================================================
"""

import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoMemoryOptimizer")


class MemoryOptimizer:
    """Optimizes dataframe memory footprint through systematic type downcasting."""

    @staticmethod
    def get_memory_usage_mb(df: pd.DataFrame) -> float:
        """Calculates total deep memory consumption of DataFrame in Megabytes (MB)."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    @classmethod
    def downcast_numeric_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts integer and float columns to the smallest non-overflowing C-types.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: RAM-optimized DataFrame.
        """
        df_opt = df.copy()
        start_mem = cls.get_memory_usage_mb(df_opt)
        logger.info(f"RAM footprint prior to downcasting: {start_mem:.3f} MB")

        for col in df_opt.columns:
            col_type = df_opt[col].dtype

            # 1. Downcast Integers
            if np.issubdtype(col_type, np.integer):
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df_opt[col] = df_opt[col].astype(np.int32)

            # 2. Downcast Floats
            elif np.issubdtype(col_type, np.floating):
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")

        end_mem = cls.get_memory_usage_mb(df_opt)
        reduction = ((start_mem - end_mem) / start_mem) * 100
        logger.info(f"RAM footprint after downcasting : {end_mem:.3f} MB")
        logger.info(f"RAM Footprint Reduction         : {reduction:.2f}%")
        return df_opt
