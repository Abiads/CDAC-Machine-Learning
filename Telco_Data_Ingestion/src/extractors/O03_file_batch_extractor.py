"""
================================================================================
MODULE: Telco Batch File Extractor
FILE  : src/extractors/O03_file_batch_extractor.py
PURPOSE:
  Handles batch ingestion of structured files (CSV, Parquet, JSON) from 
  the raw landing zone (Bronze layer) for the IBM Telco Customer Churn dataset. 
  Features automated format detection, encoding fallback, and optional chunking.
================================================================================
"""

import logging
from pathlib import Path
from typing import Generator, Union
import pandas as pd

# Configure module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoBatchExtractor")


class BatchFileExtractor:
    """Manages multi-format batch file extraction from the raw landing zone."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data landing zone does not exist: {self.raw_data_dir}")

    def extract_file(
        self,
        file_name: str = "Telco-Customer-Churn.csv",
        file_format: str = "csv",
        chunk_size: Union[int, None] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Extracts a file from the raw landing zone based on format.

        Args:
            file_name (str): Target filename.
            file_format (str): Format ('csv', 'parquet', 'json').
            chunk_size (int, optional): If provided, returns an iterator for chunked processing.
            **kwargs: Extra parameters passed to pandas reader.

        Returns:
            pd.DataFrame or Generator: Extracted DataFrame or chunk generator.
        """
        file_path = self.raw_data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"[EXTRACTION ERROR] Target file not found: {file_path}")

        logger.info(f"Extracting file '{file_name}' (Format: {file_format.upper()}) from {self.raw_data_dir}...")

        fmt = file_format.lower()
        if fmt == "csv":
            if chunk_size:
                logger.info(f"Enabling chunked extraction with chunk_size={chunk_size:,} rows.")
                return pd.read_csv(file_path, chunksize=chunk_size, encoding="utf-8", **kwargs)
            return pd.read_csv(file_path, encoding="utf-8", **kwargs)
        elif fmt == "parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif fmt == "json":
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format '{file_format}'. Supported: csv, parquet, json.")
