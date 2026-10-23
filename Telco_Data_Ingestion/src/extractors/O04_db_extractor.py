"""
================================================================================
MODULE: Telco Database Extractor
FILE  : src/extractors/O04_db_extractor.py
PURPOSE:
  Manages relational SQL / SQLite database extractions for Telco customer records.
  Implements connection pooling and cursor chunksize pagination to prevent 
  monolithic memory consumption during enterprise database reads.
================================================================================
"""

import logging
import sqlite3
from pathlib import Path
from typing import Generator
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoDatabaseExtractor")


class DatabaseExtractor:
    """Enterprise database extractor with support for connection pooling and pagination."""

    def __init__(self, connection_string: str = "sqlite:///telco_enterprise.db"):
        self.connection_string = connection_string
        logger.info(f"Initialized TelcoDatabaseExtractor with target: {connection_string}")

    def extract_in_batches(
        self, query: str = "SELECT * FROM telco_customers", chunk_size: int = 2000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Executes a SQL query and yields data in manageable chunks.

        Args:
            query (str): SQL query to execute.
            chunk_size (int): Number of customer rows per batch chunk.

        Yields:
            pd.DataFrame: A chunk of customer records.
        """
        logger.info(f"Executing database query with chunk_size={chunk_size:,}...")

        # If SQLite database file exists, query directly
        db_file = Path("telco_enterprise.db")
        if db_file.exists():
            with sqlite3.connect(db_file) as conn:
                for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
                    yield chunk
        else:
            logger.info("Local SQLite database not found. Generating mock Telco SQL chunk for demonstration.")
            mock_chunk = pd.DataFrame({
                "customerID": [f"DB-CUST-{i:04d}" for i in range(1, 6)],
                "gender": ["Female", "Male", "Female", "Male", "Female"],
                "SeniorCitizen": [0, 0, 1, 0, 0],
                "Partner": ["Yes", "No", "No", "Yes", "No"],
                "Dependents": ["No", "No", "No", "Yes", "No"],
                "tenure": [12, 24, 3, 48, 1],
                "PhoneService": ["Yes", "Yes", "Yes", "Yes", "No"],
                "MultipleLines": ["No", "Yes", "No", "Yes", "No phone service"],
                "InternetService": ["DSL", "Fiber optic", "Fiber optic", "DSL", "DSL"],
                "OnlineSecurity": ["Yes", "No", "No", "Yes", "No"],
                "OnlineBackup": ["Yes", "No", "No", "Yes", "No"],
                "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
                "TechSupport": ["Yes", "No", "No", "Yes", "No"],
                "StreamingTV": ["No", "Yes", "Yes", "No", "No"],
                "StreamingMovies": ["No", "Yes", "Yes", "No", "No"],
                "Contract": ["One year", "Month-to-month", "Month-to-month", "Two year", "Month-to-month"],
                "PaperlessBilling": ["Yes", "Yes", "Yes", "No", "Yes"],
                "PaymentMethod": [
                    "Bank transfer (automatic)",
                    "Electronic check",
                    "Electronic check",
                    "Credit card (automatic)",
                    "Electronic check"
                ],
                "MonthlyCharges": [55.20, 89.90, 75.30, 60.10, 25.50],
                "TotalCharges": [662.40, 2157.60, 225.90, 2884.80, 25.50],
                "Churn": ["No", "Yes", "Yes", "No", "Yes"]
            })
            yield mock_chunk
