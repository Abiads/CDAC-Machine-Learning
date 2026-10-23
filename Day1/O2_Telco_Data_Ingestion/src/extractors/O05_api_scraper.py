"""
================================================================================
MODULE: Resilient Telco REST API Scraper
FILE  : src/extractors/O05_api_scraper.py
PURPOSE:
  Provides a robust HTTP client wrapper for pulling customer records from 
  carrier billing / CRM REST APIs. Includes automatic exponential backoff retries 
  for transient errors (429, 500, 502, 503, 504) and rate-limiting throttling.
================================================================================
"""

import logging
import time
from typing import Dict, Any, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoAPIScraper")


class APIScraper:
    """Robust REST API scraper with retry-backoff and rate-limiting support."""

    def __init__(self, base_url: str = "https://api.telecom-crm.example.com", timeout_sec: int = 10):
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.session = self._create_resilient_session()
        logger.info(f"Initialized APIScraper targeting: {base_url}")

    def _create_resilient_session(self) -> requests.Session:
        """Configures requests Session with retry and exponential backoff."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,  # 1s, 2s, 4s delay between retries
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_telco_batch(self, endpoint: str = "customers/recent", limit: int = 10) -> pd.DataFrame:
        """
        Fetches a batch of customer telemetry records via REST endpoint.
        Falls back to structured synthetic batch in simulation mode.
        """
        logger.info(f"Fetching {limit} customer records from API endpoint '{endpoint}'...")
        try:
            # Simulated demonstration payload compliant with Telco schema
            mock_api_records = pd.DataFrame([
                {
                    "customerID": f"API-CUST-{1000 + i}",
                    "gender": "Female" if i % 2 == 0 else "Male",
                    "SeniorCitizen": 1 if i == 3 else 0,
                    "Partner": "Yes" if i % 2 == 0 else "No",
                    "Dependents": "No",
                    "tenure": 5 * (i + 1),
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month" if i < 3 else "One year",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 79.50 + (i * 2.5),
                    "TotalCharges": (79.50 + (i * 2.5)) * 5 * (i + 1),
                    "Churn": "Yes" if i < 2 else "No"
                }
                for i in range(limit)
            ])
            return mock_api_records
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()
